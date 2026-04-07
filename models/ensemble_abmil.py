"""
ABMIL wrapper for TCR repertoire classification.

Per-sequence features are produced end-to-end by TCRSeqEncoder (learned AA
embedding + 3-layer 1-D conv + learned V/J gene embeddings), then aggregated
by TCRGatedAttentionMIL.

Reference architecture: Ilse et al. 2018, "Attention-based Deep Multiple Instance
Learning" (https://arxiv.org/abs/1802.04712).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.repertoire_io import load_raw_repertoire

# 20 standard amino acids (alphabetical); index 0 is reserved for padding / unknown.
_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {aa: i + 1 for i, aa in enumerate(_AA_VOCAB)}


class _RepertoireBase:
    def __init__(
        self,
        sequence_col,
        v_gene_col,
        j_gene_col,
        subsample_fraction,
        subsample_seed,
        subsample_n,
        indices_map,
        ignore_allele,
    ):
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map
        self.ignore_allele = ignore_allele
        self._repertoire_cache = {}

    def load_repertoire(self, file_path, use_cache=True, apply_subsampling=True):
        cache_key = (file_path, apply_subsampling)
        if use_cache and cache_key in self._repertoire_cache:
            return self._repertoire_cache[cache_key]

        indices = None
        if self.indices_map is not None:
            rep_id = os.path.basename(file_path).replace(".tsv.gz", "").replace(".tsv", "")
            indices = self.indices_map.get(rep_id)

        if apply_subsampling:
            subsample_n = self.subsample_n
            subsample_fraction = self.subsample_fraction
            subsample_seed = self.subsample_seed
        else:
            subsample_n = None
            subsample_fraction = 1.0
            subsample_seed = self.subsample_seed

        df = load_raw_repertoire(
            file_path,
            subsample_n,
            subsample_fraction,
            subsample_seed,
            subsample_indices=indices,
        )

        if use_cache:
            self._repertoire_cache[cache_key] = df
        return df

    def _normalize_gene(self, gene):
        if self.ignore_allele and isinstance(gene, str):
            gene = gene.split("*")[0]
        return gene


class TCRGatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL for TCR repertoire classification.

    Input to forward():
        H: FloatTensor of shape (K, input_dim) — K sequence-level feature vectors.

    Architecture:
        encoder:     Linear(input_dim, M) → ReLU → Dropout
        attention:   gated mechanism (tanh branch × sigmoid branch → Linear → softmax)
        classifier:  Linear(M, 1) → Sigmoid
    """

    def __init__(self, input_dim, M=128, L=64, dropout=0.25):
        super().__init__()
        self.M = M
        self.L = L
        self._dropout_p = dropout

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, M),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
        self.attention_w = nn.Linear(L, 1)
        self.classifier = nn.Sequential(nn.Linear(M, 1), nn.Sigmoid())

    def forward(self, H):
        H = self.encoder(H)  # (K, M)
        A = self.attention_w(self.attention_V(H) * self.attention_U(H))  # (K, 1)
        A = F.softmax(A.transpose(1, 0), dim=1)  # (1, K)
        Z = torch.mm(A, H)  # (1, M)
        Y_prob = self.classifier(Z)  # (1, 1)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, H, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(H)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, H, Y, pos_weight=1.0):
        Y = Y.float()
        Y_prob, _, A = self.forward(H)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            pos_weight * Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )
        return neg_log_likelihood, A


class TCRSeqEncoder(nn.Module):
    """
    Per-sequence feature extractor for TCR repertoire classification.

    Sequence branch (used when features is 'full' or 'cdr3_only'):
        aa_idx (0=pad, 1-20=standard AA letters) → Embedding(21, embedding_dim_aa)
        → Conv1d(embedding_dim_aa, C0, kernel_size=kernel, padding=kernel//2) → LeakyReLU → dropout
        → Conv1d(C0, C1, kernel_size=3, stride=3, padding=1)                 → LeakyReLU → dropout
        → Conv1d(C1, C2, kernel_size=3, stride=3, padding=1)                 → LeakyReLU → dropout
        → global max-pool over positions → (K, C2)

    Gene branches (used when features is 'full' or 'vj_only'):
        v_idx → Embedding(n_v_genes, embedding_dim_genes) → (K, embedding_dim_genes)
        j_idx → Embedding(n_j_genes, embedding_dim_genes) → (K, embedding_dim_genes)

    Output (output_dim varies by features mode):
        'full':      concat(seq_features, v_features, j_features)  → C2 + 2*embedding_dim_genes
        'cdr3_only': seq_features                                   → C2
        'vj_only':   concat(v_features, j_features)                → 2*embedding_dim_genes

    AA encoding uses _AA_IDX (20 standard AAs → 1-20; 0 = pad/unknown).
    """

    def __init__(
        self,
        n_v_genes,
        n_j_genes,
        embedding_dim_aa=64,
        embedding_dim_genes=48,
        kernel=5,
        dropout=0.25,
        conv_units=(32, 64, 128),
        features="full",
    ):
        super().__init__()
        self.dropout_p = dropout
        self.features = features

        if features not in ("full", "cdr3_only", "vj_only"):
            raise ValueError(f"features must be 'full', 'cdr3_only', or 'vj_only'; got '{features}'")

        if features in ("full", "cdr3_only"):
            # +1 because indices are 0..20 inclusive:
            # 0 = padding/unknown, 1..20 = standard amino acids
            self.aa_embedding = nn.Embedding(len(_AA_IDX) + 1, embedding_dim_aa, padding_idx=0)
            self.conv1 = nn.Conv1d(
                embedding_dim_aa,
                conv_units[0],
                kernel_size=kernel,
                padding=kernel // 2,
            )
            self.conv2 = nn.Conv1d(
                conv_units[0],
                conv_units[1],
                kernel_size=3,
                stride=3,
                padding=1,
            )
            self.conv3 = nn.Conv1d(
                conv_units[1],
                conv_units[2],
                kernel_size=3,
                stride=3,
                padding=1,
            )

        if features in ("full", "vj_only"):
            self.v_embedding = nn.Embedding(n_v_genes, embedding_dim_genes)
            self.j_embedding = nn.Embedding(n_j_genes, embedding_dim_genes)

        if features == "full":
            self.output_dim = conv_units[2] + embedding_dim_genes * 2
        elif features == "cdr3_only":
            self.output_dim = conv_units[2]
        else:  # vj_only
            self.output_dim = embedding_dim_genes * 2

    def forward(self, seq_idx, v_idx, j_idx):
        """
        Args:
            seq_idx: LongTensor (K, max_length)
            v_idx:   LongTensor (K,)
            j_idx:   LongTensor (K,)

        Returns:
            FloatTensor (K, output_dim)
        """
        parts = []

        if self.features in ("full", "cdr3_only"):
            x = self.aa_embedding(seq_idx).transpose(1, 2)  # (K, embedding_dim_aa, L)
            x = F.dropout(F.leaky_relu(self.conv1(x)), p=self.dropout_p, training=self.training)
            x = F.dropout(F.leaky_relu(self.conv2(x)), p=self.dropout_p, training=self.training)
            x = F.dropout(F.leaky_relu(self.conv3(x)), p=self.dropout_p, training=self.training)
            parts.append(x.max(dim=2).values)  # (K, C2)

        if self.features in ("full", "vj_only"):
            parts.append(self.v_embedding(v_idx))
            parts.append(self.j_embedding(j_idx))

        return torch.cat(parts, dim=1)


class ABMIL(_RepertoireBase):
    """
    Gated Attention MIL classifier for TCR repertoire classification.

    Each repertoire is treated as a bag of sequences. Per-sequence features are
    produced end-to-end by TCRSeqEncoder (learned AA embedding + 3-layer 1-D
    conv + learned V/J gene embeddings), then aggregated by TCRGatedAttentionMIL.

    AA characters are encoded as _AA_IDX (0 = pad/unknown).
    V/J gene names are mapped to integer indices via vocabularies built from the
    training data (0 = unknown).

    Training uses Adam with early stopping on a held-out validation split of bags.
    """

    def __init__(
        self,
        max_instances=10000,
        M=128,
        L=64,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        patience=10,
        val_split=0.2,
        seed=7,
        sequence_col="cdr3_aa",
        v_gene_col="v_call",
        j_gene_col="j_call",
        subsample_fraction=1.0,
        subsample_seed=7,
        subsample_n=None,
        indices_map=None,
        ignore_allele=False,
        use_gpu=True,
        dropout=0.25,
        max_length=40,
        embedding_dim_aa=64,
        embedding_dim_genes=48,
        kernel=5,
        conv_units=(32, 64, 128),
        features="full",
    ):
        """
        Args:
            max_instances: Sequences randomly subsampled per bag per training epoch
                (augmentation). Evaluation always uses all sequences. None = no limit.
            M: ABMIL hidden dimension.
            L: ABMIL attention hidden dimension.
            epochs: Maximum training epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay (L2 regularisation).
            dropout: Dropout probability in both the conv encoder and ABMIL encoder.
            patience: Early-stopping patience (epochs without val-loss improvement).
            val_split: Fraction of training bags held out for early stopping.
            seed: Random seed for val split and epoch subsampling.
            sequence_col: Column containing CDR3 amino-acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls.
            subsample_fraction: Fraction of reads to sample per repertoire when loading
                training bags from disk.
            subsample_seed: Random seed for repertoire subsampling.
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction).
            indices_map: Dict mapping rep_id to pre-computed row indices.
            ignore_allele: Strip allele designations from V/J gene names.
            use_gpu: Use CUDA if available.
            max_length: Maximum CDR3 length; longer sequences are truncated.
            embedding_dim_aa: Learned AA embedding dimension.
            embedding_dim_genes: Learned V/J gene embedding dimension.
            kernel: First conv kernel size.
            conv_units: Output channels for the three conv layers.
            features: Which features to use — 'full' (CDR3 + V/J genes),
                'cdr3_only' (CDR3 sequence only), or 'vj_only' (V/J gene identities only).
        """
        super().__init__(
            sequence_col=sequence_col,
            v_gene_col=v_gene_col,
            j_gene_col=j_gene_col,
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
            subsample_n=subsample_n,
            indices_map=indices_map,
            ignore_allele=ignore_allele,
        )
        self.max_instances = max_instances
        self.M = M
        self.L = L
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.seed = seed
        self.use_gpu = use_gpu
        self.dropout = dropout
        self.max_length = max_length
        self.embedding_dim_aa = embedding_dim_aa
        self.embedding_dim_genes = embedding_dim_genes
        self.kernel = kernel
        self.conv_units = tuple(conv_units)
        self.features = features

        # Set after train()
        self.v_vocab = None  # dict: gene_name -> int index (0 = unknown)
        self.j_vocab = None
        self.encoder = None  # TCRSeqEncoder
        self.model = None  # TCRGatedAttentionMIL
        self.device = None

    def _encode_seq(self, seq):
        """Encode a CDR3 string to a zero-padded integer array of length max_length."""
        arr = np.zeros(self.max_length, dtype=np.int64)
        for i, c in enumerate(seq[: self.max_length]):
            arr[i] = _AA_IDX.get(c, 0)
        return arr

    def _encode_v(self, gene):
        if not isinstance(gene, str) or not gene:
            return 0
        return self.v_vocab.get(self._normalize_gene(gene), 0)

    def _encode_j(self, gene):
        if not isinstance(gene, str) or not gene:
            return 0
        return self.j_vocab.get(self._normalize_gene(gene), 0)

    def _get_per_seq_arrays(self, file_path, apply_subsampling=True):
        """Load a repertoire and encode all sequences + V/J genes as integer arrays.

        No sequences are filtered out. Sequences longer than max_length are
        truncated; unknown/non-standard characters map to 0 (same as padding);
        null sequences become all-zero rows.

        Returns:
            seq_arr: np.int64 (K, max_length)
            v_arr:   np.int64 (K,)
            j_arr:   np.int64 (K,)
        """
        df = self.load_repertoire(file_path, apply_subsampling=apply_subsampling)
        K = len(df)

        if K == 0:
            return (
                np.zeros((0, self.max_length), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )

        # Encode sequences — truncate at max_length, unknown chars → 0
        seq_arr = np.zeros((K, self.max_length), dtype=np.int64)
        if self.sequence_col in df.columns:
            for k, seq in enumerate(df[self.sequence_col]):
                if isinstance(seq, str):
                    for i, c in enumerate(seq[: self.max_length]):
                        seq_arr[k, i] = _AA_IDX.get(c, 0)

        # Encode V genes — unknown genes → 0
        v_arr = np.zeros(K, dtype=np.int64)
        if self.v_gene_col in df.columns:
            for k, gene in enumerate(df[self.v_gene_col]):
                v_arr[k] = self._encode_v(gene)

        # Encode J genes — unknown genes → 0
        j_arr = np.zeros(K, dtype=np.int64)
        if self.j_gene_col in df.columns:
            for k, gene in enumerate(df[self.j_gene_col]):
                j_arr[k] = self._encode_j(gene)

        return seq_arr, v_arr, j_arr

    def _to_tensors(self, seq_arr, v_arr, j_arr, row_inds=None):
        if row_inds is not None:
            seq_arr = seq_arr[row_inds]
            v_arr = v_arr[row_inds]
            j_arr = j_arr[row_inds]

        return (
            torch.from_numpy(seq_arr).long().to(self.device),
            torch.from_numpy(v_arr).long().to(self.device),
            torch.from_numpy(j_arr).long().to(self.device),
        )

    def train(self, train_files, train_labels):
        """
        Train the ABMIL model end-to-end.

        Args:
            train_files: List of repertoire file paths.
            train_labels: List/array of binary labels (0 = healthy, 1 = disease).

        Returns:
            Dict with 'best_val_loss' and 'epochs_trained'.
        """
        train_files = list(train_files)
        train_labels = np.array(train_labels)

        self.device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # --- Build V/J gene vocabularies from training data ---
        print("Scanning training bags for gene vocabulary...")
        v_genes, j_genes = set(), set()
        for fp in tqdm(train_files, desc="Scanning bags"):
            df = self.load_repertoire(fp, apply_subsampling=True)
            if self.v_gene_col in df.columns:
                for g in df[self.v_gene_col].dropna():
                    if isinstance(g, str) and g:
                        v_genes.add(self._normalize_gene(g))
            if self.j_gene_col in df.columns:
                for g in df[self.j_gene_col].dropna():
                    if isinstance(g, str) and g:
                        j_genes.add(self._normalize_gene(g))

        # index 0 is reserved for unknown genes
        self.v_vocab = {g: i + 1 for i, g in enumerate(sorted(v_genes))}
        self.j_vocab = {g: i + 1 for i, g in enumerate(sorted(j_genes))}
        n_v = len(self.v_vocab) + 1
        n_j = len(self.j_vocab) + 1
        print(f"V-gene vocabulary: {n_v} classes  |  J-gene vocabulary: {n_j} classes")

        # --- Encode all bags as integer arrays ---
        # Training bags may use repertoire-level subsampling if configured.
        print("Encoding bags as integer arrays...")
        train_bag_arrays = [
            self._get_per_seq_arrays(fp, apply_subsampling=True)
            for fp in tqdm(train_files, desc="Encoding bags")
        ]

        # Validation bags should use all sequences.
        val_bag_arrays = [
            self._get_per_seq_arrays(fp, apply_subsampling=False)
            for fp in tqdm(train_files, desc="Encoding full bags for validation")
        ]

        # --- Train / val split by bag ---
        n = len(train_files)
        tr_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=self.val_split,
            random_state=self.seed,
            stratify=train_labels,
        )

        n_pos = int(train_labels[tr_idx].sum())
        n_neg = len(tr_idx) - n_pos
        pos_weight = n_neg / max(n_pos, 1)
        print(f"Training set: {n_pos} positive, {n_neg} negative  (pos_weight={pos_weight:.3f})")

        # --- Build encoder + ABMIL model ---
        self.encoder = TCRSeqEncoder(
            n_v_genes=n_v,
            n_j_genes=n_j,
            embedding_dim_aa=self.embedding_dim_aa,
            embedding_dim_genes=self.embedding_dim_genes,
            kernel=self.kernel,
            dropout=self.dropout,
            conv_units=self.conv_units,
            features=self.features,
        ).to(self.device)

        self.model = TCRGatedAttentionMIL(
            input_dim=self.encoder.output_dim,
            M=self.M,
            L=self.L,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # --- Training loop with early stopping ---
        rng = np.random.RandomState(self.seed)
        best_val_loss = float("inf")
        patience_ctr = 0
        best_enc_state = None
        best_mil_state = None
        epochs_trained = 0

        for epoch in range(1, self.epochs + 1):
            epochs_trained = epoch
            epoch_rng = np.random.RandomState(self.seed + epoch)

            self.encoder.train()
            self.model.train()
            train_loss = 0.0
            n_train_used = 0

            for i in rng.permutation(tr_idx):
                seq_arr, v_arr, j_arr = train_bag_arrays[i]
                K = seq_arr.shape[0]
                if K == 0:
                    continue

                row_inds = None
                if self.max_instances is not None and K > self.max_instances:
                    row_inds = np.sort(epoch_rng.choice(K, self.max_instances, replace=False))

                seq_t, v_t, j_t = self._to_tensors(seq_arr, v_arr, j_arr, row_inds)
                H = self.encoder(seq_t, v_t, j_t)
                y = torch.tensor([[train_labels[i]]], dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                loss, _ = self.model.calculate_objective(H, y, pos_weight=pos_weight)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train_used += 1

            # Validation — all sequences, no repertoire-level subsampling
            self.encoder.eval()
            self.model.eval()
            val_loss = 0.0
            n_val_used = 0

            with torch.no_grad():
                for i in val_idx:
                    seq_arr, v_arr, j_arr = val_bag_arrays[i]
                    if seq_arr.shape[0] == 0:
                        continue

                    seq_t, v_t, j_t = self._to_tensors(seq_arr, v_arr, j_arr)
                    H = self.encoder(seq_t, v_t, j_t)
                    y = torch.tensor([[train_labels[i]]], dtype=torch.float32, device=self.device)
                    loss, _ = self.model.calculate_objective(H, y, pos_weight=pos_weight)

                    val_loss += loss.item()
                    n_val_used += 1

            mean_train_loss = train_loss / max(n_train_used, 1)
            mean_val_loss = val_loss / max(n_val_used, 1)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"train_loss={mean_train_loss:.4f}  "
                    f"val_loss={mean_val_loss:.4f}"
                )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_ctr = 0
                best_enc_state = {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}
                best_mil_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_enc_state is not None:
            self.encoder.load_state_dict(best_enc_state)
            self.model.load_state_dict(best_mil_state)

        return {"best_val_loss": best_val_loss, "epochs_trained": epochs_trained}

    def predict_diagnosis(self, file_path):
        """
        Predict disease probability for a single repertoire.

        Args:
            file_path: Path to a repertoire .tsv / .tsv.gz file.

        Returns:
            Dict with 'probability_positive' (float) and 'diagnosis' (str).
        """
        if self.encoder is None or self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        self.encoder.eval()
        self.model.eval()

        # Prediction uses all sequences, not repertoire-level subsampling.
        seq_arr, v_arr, j_arr = self._get_per_seq_arrays(file_path, apply_subsampling=False)
        if seq_arr.shape[0] == 0:
            return {"probability_positive": 0.5, "diagnosis": "Healthy"}

        seq_t, v_t, j_t = self._to_tensors(seq_arr, v_arr, j_arr)
        with torch.no_grad():
            H = self.encoder(seq_t, v_t, j_t)
            Y_prob, _, _ = self.model(H)

        prob = float(Y_prob.squeeze().cpu().item())
        return {
            "probability_positive": prob,
            "diagnosis": "Diseased" if prob >= 0.5 else "Healthy",
        }
