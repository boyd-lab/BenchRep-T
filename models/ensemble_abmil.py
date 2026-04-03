"""
ABMIL wrapper for TCR repertoire classification using gapped 4-mer + V/J gene features.

Reuses per-sequence feature extraction from ensemble_regression.py and applies
Gated Attention MIL (TCRGatedAttentionMIL) to treat each repertoire as a bag of
sequence-level instances.

Reference architecture: Ilse et al. 2018, "Attention-based Deep Multiple Instance
Learning" (https://arxiv.org/abs/1802.04712).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.ensemble_regression import _extract_gapped_4mers, Gapped_4mer_VJgene


class TCRGatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL for TCR repertoire classification.

    Replaces the CNN feature extractor from the original MNIST model with a
    linear encoder so that pre-computed dense instance features (one vector per
    sequence) can be used directly.

    Input to forward():
        H: FloatTensor of shape (K, input_dim) — K sequence-level feature vectors.

    Architecture:
        encoder:     Linear(input_dim, M) → ReLU
        attention:   gated mechanism (tanh branch × sigmoid branch → Linear → softmax)
        classifier:  Linear(M, 1) → Sigmoid
    """

    def __init__(self, input_dim, M=256, L=128):
        super(TCRGatedAttentionMIL, self).__init__()
        self.M = M
        self.L = L

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(M, L),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(M, L),
            nn.Sigmoid(),
        )

        self.attention_w = nn.Linear(L, 1)

        self.classifier = nn.Sequential(
            nn.Linear(M, 1),
            nn.Sigmoid(),
        )

    def forward(self, H):
        # H: (K, input_dim)
        H = self.encoder(H)           # (K, M)

        A_V = self.attention_V(H)     # (K, L)
        A_U = self.attention_U(H)     # (K, L)
        A = self.attention_w(A_V * A_U)        # (K, 1)
        A = torch.transpose(A, 1, 0)           # (1, K)
        A = F.softmax(A, dim=1)                # (1, K)

        Z = torch.mm(A, H)            # (1, M)

        Y_prob = self.classifier(Z)   # (1, 1)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, H, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(H)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, H, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(H)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (
            Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)
        )
        return neg_log_likelihood, A


class ABMIL_4mer_VJgene(Gapped_4mer_VJgene):
    """
    Gated Attention MIL classifier over per-sequence gapped 4-mer + V/J gene features.

    Each repertoire is treated as a bag of sequence instances. Per-sequence feature
    vectors are built by combining gapped 4-mer counts with V/J gene indicators
    (reusing feature extraction from Gapped_4mer_VJgene). A DictVectorizer + scaler
    fitted on all training sequences maps these sparse dicts to dense vectors, which
    are then processed by TCRGatedAttentionMIL.

    Training uses Adam with early stopping on a held-out validation split of bags.
    """

    def __init__(
        self,
        max_instances=500,
        M=256,
        L=128,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-5,
        patience=10,
        val_split=0.2,
        seed=7,
        sequence_col='cdr3_aa',
        v_gene_col='v_call',
        j_gene_col='j_call',
        subsample_fraction=1.0,
        subsample_seed=7,
        subsample_n=None,
        indices_map=None,
        ignore_allele=False,
        use_gpu=True,
    ):
        """
        Args:
            max_instances: Maximum sequences to sample per bag (memory / regularisation).
            M: ABMIL hidden dimension (encoder output and bag embedding size).
            L: ABMIL attention hidden dimension.
            epochs: Maximum training epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay (L2 regularisation).
            patience: Early-stopping patience (epochs without val-loss improvement).
            val_split: Fraction of training bags held out for early stopping.
            seed: Random seed for val split and bag subsampling.
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls.
            subsample_fraction: Fraction of reads to sample per repertoire.
            subsample_seed: Random seed for repertoire subsampling.
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction).
            indices_map: Dict mapping rep_id to pre-computed row indices.
            ignore_allele: If True, strip allele designations from V/J gene names.
            use_gpu: Use CUDA if available.
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

        # Set after train()
        self.vectorizer = None
        self.scaler = None
        self.model = None
        self.device = None

    # ------------------------------------------------------------------
    # Per-sequence feature extraction
    # ------------------------------------------------------------------

    def _get_per_seq_feature_dicts(self, file_path):
        """
        Extract per-sequence combined gapped 4-mer + V/J gene feature dicts.

        Subsamples to max_instances sequences if the repertoire is larger.

        Returns:
            List of dicts, one per sequence. Each dict has keys prefixed
            with 'K:' for k-mer features and 'V:'/'J:' for gene features.
        """
        df = self.load_repertoire(file_path)
        if len(df) > self.max_instances:
            rng = np.random.RandomState(self.subsample_seed)
            df = df.sample(n=self.max_instances, random_state=rng).reset_index(drop=True)

        has_v = self.v_gene_col in df.columns
        has_j = self.j_gene_col in df.columns

        feature_dicts = []
        for _, row in df.iterrows():
            d = {}

            seq = row[self.sequence_col] if self.sequence_col in df.columns else None
            if isinstance(seq, str) and seq:
                for kmer in _extract_gapped_4mers(seq):
                    key = f'K:{kmer}'
                    d[key] = d.get(key, 0) + 1

            if has_v:
                v_gene = row[self.v_gene_col]
                if isinstance(v_gene, str) and v_gene:
                    v_gene = self._normalize_gene(v_gene)
                    d[f'V:{v_gene}'] = 1

            if has_j:
                j_gene = row[self.j_gene_col]
                if isinstance(j_gene, str) and j_gene:
                    j_gene = self._normalize_gene(j_gene)
                    d[f'J:{j_gene}'] = 1

            feature_dicts.append(d)

        return feature_dicts

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _build_bag_tensor(self, feature_dicts):
        """
        Vectorize and scale a list of per-sequence feature dicts into a
        FloatTensor of shape (K, input_dim) on self.device.

        Returns None if feature_dicts is empty.
        """
        if not feature_dicts:
            return None
        X = self.vectorizer.transform(feature_dicts)
        X = self.scaler.transform(X)
        return torch.FloatTensor(X.toarray()).to(self.device)

    # ------------------------------------------------------------------
    # Training and inference
    # ------------------------------------------------------------------

    def train(self, train_files, train_labels):
        """
        Train the ABMIL model.

        Args:
            train_files: List of repertoire file paths.
            train_labels: List/array of binary labels (0 = healthy, 1 = disease).

        Returns:
            Dict with 'best_val_loss' and 'epochs_trained'.
        """
        train_files = list(train_files)
        train_labels = np.array(train_labels)

        self.device = torch.device(
            'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        )

        # --- Extract per-sequence feature dicts from all training bags ---
        print("Extracting per-sequence features from training bags...")
        bag_seq_dicts = []
        all_seq_dicts = []
        for fp in tqdm(train_files, desc="Loading bags"):
            dicts = self._get_per_seq_feature_dicts(fp)
            bag_seq_dicts.append(dicts)
            all_seq_dicts.extend(dicts)

        # --- Fit vectorizer and scaler on all training sequences ---
        print(f"Fitting vectorizer on {len(all_seq_dicts)} sequences...")
        self.vectorizer = DictVectorizer(sparse=True)
        X_all = self.vectorizer.fit_transform(all_seq_dicts)
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(X_all)
        input_dim = X_all.shape[1]
        print(f"Feature dimension: {input_dim}")

        # --- Build and vectorize all bag tensors ---
        print("Vectorizing bags...")
        bag_tensors = [
            self._build_bag_tensor(dicts)
            for dicts in tqdm(bag_seq_dicts, desc="Building tensors")
        ]

        # --- Train / val split by bag ---
        n = len(train_files)
        tr_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=self.val_split,
            random_state=self.seed,
            stratify=train_labels,
        )

        # --- Build ABMIL model ---
        self.model = TCRGatedAttentionMIL(
            input_dim=input_dim, M=self.M, L=self.L
        ).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # --- Training loop with early stopping ---
        rng = np.random.RandomState(self.seed)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        epochs_trained = 0

        for epoch in range(1, self.epochs + 1):
            epochs_trained = epoch
            self.model.train()
            train_loss = 0.0
            for i in rng.permutation(tr_idx):
                H = bag_tensors[i]
                if H is None or H.shape[0] == 0:
                    continue
                y = torch.tensor(
                    [[train_labels[i]]], dtype=torch.float32
                ).to(self.device)
                optimizer.zero_grad()
                loss, _ = self.model.calculate_objective(H, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in val_idx:
                    H = bag_tensors[i]
                    if H is None or H.shape[0] == 0:
                        continue
                    y = torch.tensor(
                        [[train_labels[i]]], dtype=torch.float32
                    ).to(self.device)
                    loss, _ = self.model.calculate_objective(H, y)
                    val_loss += loss.item()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"train_loss={train_loss / max(len(tr_idx), 1):.4f}  "
                    f"val_loss={val_loss / max(len(val_idx), 1):.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {'best_val_loss': best_val_loss, 'epochs_trained': epochs_trained}

    def predict_diagnosis(self, file_path):
        """
        Predict disease probability for a single repertoire.

        Args:
            file_path: Path to a repertoire .tsv / .tsv.gz file.

        Returns:
            Dict with 'probability_positive' (float) and 'diagnosis' (str).
        """
        self.model.eval()
        dicts = self._get_per_seq_feature_dicts(file_path)
        H = self._build_bag_tensor(dicts)
        if H is None or H.shape[0] == 0:
            return {'probability_positive': 0.5, 'diagnosis': 'Healthy'}
        with torch.no_grad():
            Y_prob, _, _ = self.model(H)
        prob = float(Y_prob.squeeze().cpu().item())
        return {
            'probability_positive': prob,
            'diagnosis': 'Diseased' if prob >= 0.5 else 'Healthy',
        }
