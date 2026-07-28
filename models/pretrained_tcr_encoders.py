"""Frozen pretrained sequence encoders used by ABMIL.

The adapters expose NumPy arrays rather than trainable parameters. ABMIL
caches these representations and trains its attention and gene-embedding
layers without repeatedly running the pretrained model every epoch.
"""

import csv
import os
from pathlib import Path
import warnings

import numpy as np


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_FROM_INDEX = np.array([""] + list(AA_VOCAB), dtype=object)

DEFAULT_ESM2_MODEL = "facebook/esm2_t30_150M_UR50D"
DEFAULT_TCRBERT_MODEL = "wukevin/tcr-bert"
DEFAULT_TCRVALID_MODEL = "1_2_full_40"


def arrays_to_sequences(seq_idx):
    """Convert ABMIL padded amino-acid indices back to CDR3 strings."""
    sequences = []
    for row in np.asarray(seq_idx):
        sequences.append("".join(AA_FROM_INDEX[row[row > 0]]))
    return sequences


def _mean_pool(hidden, attention_mask, special_tokens_mask):
    residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
    residue_mask = residue_mask.unsqueeze(-1)
    denominator = residue_mask.sum(dim=1).clamp(min=1)
    return (hidden * residue_mask).sum(dim=1) / denominator


class HuggingFaceSequenceEncoder:
    """Residue-mean ESM-2 or eighth-layer TCR-BERT representations."""

    def __init__(self, encoder_name, model_name, device, batch_size):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Pretrained ABMIL encoders require transformers. Install the "
                "project's pretrained_abmil extra."
            ) from exc

        self.encoder_name = encoder_name
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, add_pooling_layer=False
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.output_dim = int(self.model.config.hidden_size)
        self.cache_id = f"{encoder_name}:{model_name}"

    def encode(
        self, seq_idx, _v_idx=None, _inverse_v_vocab=None, _raw_v_genes=None
    ):
        import torch

        sequences = arrays_to_sequences(seq_idx)
        chunks = []
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            if self.encoder_name == "tcrbert":
                batch = [" ".join(sequence) for sequence in batch]
            tokens = self.tokenizer(
                batch,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = tokens.pop("special_tokens_mask")
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.inference_mode():
                outputs = self.model(
                    **tokens,
                    output_hidden_states=self.encoder_name == "tcrbert",
                )
                # hidden_states[0] is the input embedding; index 8 is layer 8.
                hidden = (
                    outputs.hidden_states[8]
                    if self.encoder_name == "tcrbert"
                    else outputs.last_hidden_state
                )
                pooled = _mean_pool(
                    hidden,
                    tokens["attention_mask"],
                    special_tokens_mask.to(self.device),
                )
            chunks.append(pooled.float().cpu().numpy().astype(np.float16))
        if not chunks:
            return np.empty((0, self.output_dim), dtype=np.float16)
        return np.concatenate(chunks, axis=0)


_MEILER_FEATURES = {
    "A": [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
    "C": [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    "D": [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
    "E": [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    "F": [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
    "G": [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
    "H": [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
    "I": [4.19, 0.19, 4.00, 1.80, 6.04, 0.10, 0.45],
    "K": [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    "L": [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
    "M": [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    "N": [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
    "P": [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
    "Q": [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    "S": [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    "T": [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
    "V": [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
    "W": [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    "Y": [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
}


def _standardized_meiler_features():
    chars = sorted(_MEILER_FEATURES)
    values = np.asarray([_MEILER_FEATURES[char] for char in chars], dtype=np.float32)
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    features = {
        char: np.concatenate([row, np.zeros(1, dtype=np.float32)])
        for char, row in zip(chars, values)
    }
    features["-"] = np.asarray([0.0] * 7 + [1.0], dtype=np.float32)
    return features


def tcrvalid_inputs(cdr3_sequences, v_genes, reference_csv, max_length=28):
    """Build TCR-VALID's standardized Meiler CDR2-CDR3 input tensor."""
    with open(reference_csv, newline="") as handle:
        cdr2_by_v = {
            row["new_meta_vcall"]: row["cdr2_no_gaps"]
            for row in csv.DictReader(handle)
        }
    features = _standardized_meiler_features()
    output = np.zeros((len(cdr3_sequences), max_length, 8), dtype=np.float32)
    output[:, :, -1] = 1.0
    missing = set()
    for index, (cdr3, v_gene) in enumerate(zip(cdr3_sequences, v_genes)):
        normalized_v = v_gene.split("*")[0] if v_gene else ""
        cdr2 = cdr2_by_v.get(normalized_v)
        if cdr2 is None:
            missing.add(normalized_v or "<unknown>")
            cdr2 = "------"
        combined = f"{cdr2}-{cdr3}"[:max_length]
        for position, amino_acid in enumerate(combined):
            output[index, position] = features.get(amino_acid, features["-"])
    if missing:
        warnings.warn(
            "TCR-VALID used a gapped CDR2 for unrecognized V genes: "
            + ", ".join(sorted(missing)),
            RuntimeWarning,
        )
    return output


class TCRValidSequenceEncoder:
    """Released 16-dimensional TCR-VALID TRB encoder."""

    def __init__(self, repo_dir, model_name, batch_size):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TCR-VALID ABMIL requires tensorflow. Install the project's "
                "pretrained_abmil extra."
            ) from exc

        repo_dir = Path(repo_dir).resolve()
        package_dir = repo_dir / "tcrvalid"
        self.reference_csv = package_dir / "data" / "TRBV_reference.csv"
        self.model_path = (
            package_dir / "logged_models" / "TRB" / model_name / "encoder_kr"
        )
        if not self.reference_csv.exists() or not self.model_path.exists():
            raise FileNotFoundError(
                f"Invalid TCR-VALID checkout at {repo_dir}; expected "
                f"{self.reference_csv} and {self.model_path}"
            )
        use_gpu = os.environ.get("TCRVALID_USE_GPU", "0") == "1"
        try:
            if use_gpu:
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        self.batch_size = batch_size
        self.output_dim = 16
        self.cache_id = f"tcrvalid:{model_name}:{repo_dir}:raw-v2"

    def encode(self, seq_idx, v_idx, inverse_v_vocab, raw_v_genes=None):
        sequences = arrays_to_sequences(seq_idx)
        if raw_v_genes is None:
            v_genes = [inverse_v_vocab.get(int(index), "") for index in v_idx]
        else:
            v_genes = raw_v_genes
        inputs = tcrvalid_inputs(sequences, v_genes, self.reference_csv)
        outputs = self.model.predict(inputs, batch_size=self.batch_size, verbose=0)
        # The released encoder returns (z_mean, z_log_var, sampled_z).
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return np.asarray(outputs, dtype=np.float16)


def build_pretrained_sequence_encoder(
    encoder_name,
    model_name,
    device,
    batch_size,
    tcrvalid_repo=None,
):
    if encoder_name == "esm2":
        return HuggingFaceSequenceEncoder(
            encoder_name, model_name or DEFAULT_ESM2_MODEL, device, batch_size
        )
    if encoder_name == "tcrbert":
        return HuggingFaceSequenceEncoder(
            encoder_name, model_name or DEFAULT_TCRBERT_MODEL, device, batch_size
        )
    if encoder_name == "tcrvalid":
        if tcrvalid_repo is None:
            tcrvalid_repo = Path(__file__).resolve().parents[2] / "tcrvalid"
        return TCRValidSequenceEncoder(
            tcrvalid_repo, model_name or DEFAULT_TCRVALID_MODEL, batch_size
        )
    raise ValueError(f"Unsupported pretrained encoder: {encoder_name}")
