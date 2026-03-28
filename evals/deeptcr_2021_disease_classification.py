"""
Evaluation script for DeepTCR (Sidhom et al. 2021) disease classification.

Reference: Sidhom et al. 2021, "DeepTCR is a deep learning framework for
revealing sequence concepts within T-cell repertoires"
"""

import os
import sys
import argparse
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# All DeepTCR sources live in a flat models/DeepTCR/ directory.
_DEEPTCR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'DeepTCR'
)
sys.path.insert(0, _DEEPTCR_DIR)

from DeepTCR import DeepTCR_WF
from utils_s import Get_Train_Valid_Test_KFold


class DeepTCREvaluator:
    """
    Evaluator for DeepTCR_WF on binary disease classification.

    Reads AIRR .tsv.gz files, extracts CDR3 beta + duplicate_count columns,
    writes temporary flat TSV files organised by class label, then runs
    DeepTCR's whole-repertoire classifier with pre-defined cross-validation
    folds that match the benchmark's fold assignments.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self,
                 sequence_col='cdr3_aa',
                 count_col='duplicate_count',
                 train_val_ratio=0.9,
                 random_state=7,
                 # DeepTCR network / training hyperparameters
                 kernel=5,
                 num_concepts=64,
                 size_of_net='small',
                 epochs_min=10,
                 hinge_loss_t=0.1,
                 train_loss_min=0.1,
                 combine_train_valid=True,
                 batch_size=25,
                 n_jobs=4,
                 device=0,
                 results_dir='results/deeptcr',
                 indices_map=None):
        """
        Args:
            sequence_col: AIRR column with CDR3 amino acid sequences.
            count_col: AIRR column with duplicate counts; uses 1 if absent.
            train_val_ratio: Fraction of non-test data used for training
                             (remainder is validation for early stopping).
            random_state: Seed for train/val split.
            kernel: CNN kernel size.
            num_concepts: Number of MIL attention concepts.
            size_of_net: 'small', 'medium', or 'large'.
            epochs_min: Minimum epochs before early stopping.
            hinge_loss_t: Per-sample loss threshold (hinge regularisation).
            train_loss_min: Stop training once training loss < this value.
            combine_train_valid: Combine train+val into training set and use
                                  train_loss_min as stopping criterion.
            batch_size: Repertoires per mini-batch.
            n_jobs: Parallel processes for DeepTCR data loading.
            device: GPU index (0-based) for TensorFlow; CPU is used
                    automatically if no GPU is available.
            results_dir: Base directory for DeepTCR checkpoint files.
            indices_map: Dict mapping specimen_label to row indices for
                         sequencing-depth experiments.
        """
        self.sequence_col = sequence_col
        self.count_col = count_col
        self.train_val_ratio = train_val_ratio
        self.random_state = random_state
        self.kernel = kernel
        self.num_concepts = num_concepts
        self.size_of_net = size_of_net
        self.epochs_min = epochs_min
        self.hinge_loss_t = hinge_loss_t
        self.train_loss_min = train_loss_min
        self.combine_train_valid = combine_train_valid
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.device = device
        self.results_dir = results_dir
        self.indices_map = indices_map

    # ------------------------------------------------------------------
    # Metadata helpers (same pattern as other evaluators)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease ({target_disease}): {n_disease} samples")
        print(f"  Healthy ({self.HEALTHY_LABEL}): {n_healthy} samples")
        print(f"  Total: {len(filtered)} samples")
        return filtered

    def get_available_diseases(self, metadata_path, disease_col='disease'):
        metadata = self.load_metadata(metadata_path)
        return [d for d in metadata[disease_col].unique() if d != self.HEALTHY_LABEL]

    def construct_file_path(self, participant_label, specimen_label, data_dir,
                            file_prefix='part_table_', file_suffix='.tsv.gz'):
        return os.path.join(data_dir,
                            f"{file_prefix}{participant_label}_{specimen_label}{file_suffix}")

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: self.construct_file_path(
                row[participant_col], row['specimen_label'], data_dir, file_prefix, file_suffix
            ), axis=1
        )
        return metadata

    def filter_existing_files(self, metadata):
        original_count = len(metadata)
        metadata = metadata.copy()
        metadata['file_exists'] = metadata['file_path'].apply(os.path.exists)
        filtered = metadata[metadata['file_exists']].drop(columns=['file_exists'])
        missing = original_count - len(filtered)
        if missing > 0:
            print(f"Note: {missing} of {original_count} files not found; "
                  f"proceeding with {len(filtered)}.")
        return filtered

    # ------------------------------------------------------------------
    # Temp-directory helpers
    # ------------------------------------------------------------------

    def _write_deeptcr_tsv(self, file_path, specimen_label, out_dir):
        """
        Read one AIRR .tsv.gz file and write a two-column TSV that DeepTCR
        can parse with aa_column_beta=0, count_column=1.

        Returns the output path, or None if the file has no usable sequences.
        """
        try:
            df = pd.read_csv(file_path, sep='\t', usecols=lambda c: c in
                             [self.sequence_col, self.count_col])
        except Exception as e:
            print(f"  Warning: could not read {file_path}: {e}")
            return None

        if self.sequence_col not in df.columns:
            print(f"  Warning: '{self.sequence_col}' not in {file_path}, skipping.")
            return None

        # Apply indices_map subsampling if provided
        if self.indices_map is not None and specimen_label in self.indices_map:
            idx = self.indices_map[specimen_label]
            df = df.iloc[idx]

        if self.count_col not in df.columns:
            df[self.count_col] = 1

        out = df[[self.sequence_col, self.count_col]].dropna(subset=[self.sequence_col])
        out = out[out[self.sequence_col].str.len() > 0]
        if len(out) == 0:
            return None

        out_path = os.path.join(out_dir, f"{specimen_label}.tsv")
        out.to_csv(out_path, sep='\t', index=False, header=True)
        return out_path

    def _populate_temp_dir(self, metadata, tmpdir, target_disease):
        """
        Write per-sample TSV files into class subdirectories inside tmpdir.
        Returns the set of specimen_labels successfully written.
        """
        disease_dir = os.path.join(tmpdir, target_disease)
        healthy_dir = os.path.join(tmpdir, 'Healthy')
        os.makedirs(disease_dir, exist_ok=True)
        os.makedirs(healthy_dir, exist_ok=True)

        written = set()
        for _, row in metadata.iterrows():
            dest_dir = disease_dir if row['label'] == 1 else healthy_dir
            path = self._write_deeptcr_tsv(row['file_path'], row['specimen_label'], dest_dir)
            if path is not None:
                written.add(row['specimen_label'])

        return written

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3,
                              random_state=None,
                              tune_parameters=True,
                              p_value_candidates=None,
                              allowed_participants=None):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        For each fold the model is trained from scratch on all non-test
        repertoires and evaluated on the held-out test fold, mirroring the
        approach in deeprc_2020_disease_classification.py.

        Args:
            metadata_path: Path to metadata.tsv.
            target_disease: Disease label to classify against Healthy/Background.
            data_dir: Directory containing AIRR .tsv.gz files.
            participant_col: Column with participant labels.
            file_prefix: Filename prefix (default 'part_table_').
            file_suffix: Filename suffix (default '.tsv.gz').
            disease_col: Column with disease labels.
            fold_col: Column with pre-defined test-fold IDs (0, 1, 2).
            n_folds: Number of cross-validation folds.
            random_state: Seed for train/val split (overrides self.random_state).
            tune_parameters: Accepted for API compatibility; ignored.
            p_value_candidates: Accepted for API compatibility; ignored.
            allowed_participants: Optional set of specimen_labels to restrict to.

        Returns:
            pd.DataFrame with per-sample scores across all folds.
        """
        rng = random_state if random_state is not None else self.random_state

        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)} of {before} specimens "
                  f"based on allowed_participants.")

        all_test_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_val_data = metadata[~test_mask]

            train_data, val_data = train_test_split(
                train_val_data,
                train_size=self.train_val_ratio,
                random_state=rng,
                stratify=train_val_data['label'],
            )
            print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

            # Write ALL samples (train+val+test) into a single temp directory.
            # DeepTCR needs to see all classes at Get_Data time so that its
            # LabelEncoder encodes consistently.
            with tempfile.TemporaryDirectory() as tmpdir:
                written = self._populate_temp_dir(metadata, tmpdir, target_disease)

                # Unique name per fold so checkpoints don't collide
                fold_name = os.path.join(
                    self.results_dir,
                    f"{target_disease}_fold{test_fold}"
                )
                os.makedirs(fold_name, exist_ok=True)

                dtcr = DeepTCR_WF(fold_name, device=self.device)
                dtcr.Get_Data(
                    directory=tmpdir,
                    Load_Prev_Data=False,
                    aggregate_by_aa=True,
                    aa_column_beta=0,   # cdr3_aa is column 0 in our temp TSV
                    count_column=1,      # duplicate_count is column 1
                    n_jobs=self.n_jobs,
                )

                # ----------------------------------------------------------
                # Map specimen_label → index in dtcr.sample_list.
                # DeepTCR uses filenames (e.g. "SPECIMEN.tsv") as sample IDs.
                # ----------------------------------------------------------
                sample_name_to_idx = {
                    s.replace('.tsv', ''): i
                    for i, s in enumerate(dtcr.sample_list)
                }

                # Build per-sample label array (shape [n_samples, n_classes])
                Y = np.vstack([
                    dtcr.Y[np.where(dtcr.sample_id == s)[0][0]]
                    for s in dtcr.sample_list
                ])
                Vars = [np.asarray(dtcr.sample_list)]

                # Determine which column in y_pred corresponds to disease
                disease_class_idx = int(
                    np.where(dtcr.lb.classes_ == target_disease)[0][0]
                )

                # Resolve specimen labels that were successfully loaded
                def _get_indices(specimens):
                    idx = [sample_name_to_idx[s]
                           for s in specimens['specimen_label']
                           if s in sample_name_to_idx]
                    return np.array(idx, dtype=int)

                test_idx = _get_indices(test_data)
                train_val_idx = _get_indices(
                    pd.concat([train_data, val_data], ignore_index=True)
                )

                if len(test_idx) == 0:
                    print(f"Warning: no test samples found for fold {test_fold}, skipping.")
                    continue

                # Split train_val into train / val for early stopping
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    train_size=self.train_val_ratio,
                    random_state=rng,
                    stratify=Y[train_val_idx].argmax(axis=1),
                )

                dtcr.train, dtcr.valid, dtcr.test = Get_Train_Valid_Test_KFold(
                    Vars=Vars,
                    train_idx=train_idx,
                    valid_idx=val_idx,
                    test_idx=test_idx,
                    Y=Y,
                )
                dtcr.LOO = None

                if self.combine_train_valid:
                    for i in range(len(dtcr.train)):
                        dtcr.train[i] = np.concatenate(
                            (dtcr.train[i], dtcr.valid[i]), axis=0
                        )
                        dtcr.valid[i] = dtcr.test[i]

                # Build and train
                dtcr._reset_models()
                dtcr._build(
                    kernel=self.kernel,
                    num_concepts=self.num_concepts,
                    size_of_net=self.size_of_net,
                    epochs_min=self.epochs_min,
                    hinge_loss_t=self.hinge_loss_t,
                    train_loss_min=self.train_loss_min if self.combine_train_valid else None,
                    convergence='training' if self.combine_train_valid else 'validation',
                    batch_size=self.batch_size,
                    suppress_output=False,
                )
                dtcr._train(write=False, batch_seed=None, iteration=test_fold)

                # ----------------------------------------------------------
                # Collect per-sample predictions
                # ----------------------------------------------------------
                # dtcr.y_pred: shape [n_test, n_classes], same order as test[0]
                # dtcr.test[0]: array of sample filenames
                test_sample_names = dtcr.test[0]   # filenames like 'SPECIMEN.tsv'
                test_preds = dtcr.y_pred            # [n_test, n_classes]
                test_labels_arr = dtcr.y_test       # [n_test, n_classes] one-hot

                id_to_row = {
                    row['specimen_label']: row
                    for _, row in test_data.iterrows()
                }

                fold_probs = []
                fold_labels = []

                for fname, pred_vec, label_vec in zip(
                        test_sample_names, test_preds, test_labels_arr):
                    specimen = fname.replace('.tsv', '')
                    if specimen not in id_to_row:
                        continue
                    row = id_to_row[specimen]
                    score = float(pred_vec[disease_class_idx])
                    true_label = int(row['label'])
                    fold_probs.append(score)
                    fold_labels.append(true_label)
                    all_test_rows.append({
                        'participant_label': row[participant_col],
                        'specimen_label': specimen,
                        'disease_label': true_label,
                        'disease_label_str': row[disease_col],
                        'method': 'DeepTCR',
                        'disease_model': target_disease,
                        'model_score': score,
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    })

                if len(fold_labels) < 2 or len(set(fold_labels)) < 2:
                    print(f"Warning: fold {test_fold} has <2 classes in test, skipping metrics.")
                    all_probs.extend(fold_probs)
                    all_labels.extend(fold_labels)
                    continue

                fold_auroc = roc_auc_score(fold_labels, fold_probs)
                fold_aupr = average_precision_score(fold_labels, fold_probs)
                print(f"Test AUROC: {fold_auroc:.4f}, Test AUPR: {fold_aupr:.4f}")

                fold_results.append({
                    'fold': test_fold,
                    'test_auroc': fold_auroc,
                    'test_aupr': fold_aupr,
                })
                all_probs.extend(fold_probs)
                all_labels.extend(fold_labels)

        if len(all_labels) >= 2 and len(set(all_labels)) >= 2:
            all_probs_arr = np.array(all_probs)
            all_labels_arr = np.array(all_labels)
            overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
            overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)

            print(f"\n{'='*60}")
            print(f"OVERALL RESULTS: {target_disease} vs Healthy")
            print(f"{'='*60}")
            if fold_results:
                fold_aurocs = [r['test_auroc'] for r in fold_results]
                fold_auprs = [r['test_aupr'] for r in fold_results]
                print(f"Mean Test AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
                print(f"Mean Test AUPR:  {np.mean(fold_auprs):.4f}  ± {np.std(fold_auprs):.4f}")
            print(f"Overall AUROC (all folds combined): {overall_auroc:.4f}")
            print(f"Overall AUPR  (all folds combined): {overall_aupr:.4f}")

        return pd.DataFrame(all_test_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepTCR Disease Classification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing AIRR .tsv.gz repertoire files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--results_dir', type=str, default='results/deeptcr',
                        help='Directory for DeepTCR checkpoints (default: results/deeptcr)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (default: 0; TF falls back to CPU if unavailable)')
    parser.add_argument('--kernel', type=int, default=5,
                        help='CNN kernel size (default: 5)')
    parser.add_argument('--num_concepts', type=int, default=64,
                        help='Number of MIL attention concepts (default: 64)')
    parser.add_argument('--size_of_net', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Network size (default: small)')
    parser.add_argument('--epochs_min', type=int, default=10,
                        help='Minimum training epochs (default: 10)')
    parser.add_argument('--hinge_loss_t', type=float, default=0.1,
                        help='Per-sample hinge loss threshold (default: 0.1)')
    parser.add_argument('--train_loss_min', type=float, default=0.1,
                        help='Stop training when training loss < this (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Repertoires per mini-batch (default: 25)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Parallel data-loading processes (default: 4)')
    args = parser.parse_args()

    evaluator = DeepTCREvaluator(
        kernel=args.kernel,
        num_concepts=args.num_concepts,
        size_of_net=args.size_of_net,
        epochs_min=args.epochs_min,
        hinge_loss_t=args.hinge_loss_t,
        train_loss_min=args.train_loss_min,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        device=args.device,
        results_dir=args.results_dir,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
