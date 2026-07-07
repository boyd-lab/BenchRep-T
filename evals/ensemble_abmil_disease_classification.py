"""
Evaluation script for the ABMIL disease classification model.

Each repertoire is treated as a bag of sequences. Per-sequence features are
produced end-to-end by TCRSeqEncoder (learned AA embedding + 1-D conv +
V/J gene embeddings), then aggregated by Gated Attention MIL.

Reference architecture: Ilse et al. 2018, "Attention-based Deep Multiple Instance
Learning" (https://arxiv.org/abs/1802.04712).
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from models.ensemble_abmil import ABMIL
from utils.covariate_residualization import covariate_adjusted_predict, filter_complete_demographics
from utils.cohort_adjustments import apply_cohort_adjustment


class ABMILEvaluator:
    """
    Evaluator for the ABMIL disease classification model.

    The evaluator passes all non-test data directly to ABMIL.train() for each
    cross-validation fold. A fresh model is instantiated per fold.
    """

    HEALTHY_LABEL = "Healthy/Background"

    @staticmethod
    def _feature_csv(output_csv, feature):
        """Derive a per-feature CSV path, e.g. results/foo.csv -> results/foo_vj_only.csv."""
        if output_csv is None:
            return None
        base, ext = os.path.splitext(output_csv)
        return f"{base}_{feature}{ext}"

    def __init__(
        self,
        sequence_col='cdr3_aa',
        v_gene_col='v_call',
        j_gene_col='j_call',
        max_instances=10000,
        M=128,
        L=64,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        patience=10,
        val_split=0.2,
        seed=7,
        subsample_fraction=1.0,
        subsample_seed=7,
        use_gpu=True,
        indices_map=None,
        max_repertoires_per_class=None,
        dropout=0.25,
        max_length=40,
        embedding_dim_aa=64,
        embedding_dim_genes=48,
        kernel=5,
        conv_units=(32, 64, 128),
        features='full',
        healthy_label=None,
    ):
        """
        Args:
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls.
            max_instances: Sequences randomly subsampled per bag per training epoch
                (augmentation / memory control). Evaluation always uses all sequences.
            M: ABMIL hidden dimension.
            L: ABMIL attention hidden dimension.
            epochs: Maximum training epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            patience: Early-stopping patience in epochs.
            val_split: Fraction of training bags held out for early stopping.
            seed: Random seed for val split and subsampling.
            subsample_fraction: Fraction of reads to sample per repertoire.
            subsample_seed: Random seed for repertoire subsampling.
            use_gpu: Use CUDA if available.
            indices_map: Dict mapping specimen_label to pre-computed row indices.
            max_length: Maximum CDR3 length; longer sequences are truncated.
            embedding_dim_aa: Learned AA embedding dimension.
            embedding_dim_genes: Learned V/J gene embedding dimension.
            kernel: First conv kernel size.
            conv_units: Output channels for the three conv layers.
            features: Which features to use — 'full' (CDR3 + V/J genes),
                'cdr3_only' (CDR3 sequence only), or 'vj_only' (V/J gene identities only).
        """
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.max_instances = max_instances
        self.M = M
        self.L = L
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.seed = seed
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.use_gpu = use_gpu
        self.indices_map = indices_map
        self.max_repertoires_per_class = max_repertoires_per_class
        self.dropout = dropout
        self.max_length = max_length
        self.embedding_dim_aa = embedding_dim_aa
        self.embedding_dim_genes = embedding_dim_genes
        self.kernel = kernel
        self.conv_units = tuple(conv_units)
        self.features = features
        if healthy_label is not None:
            self.HEALTHY_LABEL = healthy_label
        self.canonicalize_genes = False
        self.model = None

    # ------------------------------------------------------------------
    # Metadata helpers (shared pattern across evaluators)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease',
                             require_demographics=False,
                             adjust_distribution_by_demographics=False,
                             random_baseline=False, random_baseline_seed=7):
        """
        Filter metadata to target disease vs. Healthy/Background and add binary labels.

        Args:
            metadata: DataFrame with metadata.
            target_disease: Disease name to classify.
            disease_col: Column with disease labels.
            require_demographics: If True, drop rows with missing age, sex,
                or ancestry so the subset matches the demographic baseline.
            adjust_distribution_by_demographics: If True, apply per-disease cohort
                distribution adjustment (see utils.cohort_adjustments).
            random_baseline: If True (with adjust_distribution_by_demographics),
                resample healthy uniformly at random to the same target N.
            random_baseline_seed: RNG seed for the random-baseline draw.

        Returns:
            DataFrame with a 'label' column (1 = disease, 0 = healthy).
        """
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        if adjust_distribution_by_demographics:
            filtered = apply_cohort_adjustment(
                filtered, target_disease,
                seed=random_baseline_seed,
                random_baseline=random_baseline,
            )

        if require_demographics:
            before = len(filtered)
            filtered = filtered.dropna(subset=['age', 'sex', 'ancestry'])
            filtered = filtered[filtered['ancestry'].str.strip() != '']
            after = len(filtered)
            if before != after:
                print(f"  Dropped {before - after} rows with missing demographics "
                      f"({before} -> {after})")

        if self.max_repertoires_per_class is not None:
            pos = filtered[filtered['label'] == 1].head(self.max_repertoires_per_class)
            neg = filtered[filtered['label'] == 0].head(self.max_repertoires_per_class)
            filtered = pd.concat([pos, neg])
            print(f"  (debug) Capped to {self.max_repertoires_per_class} repertoires per class.")

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(filtered)}")
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
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7,
                              allowed_participants=None,
                              require_demographics=False,
                              adjust_distribution_by_demographics=False,
                              random_baseline=False, random_baseline_seed=7,
                              covariate_adjust=False,
                              model_save_dir=None,
                              ext_metadata_path=None, ext_data_dir=None,
                              ext_file_template='{participant_label}_TCRB.tsv'):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        For each fold, all non-test samples are passed to model.train(), which
        fits the DictVectorizer, scaler, and ABMIL network from scratch.

        Args:
            metadata_path: Path to metadata.tsv.
            target_disease: Disease name to classify against Healthy/Background.
            data_dir: Directory containing repertoire .tsv.gz files.
            participant_col: Column with participant labels.
            file_prefix: Filename prefix (default: 'part_table_').
            file_suffix: Filename suffix (default: '.tsv.gz').
            disease_col: Column with disease labels.
            fold_col: Column with pre-defined fold IDs (0, 1, 2).
            n_folds: Number of folds (default: 3).
            random_state: Random seed for reproducibility (default: 7).
            allowed_participants: Optional set of specimen labels to restrict to.
            require_demographics: If True, drop repertoires with missing
                                  demographic data (age, sex, ancestry).
            covariate_adjust: If True, extract the trained model's attention-
                weighted bag embedding (Z, shape M) for each repertoire, residualize
                against demographic covariates (age, sex, ancestry) fitted on the
                non-test fold only, then train an L1 logistic regression on the
                residualized embeddings. Samples with missing demographics are
                excluded. The ABMIL network is still trained on all non-test data.
                Default: False.

        Returns:
            DataFrame with per-sample scores across all folds.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col,
                                             require_demographics=require_demographics,
                                             adjust_distribution_by_demographics=adjust_distribution_by_demographics,
                                             random_baseline=random_baseline,
                                             random_baseline_seed=random_baseline_seed)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if ext_metadata_path is not None:
            from utils.cohort_merge import prepare_merged_cohort
            metadata = prepare_merged_cohort(
                metadata, ext_metadata_path, ext_data_dir, target_disease,
                ext_file_template=ext_file_template,
                healthy_label=self.HEALTHY_LABEL,
                fold_col=fold_col, disease_col=disease_col,
            )
            self.canonicalize_genes = True

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)} of {before} specimens "
                  f"based on allowed_participants set.")

        all_test_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        base_method = 'ABMIL_CovAdj' if covariate_adjust else 'ABMIL'
        if self.features == 'cdr3_only':
            base_method += '_CDR3'
        elif self.features == 'vj_only':
            base_method += '_VJ'
        if random_baseline:
            base_method += '_RandomBaseline'

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_data = metadata[~test_mask]

            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()
            test_files = test_data['file_path'].tolist()
            test_labels = test_data['label'].tolist()

            # Fresh model per fold
            self.model = ABMIL(
                max_instances=self.max_instances,
                M=self.M,
                L=self.L,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                patience=self.patience,
                val_split=self.val_split,
                seed=self.seed,
                sequence_col=self.sequence_col,
                v_gene_col=self.v_gene_col,
                j_gene_col=self.j_gene_col,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed,
                indices_map=self.indices_map,
                canonicalize_genes=self.canonicalize_genes,
                use_gpu=self.use_gpu,
                dropout=self.dropout,
                max_length=self.max_length,
                embedding_dim_aa=self.embedding_dim_aa,
                embedding_dim_genes=self.embedding_dim_genes,
                kernel=self.kernel,
                conv_units=self.conv_units,
                features=self.features,
            )

            train_result = self.model.train(train_files, train_labels)

            if model_save_dir is not None:
                self.model.save(os.path.join(model_save_dir, target_disease,
                                             f'fold{test_fold}'))

            if covariate_adjust:
                # ----------------------------------------------------------
                # Covariate-adjusted prediction via bag embeddings
                # ----------------------------------------------------------
                train_cov = filter_complete_demographics(train_data)
                test_cov  = filter_complete_demographics(test_data)
                print(f"  Covariate adjust: {len(train_cov)} train, "
                      f"{len(test_cov)} test samples with complete demographics.")

                print("  Extracting train bag embeddings...")
                X_train = np.stack([
                    self.model.get_bag_embedding(fp)
                    for fp in tqdm(train_cov['file_path'].tolist(), leave=False)
                ])
                print("  Extracting test bag embeddings...")
                X_test = np.stack([
                    self.model.get_bag_embedding(fp)
                    for fp in tqdm(test_cov['file_path'].tolist(), leave=False)
                ])

                test_probs      = covariate_adjusted_predict(
                    X_train, train_cov, train_cov['label'].values,
                    X_test,  test_cov,
                )
                test_labels_arr = test_cov['label'].values

                for (_, row), score in zip(test_cov.iterrows(), test_probs):
                    entry = {
                        'participant_label': row[participant_col],
                        'specimen_label': row['specimen_label'],
                        'disease_label': int(row['label']),
                        'disease_label_str': row[disease_col],
                        'method': base_method,
                        'disease_model': target_disease,
                        'model_score': float(score),
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    }
                    if random_baseline:
                        entry['random_baseline_seed'] = int(random_baseline_seed)
                    all_test_rows.append(entry)
            else:
                # ----------------------------------------------------------
                # Standard ABMIL prediction
                # ----------------------------------------------------------
                raw_probs = []
                for fp in tqdm(test_files, desc="Testing"):
                    result = self.model.predict_diagnosis(fp)
                    raw_probs.append(result['probability_positive'])

                test_probs      = np.array(raw_probs)
                test_labels_arr = np.array(test_labels)

                for (_, row), score in zip(test_data.iterrows(), test_probs):
                    entry = {
                        'participant_label': row[participant_col],
                        'specimen_label': row['specimen_label'],
                        'disease_label': int(row['label']),
                        'disease_label_str': row[disease_col],
                        'method': base_method,
                        'disease_model': target_disease,
                        'model_score': float(score),
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    }
                    if random_baseline:
                        entry['random_baseline_seed'] = int(random_baseline_seed)
                    all_test_rows.append(entry)

            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            test_preds = (test_probs >= 0.5).astype(int)
            test_balanced_acc = balanced_accuracy_score(test_labels_arr, test_preds)
            test_f1 = f1_score(test_labels_arr, test_preds)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                  f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

            fold_results.append({
                'fold': test_fold,
                'best_val_loss': train_result['best_val_loss'],
                'epochs_trained': train_result['epochs_trained'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels_arr.tolist())

        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)
        overall_preds = (all_probs_arr >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(all_labels_arr, overall_preds)
        overall_f1 = f1_score(all_labels_arr, overall_preds)

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
        fold_aurocs = [r['test_auroc'] for r in fold_results]
        fold_auprs = [r['test_aupr'] for r in fold_results]
        fold_balanced_accs = [r['test_balanced_acc'] for r in fold_results]
        fold_f1s = [r['test_f1'] for r in fold_results]
        print(f"Mean Test AUROC:        {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"Mean Test AUPR:         {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")
        print(f"Mean Test Balanced Acc: {np.mean(fold_balanced_accs):.4f} ± {np.std(fold_balanced_accs):.4f}")
        print(f"Mean Test F1:           {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"Overall AUROC (all folds combined):        {overall_auroc:.4f}")
        print(f"Overall AUPR  (all folds combined):        {overall_aupr:.4f}")
        print(f"Overall Balanced Acc (all folds combined): {overall_balanced_acc:.4f}")
        print(f"Overall F1 (all folds combined):           {overall_f1:.4f}")

        return pd.DataFrame(all_test_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABMIL Disease Classification")
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--max_instances', type=int, default=10000,
                        help='Sequences randomly subsampled per bag per training epoch '
                             '(default: 10000; evaluation always uses all sequences)')
    parser.add_argument('--M', type=int, default=128,
                        help='ABMIL hidden dim (default: 128)')
    parser.add_argument('--L', type=int, default=64,
                        help='ABMIL attention hidden dim (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout probability (default: 0.25)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Adam weight decay (default: 1e-4)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Adam learning rate (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early-stopping patience in epochs (default: 10)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of train bags held out for early stopping (default: 0.2)')
    parser.add_argument('--max_length', type=int, default=40,
                        help='Maximum CDR3 length; longer sequences are truncated (default: 40)')
    parser.add_argument('--embedding_dim_aa', type=int, default=64,
                        help='Learned AA embedding dimension (default: 64)')
    parser.add_argument('--embedding_dim_genes', type=int, default=48,
                        help='Learned V/J gene embedding dimension (default: 48)')
    parser.add_argument('--kernel', type=int, default=5,
                        help='First conv kernel size (default: 5)')
    parser.add_argument('--features', type=str, default='full',
                        choices=['full', 'cdr3_only', 'vj_only'],
                        help="Which features to use: 'full' (CDR3 + V/J genes), "
                             "'cdr3_only' (CDR3 sequence only), or "
                             "'vj_only' (V/J gene identities only). Default: full")
    parser.add_argument('--include_feature_baselines', action='store_true',
                        help='When --features full, also run cdr3_only and vj_only '
                             'ABMIL models with the same dataset arguments. Sibling '
                             'CSV files are written by appending _cdr3_only and '
                             '_vj_only to --output_csv.')
    parser.add_argument('--healthy_label', type=str,
                        default=ABMILEvaluator.HEALTHY_LABEL,
                        help='Negative-class label in the disease column.')
    parser.add_argument('--participant_col', type=str, default='participant_label',
                        help='Metadata column used as participant_label in outputs and internal file paths.')
    parser.add_argument('--disease_col', type=str, default='disease',
                        help='Metadata column containing disease/control labels.')
    parser.add_argument('--fold_col', type=str,
                        default='malid_cross_validation_fold_id_when_in_test_set',
                        help='Metadata column containing fold IDs.')
    parser.add_argument('--file_prefix', type=str, default='part_table_',
                        help='Internal cohort file prefix.')
    parser.add_argument('--file_suffix', type=str, default='.tsv.gz',
                        help='Internal cohort file suffix.')
    parser.add_argument('--max_repertoires_per_class', type=int, default=None,
                        help='Cap repertoires per class (debug; default: no limit)')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU even if CUDA is available')
    parser.add_argument('--require_demographics', action='store_true',
                        help='Drop repertoires with missing demographic data '
                             '(age, sex, ancestry) to match demographic baseline subset')
    parser.add_argument('--covariate_adjust', action='store_true',
                        help='Residualize bag embeddings against demographics (age, sex, ancestry) '
                             'and train an L1 logistic regression head (requires complete demographics)')
    parser.add_argument('--adjust_distribution_by_demographics', action='store_true',
                        help='Apply per-disease cohort distribution adjustment for fair '
                             'comparison. HIV: filter both cohorts to African ancestry. '
                             'Lupus/T1D/Influenza/Covid19: keep the disease cohort unchanged '
                             'and subsample Healthy/Background so its age distribution (10y '
                             'bins) matches the disease cohort.')
    parser.add_argument('--random_baseline_seeds', type=int, nargs='+', default=None,
                        help='Run the random-sampling healthy baseline for each seed '
                             '(implies --adjust_distribution_by_demographics). For each '
                             'seed, healthy is resampled uniformly at random to the same '
                             'target N as the demographic-matched cohort; disease side '
                             'mirrors the demographic-matched run. Results from all seeds '
                             'are concatenated into one output, with a '
                             '`random_baseline_seed` column. Example: 7 14 21 28 35.')
    parser.add_argument('--model_save_dir', type=str, default=None,
                        help='Directory to save trained per-fold models for driver evaluation.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--ext_metadata_path', type=str, default=None,
                        help='Optional external-cohort metadata TSV (MAL-ID column style). '
                             'Merges external samples by fold; auto-canonicalizes V/J.')
    parser.add_argument('--ext_data_dir', type=str, default=None,
                        help='Directory containing the external cohort repertoire files.')
    parser.add_argument('--ext_file_template', type=str,
                        default='{participant_label}_TCRB.tsv',
                        help='Filename template for external repertoires.')
    parser.add_argument('--max_folds', type=int, default=None,
                        help='Limit cross-validation to this many folds (default: all 3). '
                             'Useful for resource probes, e.g. --max_folds 1.')
    args = parser.parse_args()

    if args.max_folds is not None and args.max_folds < 1:
        parser.error('--max_folds must be >= 1')
    n_outer_folds = args.max_folds if args.max_folds is not None else 3

    if args.include_feature_baselines:
        if args.features != 'full':
            raise ValueError("--include_feature_baselines requires --features full")
        if args.random_baseline_seeds:
            raise ValueError("--include_feature_baselines is not supported with "
                             "--random_baseline_seeds")

    def make_evaluator(feature):
        return ABMILEvaluator(
            max_instances=args.max_instances,
            M=args.M,
            L=args.L,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            val_split=args.val_split,
            use_gpu=not args.no_gpu,
            max_repertoires_per_class=args.max_repertoires_per_class,
            dropout=args.dropout,
            max_length=args.max_length,
            embedding_dim_aa=args.embedding_dim_aa,
            embedding_dim_genes=args.embedding_dim_genes,
            kernel=args.kernel,
            features=feature,
            healthy_label=args.healthy_label,
        )

    def feature_model_save_dir(feature):
        if args.model_save_dir is None:
            return None
        if args.include_feature_baselines and args.features == 'full':
            return os.path.join(args.model_save_dir, feature)
        return args.model_save_dir

    evaluator = make_evaluator(args.features)

    if args.random_baseline_seeds:
        seed_dfs = []
        for seed in args.random_baseline_seeds:
            print(f"\n{'#'*60}")
            print(f"# RANDOM BASELINE RUN — seed={seed}")
            print(f"{'#'*60}")
            seed_df = evaluator.run_cross_validation(
                metadata_path=args.metadata_path,
                target_disease=args.target_disease,
                data_dir=args.repertoire_data_dir,
                participant_col=args.participant_col,
                file_prefix=args.file_prefix,
                file_suffix=args.file_suffix,
                disease_col=args.disease_col,
                fold_col=args.fold_col,
                n_folds=n_outer_folds,
                require_demographics=args.require_demographics,
                adjust_distribution_by_demographics=True,
                random_baseline=True,
                random_baseline_seed=seed,
                covariate_adjust=args.covariate_adjust,
                model_save_dir=args.model_save_dir,
                ext_metadata_path=args.ext_metadata_path,
                ext_data_dir=args.ext_data_dir,
                ext_file_template=args.ext_file_template,
            )
            seed_dfs.append(seed_df)
        scores_df = pd.concat(seed_dfs, axis=0, ignore_index=True)

        per_seed = []
        for seed, seed_df in scores_df.groupby('random_baseline_seed'):
            y = seed_df['disease_label'].values
            p = seed_df['model_score'].values
            per_seed.append({
                'random_baseline_seed': int(seed),
                'overall_auroc': roc_auc_score(y, p),
                'overall_aupr': average_precision_score(y, p),
            })
        summary_df = pd.DataFrame(per_seed)
        print(f"\n{'#'*60}")
        print(f"# RANDOM BASELINE SUMMARY — across {len(summary_df)} seeds")
        print(f"{'#'*60}")
        print(summary_df.to_string(index=False))
        print(f"Mean overall AUROC: {summary_df['overall_auroc'].mean():.4f} "
              f"± {summary_df['overall_auroc'].std(ddof=0):.4f}")
        print(f"Mean overall AUPR:  {summary_df['overall_aupr'].mean():.4f} "
              f"± {summary_df['overall_aupr'].std(ddof=0):.4f}")
    else:
        scores_df = evaluator.run_cross_validation(
            metadata_path=args.metadata_path,
            target_disease=args.target_disease,
            data_dir=args.repertoire_data_dir,
            participant_col=args.participant_col,
            file_prefix=args.file_prefix,
            file_suffix=args.file_suffix,
            disease_col=args.disease_col,
            fold_col=args.fold_col,
            n_folds=n_outer_folds,
            require_demographics=args.require_demographics,
            adjust_distribution_by_demographics=args.adjust_distribution_by_demographics,
            covariate_adjust=args.covariate_adjust,
            model_save_dir=args.model_save_dir,
            ext_metadata_path=args.ext_metadata_path,
            ext_data_dir=args.ext_data_dir,
            ext_file_template=args.ext_file_template,
        )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")

    if args.include_feature_baselines:
        for feature in ['cdr3_only', 'vj_only']:
            feature_output_csv = ABMILEvaluator._feature_csv(args.output_csv, feature)
            print(f"\n{'#'*60}")
            print(f"# ABMIL FEATURE BASELINE - {feature}")
            print(f"{'#'*60}")
            feature_scores_df = make_evaluator(feature).run_cross_validation(
                metadata_path=args.metadata_path,
                target_disease=args.target_disease,
                data_dir=args.repertoire_data_dir,
                participant_col=args.participant_col,
                file_prefix=args.file_prefix,
                file_suffix=args.file_suffix,
                disease_col=args.disease_col,
                fold_col=args.fold_col,
                n_folds=n_outer_folds,
                require_demographics=args.require_demographics,
                adjust_distribution_by_demographics=args.adjust_distribution_by_demographics,
                covariate_adjust=args.covariate_adjust,
                model_save_dir=feature_model_save_dir(feature),
                ext_metadata_path=args.ext_metadata_path,
                ext_data_dir=args.ext_data_dir,
                ext_file_template=args.ext_file_template,
            )
            if feature_output_csv:
                feature_scores_df.to_csv(feature_output_csv, index=False)
                print(f"\nScores saved to: {feature_output_csv}")
