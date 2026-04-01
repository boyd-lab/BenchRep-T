"""
External dataset evaluation: train on internal data, evaluate on an external dataset.

Trains a model on ALL internal data (disease vs. Healthy/Background) with an
internal train/val split for hyperparameter tuning, then evaluates on an
external dataset that may have a different metadata schema and repertoire column
names.

Supported models:
  - ml_baseline: Gapped 4-mer + V/J gene logistic regression ensemble
  - emerson_2017: Fisher's exact test + Beta-Binomial generative model
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models.ml_baseline import Gapped_4mer_VJgene
from models.emerson_2017 import CMV_Immunosequencing_Model
from utils.gene_harmonization import adaptive_to_airr


SUPPORTED_MODELS = ['ml_baseline', 'emerson_2017']

# Model name → display name for output
MODEL_DISPLAY_NAMES = {
    'ml_baseline': 'ML_Baseline',
    'emerson_2017': 'Emerson_2017',
}

# Repertoire format presets: column names and file template for known data sources
FORMAT_PRESETS = {
    'adaptive': {
        'sequence_col': 'aminoAcid',
        'v_gene_col': 'vGeneName',
        'j_gene_col': 'jGeneName',
        'file_template': '{sample_name}_TCRB.tsv',
    },
    'airr': {
        'sequence_col': 'cdr3_aa',
        'v_gene_col': 'v_call',
        'j_gene_col': 'j_call',
        'file_template': 'part_table_{sample_name}.tsv.gz',
    },
}


class ExternalEvaluator:
    """
    Train on an internal dataset and evaluate on an external dataset.

    Supports different metadata column names and repertoire file formats
    between internal (training) and external (test) datasets.
    """

    INTERNAL_HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, model_name='ml_baseline',
                 # ML Baseline hyperparameters
                 val_split=0.2, n_cv_folds=5,
                 # Emerson 2017 hyperparameters
                 train_val_ratio=0.9,
                 p_value_candidates=None,
                 # Internal repertoire column names
                 train_sequence_col='cdr3_aa',
                 train_v_gene_col='v_call',
                 train_j_gene_col='j_call',
                 # External repertoire column names
                 ext_sequence_col='aminoAcid',
                 ext_v_gene_col='vGeneName',
                 ext_j_gene_col='jGeneName'):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Supported: {SUPPORTED_MODELS}")
        self.model_name = model_name
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.train_val_ratio = train_val_ratio
        self.p_value_candidates = p_value_candidates or [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        self.train_sequence_col = train_sequence_col
        self.train_v_gene_col = train_v_gene_col
        self.train_j_gene_col = train_j_gene_col
        self.ext_sequence_col = ext_sequence_col
        self.ext_v_gene_col = ext_v_gene_col
        self.ext_j_gene_col = ext_j_gene_col
        self.model = None

    # ------------------------------------------------------------------
    # Internal (training) dataset helpers
    # ------------------------------------------------------------------

    def _prepare_train_data(self, metadata_path, data_dir, target_disease,
                            participant_col='participant_label',
                            disease_col='disease',
                            file_prefix='part_table_',
                            file_suffix='.tsv.gz'):
        """Load internal metadata, filter to disease vs healthy, resolve file paths."""
        metadata = pd.read_csv(metadata_path, sep='\t')
        mask = metadata[disease_col].isin([target_disease, self.INTERNAL_HEALTHY_LABEL])
        data = metadata[mask].copy()
        data['label'] = (data[disease_col] == target_disease).astype(int)

        n_disease = (data['label'] == 1).sum()
        n_healthy = (data['label'] == 0).sum()
        print(f"Training data for '{target_disease}' classification:")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(data)}")

        data['file_path'] = data.apply(
            lambda row: os.path.join(
                data_dir,
                f"{file_prefix}{row[participant_col]}_{row['specimen_label']}{file_suffix}"
            ), axis=1
        )

        before = len(data)
        data = data[data['file_path'].apply(os.path.exists)].copy()
        missing = before - len(data)
        if missing > 0:
            print(f"  {missing} of {before} repertoire files not found; "
                  f"proceeding with {len(data)}.")

        return data

    # ------------------------------------------------------------------
    # External (test) dataset helpers
    # ------------------------------------------------------------------

    def _prepare_external_data(self, ext_metadata_path, ext_data_dir,
                               sample_col='sample_name',
                               disease_col='disease_label',
                               healthy_label='Healthy',
                               disease_label='T1D',
                               file_template='{sample_name}_TCRB.tsv'):
        """Load external metadata, construct file paths, assign binary labels."""
        metadata = pd.read_csv(ext_metadata_path, sep='\t')
        mask = metadata[disease_col].isin([disease_label, healthy_label])
        data = metadata[mask].copy()
        data['label'] = (data[disease_col] == disease_label).astype(int)

        n_disease = (data['label'] == 1).sum()
        n_healthy = (data['label'] == 0).sum()
        print(f"\nExternal data for '{disease_label}' vs '{healthy_label}':")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(data)}")

        data['file_path'] = data[sample_col].apply(
            lambda name: os.path.join(
                ext_data_dir,
                file_template.format(sample_name=name)
            )
        )

        before = len(data)
        data = data[data['file_path'].apply(os.path.exists)].copy()
        missing = before - len(data)
        if missing > 0:
            print(f"  {missing} of {before} repertoire files not found; "
                  f"proceeding with {len(data)}.")

        return data

    # ------------------------------------------------------------------
    # Model-specific training
    # ------------------------------------------------------------------

    def _train_ml_baseline(self, train_files, train_labels):
        """Train the ML Baseline model (handles internal 80/20 split)."""
        self.model = Gapped_4mer_VJgene(
            val_split=self.val_split,
            n_cv_folds=self.n_cv_folds,
            sequence_col=self.train_sequence_col,
            v_gene_col=self.train_v_gene_col,
            j_gene_col=self.train_j_gene_col,
            ignore_allele=True,
        )
        train_result = self.model.train(train_files, train_labels)
        return train_result

    def _train_emerson_2017(self, train_files, train_labels, random_state=7):
        """
        Train the Emerson 2017 model with p-value threshold tuning.

        Splits training data into train/val, tunes p_value_threshold on val,
        then retrains with the best threshold on all training data.
        """
        # Split into train and validation for p-value tuning
        indices = np.arange(len(train_files))
        train_idx, val_idx = train_test_split(
            indices,
            train_size=self.train_val_ratio,
            random_state=random_state,
            stratify=train_labels,
        )

        base_files = [train_files[i] for i in train_idx]
        base_labels = [train_labels[i] for i in train_idx]
        val_files = [train_files[i] for i in val_idx]
        val_labels = [train_labels[i] for i in val_idx]

        print(f"  Train/Val split: {len(base_files)} train, {len(val_files)} val")

        # Create base model for preloading and caching.
        # ignore_allele=True so diagnostic TCRs are stored allele-free,
        # enabling gene-level matching with external data.
        base_model = CMV_Immunosequencing_Model(
            sequence_col=self.train_sequence_col,
            v_col=self.train_v_gene_col,
            j_col=self.train_j_gene_col,
            ignore_allele=True,
        )

        # Preload all repertoires once
        base_model.preload_repertoires(train_files)

        # Compute TCR statistics on the train split (one-time)
        base_model.compute_tcr_statistics(base_files, base_labels)

        # Tune p-value threshold on validation set
        print(f"\n--- Tuning p-value threshold ---")
        print(f"Candidates: {self.p_value_candidates}")
        tuning_results = []

        for p_val in self.p_value_candidates:
            model = CMV_Immunosequencing_Model(
                p_value_threshold=p_val,
                sequence_col=self.train_sequence_col,
                v_col=self.train_v_gene_col,
                j_col=self.train_j_gene_col,
                ignore_allele=True,
            )
            model._repertoire_cache = base_model._repertoire_cache
            model._tcr_stats_cache = base_model._tcr_stats_cache

            model.select_diagnostic_tcrs_from_cache(p_val)

            if len(model.diagnostic_tcrs) == 0:
                print(f"  p={p_val:.0e}: No diagnostic TCRs found, skipping...")
                tuning_results.append({
                    'p_value': p_val, 'n_tcrs': 0,
                    'val_auroc': 0.0, 'val_aupr': 0.0,
                })
                continue

            model.train_beta_binomial_model(base_files, base_labels)

            val_probs = []
            for fp in val_files:
                result = model.predict_diagnosis(fp)
                val_probs.append(result['probability_positive'])

            val_probs_arr = np.array(val_probs)
            val_labels_arr = np.array(val_labels)
            val_auroc = roc_auc_score(val_labels_arr, val_probs_arr)
            val_aupr = average_precision_score(val_labels_arr, val_probs_arr)

            tuning_results.append({
                'p_value': p_val,
                'n_tcrs': len(model.diagnostic_tcrs),
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
            })
            print(f"  p={p_val:.0e}: {len(model.diagnostic_tcrs)} TCRs, "
                  f"Val AUROC={val_auroc:.4f}, Val AUPR={val_aupr:.4f}")

        best = max(tuning_results, key=lambda x: x['val_auroc'])
        best_p_value = best['p_value']
        print(f"\nBest p-value: {best_p_value:.0e} "
              f"(Val AUROC={best['val_auroc']:.4f})")

        if best['n_tcrs'] == 0:
            print("WARNING: No diagnostic TCRs found at any threshold.")
            self.model = None
            return {
                'best_p_value': best_p_value,
                'val_auroc': 0.0,
                'val_aupr': 0.0,
                'n_diagnostic_tcrs': 0,
                'no_diagnostic_tcrs': True,
                'tuning_results': tuning_results,
            }

        # Retrain on ALL training data with the best threshold
        print(f"\nRetraining on all {len(train_files)} samples with p={best_p_value:.0e}...")
        self.model = CMV_Immunosequencing_Model(
            p_value_threshold=best_p_value,
            sequence_col=self.train_sequence_col,
            v_col=self.train_v_gene_col,
            j_col=self.train_j_gene_col,
            ignore_allele=True,
        )
        self.model._repertoire_cache = base_model._repertoire_cache
        self.model.identify_diagnostic_tcrs(train_files, train_labels)
        self.model.train_beta_binomial_model(train_files, train_labels)

        return {
            'best_p_value': best_p_value,
            'val_auroc': best['val_auroc'],
            'val_aupr': best['val_aupr'],
            'n_diagnostic_tcrs': len(self.model.diagnostic_tcrs),
            'no_diagnostic_tcrs': False,
            'tuning_results': tuning_results,
        }

    def _switch_to_external_cols(self):
        """Switch the model's column name attributes to external format."""
        if self.model_name == 'ml_baseline':
            self.model.sequence_col = self.ext_sequence_col
            self.model.v_gene_col = self.ext_v_gene_col
            self.model.j_gene_col = self.ext_j_gene_col
            # Harmonize Adaptive gene names to AIRR format for feature matching
            self.model.v_gene_harmonizer = adaptive_to_airr
            self.model.j_gene_harmonizer = adaptive_to_airr
        elif self.model_name == 'emerson_2017':
            self.model.sequence_col = self.ext_sequence_col
            self.model.v_col = self.ext_v_gene_col
            self.model.j_col = self.ext_j_gene_col
            # Harmonize Adaptive gene names to AIRR format for matching
            self.model.v_gene_harmonizer = adaptive_to_airr
            self.model.j_gene_harmonizer = adaptive_to_airr
        self.model.clear_cache()

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def run_external_evaluation(self,
                                train_metadata_path,
                                train_data_dir,
                                target_disease,
                                ext_metadata_path,
                                ext_data_dir,
                                ext_sample_col='sample_name',
                                ext_disease_col='disease_label',
                                ext_healthy_label='Healthy',
                                ext_disease_label='T1D',
                                ext_file_template='{sample_name}_TCRB.tsv',
                                random_state=7,
                                output_csv=None):
        """
        Train on all internal data and evaluate on external dataset.

        Args:
            train_metadata_path: Path to internal metadata.tsv.
            train_data_dir: Directory with internal repertoire files.
            target_disease: Disease name in internal metadata (e.g. 'T1D').
            ext_metadata_path: Path to external metadata.tsv.
            ext_data_dir: Directory with external repertoire files.
            ext_sample_col: Column with sample identifiers in external metadata.
            ext_disease_col: Column with disease labels in external metadata.
            ext_healthy_label: Healthy label string in external metadata.
            ext_disease_label: Disease label string in external metadata.
            ext_file_template: Template for external repertoire filenames.
            random_state: Random seed for reproducibility.
            output_csv: Optional path to save per-sample scores.

        Returns:
            Tuple of (pd.DataFrame with per-sample predictions, metrics dict).
        """
        method_name = MODEL_DISPLAY_NAMES[self.model_name]

        # --- Prepare training data ---
        train_data = self._prepare_train_data(
            train_metadata_path, train_data_dir, target_disease
        )
        train_files = train_data['file_path'].tolist()
        train_labels = train_data['label'].tolist()

        # --- Train model on all internal data ---
        print(f"\n{'='*60}")
        print(f"TRAINING {method_name} on internal dataset "
              f"({len(train_files)} samples)")
        print(f"{'='*60}")

        if self.model_name == 'ml_baseline':
            train_result = self._train_ml_baseline(train_files, train_labels)
        elif self.model_name == 'emerson_2017':
            train_result = self._train_emerson_2017(
                train_files, train_labels, random_state=random_state
            )
            if train_result.get('no_diagnostic_tcrs', False):
                print("Cannot evaluate: no diagnostic TCRs found.")
                return pd.DataFrame(), train_result

        # --- Prepare external data ---
        ext_data = self._prepare_external_data(
            ext_metadata_path, ext_data_dir,
            sample_col=ext_sample_col,
            disease_col=ext_disease_col,
            healthy_label=ext_healthy_label,
            disease_label=ext_disease_label,
            file_template=ext_file_template,
        )
        ext_files = ext_data['file_path'].tolist()
        ext_labels = ext_data['label'].values

        # --- Switch column names to external format for prediction ---
        self._switch_to_external_cols()

        # --- Predict on external data ---
        print(f"\n{'='*60}")
        print(f"EVALUATING on external dataset ({len(ext_files)} samples)")
        print(f"{'='*60}")

        ext_probs = []
        for fp in tqdm(ext_files, desc="Predicting on external data"):
            result = self.model.predict_diagnosis(fp)
            ext_probs.append(result['probability_positive'])

        ext_probs = np.array(ext_probs)

        # --- Compute metrics ---
        auroc = roc_auc_score(ext_labels, ext_probs)
        aupr = average_precision_score(ext_labels, ext_probs)

        print(f"\n{'='*60}")
        print(f"EXTERNAL EVALUATION RESULTS: {method_name} on {target_disease}")
        print(f"{'='*60}")
        print(f"Training: {len(train_files)} samples (internal)")

        if self.model_name == 'ml_baseline':
            print(f"  Best C (k-mer): {train_result['best_c_kmer']}")
            print(f"  Best C (V/J):   {train_result['best_c_vj']}")
            print(f"  Best alpha:     {train_result['best_alpha']:.1f}")
            print(f"  Val AUROC:      {train_result['val_auroc']:.4f}")
        elif self.model_name == 'emerson_2017':
            print(f"  Best p-value:       {train_result['best_p_value']:.0e}")
            print(f"  Diagnostic TCRs:    {train_result['n_diagnostic_tcrs']}")
            print(f"  Val AUROC:          {train_result['val_auroc']:.4f}")

        print(f"Evaluation: {len(ext_files)} samples (external)")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AUPR:  {aupr:.4f}")

        # --- Build per-sample output ---
        rows = []
        for (_, row), score in zip(ext_data.iterrows(), ext_probs):
            rows.append({
                'sample_name': row[ext_sample_col],
                'disease_label': int(row['label']),
                'disease_label_str': row[ext_disease_col],
                'method': method_name,
                'disease_model': target_disease,
                'model_score': float(score),
            })
        scores_df = pd.DataFrame(rows)

        if output_csv:
            scores_df.to_csv(output_csv, index=False)
            print(f"\nScores saved to: {output_csv}")

        return scores_df, {
            'auroc': auroc,
            'aupr': aupr,
            'train_result': train_result,
            'n_train': len(train_files),
            'n_external': len(ext_files),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External dataset evaluation: train on internal, evaluate on external"
    )

    # Model selection
    parser.add_argument('--model', type=str, default='ml_baseline',
                        choices=SUPPORTED_MODELS,
                        help='Model to use (default: ml_baseline)')

    # Internal (training) dataset
    parser.add_argument('--train_metadata_path', type=str, required=True,
                        help='Path to internal metadata.tsv')
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help='Directory containing internal repertoire files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (must exist in internal metadata)')

    # External (test) dataset
    parser.add_argument('--ext_metadata_path', type=str, required=True,
                        help='Path to external metadata file')
    parser.add_argument('--ext_data_dir', type=str, required=True,
                        help='Directory containing external repertoire files')
    parser.add_argument('--ext_format', type=str, default='adaptive',
                        choices=list(FORMAT_PRESETS.keys()),
                        help='External data format preset (default: adaptive). '
                             'Sets column names and file template automatically.')
    parser.add_argument('--ext_sample_col', type=str, default='sample_name',
                        help='Column with sample identifiers in external metadata')
    parser.add_argument('--ext_disease_col', type=str, default='disease_label',
                        help='Column with disease labels in external metadata')
    parser.add_argument('--ext_healthy_label', type=str, default='Healthy',
                        help='Healthy label string in external metadata')
    parser.add_argument('--ext_disease_label', type=str, default='T1D',
                        help='Disease label string in external metadata')

    # Overrides for format preset (rarely needed)
    parser.add_argument('--ext_sequence_col', type=str, default=None,
                        help='Override CDR3 sequence column in external files')
    parser.add_argument('--ext_v_gene_col', type=str, default=None,
                        help='Override V gene column in external files')
    parser.add_argument('--ext_j_gene_col', type=str, default=None,
                        help='Override J gene column in external files')
    parser.add_argument('--ext_file_template', type=str, default=None,
                        help='Override file template for external repertoires')

    # Model hyperparameters
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Internal val fraction for ML Baseline alpha tuning (default: 0.2)')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='CV folds for ML Baseline C tuning (default: 5)')
    parser.add_argument('--train_val_ratio', type=float, default=0.9,
                        help='Train/val ratio for Emerson 2017 tuning (default: 0.9)')
    parser.add_argument('--random_state', type=int, default=7,
                        help='Random seed (default: 7)')

    # Output
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')

    args = parser.parse_args()

    # Resolve format preset, then apply any explicit overrides
    preset = FORMAT_PRESETS[args.ext_format]
    ext_sequence_col = args.ext_sequence_col or preset['sequence_col']
    ext_v_gene_col = args.ext_v_gene_col or preset['v_gene_col']
    ext_j_gene_col = args.ext_j_gene_col or preset['j_gene_col']
    ext_file_template = args.ext_file_template or preset['file_template']

    evaluator = ExternalEvaluator(
        model_name=args.model,
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
        train_val_ratio=args.train_val_ratio,
        ext_sequence_col=ext_sequence_col,
        ext_v_gene_col=ext_v_gene_col,
        ext_j_gene_col=ext_j_gene_col,
    )

    scores_df, metrics = evaluator.run_external_evaluation(
        train_metadata_path=args.train_metadata_path,
        train_data_dir=args.train_data_dir,
        target_disease=args.target_disease,
        ext_metadata_path=args.ext_metadata_path,
        ext_data_dir=args.ext_data_dir,
        ext_sample_col=args.ext_sample_col,
        ext_disease_col=args.ext_disease_col,
        ext_healthy_label=args.ext_healthy_label,
        ext_disease_label=args.ext_disease_label,
        ext_file_template=ext_file_template,
        random_state=args.random_state,
        output_csv=args.output_csv,
    )
