"""
External dataset evaluation: train on internal data, evaluate on an external dataset.

Trains a model on ALL internal data (disease vs. Healthy/Background) with the
model's built-in 80/20 split for hyperparameter tuning, then evaluates on an
external dataset that may have a different metadata schema and repertoire column
names.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models.ml_baseline import Gapped_4mer_VJgene


class ExternalEvaluator:
    """
    Train on an internal dataset and evaluate on an external dataset.

    Supports different metadata column names and repertoire file formats
    between internal (training) and external (test) datasets.
    """

    INTERNAL_HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, val_split=0.2, n_cv_folds=5,
                 train_sequence_col='cdr3_aa',
                 train_v_gene_col='v_call',
                 train_j_gene_col='j_call',
                 ext_sequence_col='aminoAcid',
                 ext_v_gene_col='vGeneName',
                 ext_j_gene_col='jGeneName'):
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
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
            output_csv: Optional path to save per-sample scores.

        Returns:
            pd.DataFrame with per-sample predictions and overall metrics dict.
        """
        # --- Prepare training data ---
        train_data = self._prepare_train_data(
            train_metadata_path, train_data_dir, target_disease
        )
        train_files = train_data['file_path'].tolist()
        train_labels = train_data['label'].values

        # --- Train model on all internal data ---
        print(f"\n{'='*60}")
        print(f"TRAINING on internal dataset ({len(train_files)} samples)")
        print(f"{'='*60}")

        self.model = Gapped_4mer_VJgene(
            val_split=self.val_split,
            n_cv_folds=self.n_cv_folds,
            sequence_col=self.train_sequence_col,
            v_gene_col=self.train_v_gene_col,
            j_gene_col=self.train_j_gene_col,
        )
        train_result = self.model.train(train_files, train_labels)

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
        self.model.sequence_col = self.ext_sequence_col
        self.model.v_gene_col = self.ext_v_gene_col
        self.model.j_gene_col = self.ext_j_gene_col
        self.model.clear_cache()

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
        print(f"EXTERNAL EVALUATION RESULTS: {target_disease}")
        print(f"{'='*60}")
        print(f"Training: {len(train_files)} samples (internal)")
        print(f"  Best C (k-mer): {train_result['best_c_kmer']}")
        print(f"  Best C (V/J):   {train_result['best_c_vj']}")
        print(f"  Best alpha:     {train_result['best_alpha']:.1f}")
        print(f"  Val AUROC:      {train_result['val_auroc']:.4f}")
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
                'method': 'ML_Baseline',
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
    parser.add_argument('--ext_sample_col', type=str, default='sample_name',
                        help='Column with sample identifiers in external metadata')
    parser.add_argument('--ext_disease_col', type=str, default='disease_label',
                        help='Column with disease labels in external metadata')
    parser.add_argument('--ext_healthy_label', type=str, default='Healthy',
                        help='Healthy label string in external metadata')
    parser.add_argument('--ext_disease_label', type=str, default='T1D',
                        help='Disease label string in external metadata')
    parser.add_argument('--ext_file_template', type=str,
                        default='{sample_name}_TCRB.tsv',
                        help='Template for external repertoire filenames')

    # External repertoire column names
    parser.add_argument('--ext_sequence_col', type=str, default='aminoAcid',
                        help='CDR3 sequence column in external repertoire files')
    parser.add_argument('--ext_v_gene_col', type=str, default='vGeneName',
                        help='V gene column in external repertoire files')
    parser.add_argument('--ext_j_gene_col', type=str, default='jGeneName',
                        help='J gene column in external repertoire files')

    # Model hyperparameters
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Internal val fraction for alpha tuning (default: 0.2)')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='CV folds for C tuning (default: 5)')

    # Output
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')

    args = parser.parse_args()

    evaluator = ExternalEvaluator(
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
        ext_sequence_col=args.ext_sequence_col,
        ext_v_gene_col=args.ext_v_gene_col,
        ext_j_gene_col=args.ext_j_gene_col,
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
        ext_file_template=args.ext_file_template,
        output_csv=args.output_csv,
    )
