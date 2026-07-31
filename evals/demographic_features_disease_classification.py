"""
Evaluation script for disease classification using demographic features only.

Uses configurable demographic features with logistic regression. Age is
represented by its raw numeric value, sex is binary, and other selected
columns are one-hot encoded. Evaluates whether measured covariates alone can
predict disease status.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class DemographicFeaturesEvaluator:
    """
    Evaluator for disease classification using demographic features only.

    Feature encoding:
    - age: raw numeric value
    - sex: binary (M=1, F=0)
    - all other selected columns: one-hot encoded over categories observed in
      the training data
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(
        self,
        train_val_ratio=0.9,
        feature_columns=None,
        complete_case_columns=None,
    ):
        self.train_val_ratio = train_val_ratio
        self.feature_columns = (
            list(feature_columns)
            if feature_columns is not None
            else ['age', 'sex', 'ancestry']
        )
        self.complete_case_columns = (
            list(complete_case_columns)
            if complete_case_columns is not None
            else list(self.feature_columns)
        )
        if not self.feature_columns:
            raise ValueError("At least one demographic feature is required.")
        if not self.complete_case_columns:
            raise ValueError(
                "At least one complete-case column is required."
            )

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        required_columns = list(
            dict.fromkeys(
                self.feature_columns + self.complete_case_columns
            )
        )
        missing_columns = [
            column for column in required_columns
            if column not in filtered.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Metadata is missing requested feature columns: {missing_columns}"
            )

        # Restrict to the requested common complete-case cohort. By default
        # this is simply the set of fitted features, while feature-ablation
        # analyses can supply a shared superset for like-for-like evaluation.
        before = len(filtered)
        filtered = filtered.dropna(subset=self.complete_case_columns)
        for column in self.complete_case_columns:
            if pd.api.types.is_object_dtype(filtered[column]):
                filtered = filtered[filtered[column].str.strip() != '']
        after = len(filtered)

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()

        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease ({target_disease}): {n_disease} samples")
        print(f"  Healthy ({self.HEALTHY_LABEL}): {n_healthy} samples")
        print(f"  Total: {after} samples (dropped {before - after} with missing demographics)")

        return filtered

    def featurize(self, data, categorical_categories=None):
        """
        Convert selected demographic columns into a numeric feature matrix.

        Args:
            data: DataFrame containing the selected feature columns.
            categorical_categories: Mapping from categorical column name to
                training categories. If None, categories are derived from data.
                Pass the training mapping for validation and test data to ensure
                consistent encoding.

        Returns:
            (feature_matrix, categorical category mapping, feature names)
        """
        if categorical_categories is None:
            categorical_categories = {}

        feature_arrays = []
        feature_names = []
        fitted_categories = {}

        for column in self.feature_columns:
            if column == 'age':
                feature_arrays.append(data[column].values.astype(float)[:, None])
                feature_names.append(column)
                continue

            if column == 'sex':
                feature_arrays.append(
                    (data[column] == 'M').astype(int).values[:, None]
                )
                feature_names.append(column)
                continue

            categories = categorical_categories.get(column)
            if categories is None:
                categories = sorted(data[column].unique().tolist())
            fitted_categories[column] = categories

            dummies = np.zeros((len(data), len(categories)), dtype=float)
            for index, category in enumerate(categories):
                dummies[:, index] = (
                    data[column].values == category
                ).astype(float)
            feature_arrays.append(dummies)
            feature_names.extend(
                [f'{column}_{category}' for category in categories]
            )

        return (
            np.column_stack(feature_arrays),
            fitted_categories,
            feature_names,
        )

    def tune_and_train(self, X_train, y_train, X_val, y_val,
                       C_candidates=None):
        """
        Tune regularization strength C on validation set, then retrain on
        train+val with the best C.
        """
        if C_candidates is None:
            C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        print("--- Parameter Tuning ---")
        print(f"Testing C values: {C_candidates}")

        tuning_results = []
        for C in C_candidates:
            model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            val_auroc = roc_auc_score(y_val, val_probs)
            val_aupr = average_precision_score(y_val, val_probs)
            val_preds = (val_probs >= 0.5).astype(int)
            val_balanced_acc = balanced_accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds)
            tuning_results.append({
                'C': C,
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
                'val_balanced_acc': val_balanced_acc,
                'val_f1': val_f1,
            })
            print(f"  C={C}: Val AUROC={val_auroc:.4f}, Val AUPR={val_aupr:.4f}, "
                  f"Balanced Acc={val_balanced_acc:.4f}, F1={val_f1:.4f}")

        best = max(tuning_results, key=lambda x: x['val_auroc'])
        best_C = best['C']
        print(f"\nBest C: {best_C} "
              f"(Val AUROC={best['val_auroc']:.4f}, Val AUPR={best['val_aupr']:.4f}, "
              f"Balanced Acc={best['val_balanced_acc']:.4f}, F1={best['val_f1']:.4f})")

        # Retrain on train+val with best C
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        final_model = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs')
        final_model.fit(X_combined, y_combined)

        return final_model, {
            'tuning_results': tuning_results,
            'best_C': best_C,
            'best_val_auroc': best['val_auroc'],
            'best_val_aupr': best['val_aupr'],
            'best_val_balanced_acc': best['val_balanced_acc'],
            'best_val_f1': best['val_f1'],
        }

    def run_cross_validation(self, metadata_path, target_disease,
                              disease_col='disease',
                              participant_col='participant_label',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7,
                              C_candidates=None,
                              ext_metadata_path=None):
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)

        if ext_metadata_path is not None:
            from utils.cohort_merge import prepare_merged_cohort
            metadata = prepare_merged_cohort(
                metadata, ext_metadata_path, ext_data_dir=None,
                target_disease=target_disease,
                healthy_label=self.HEALTHY_LABEL,
                fold_col=fold_col, disease_col=disease_col,
            )

        all_test_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            train_val_data = metadata[~test_mask]
            test_data = metadata[test_mask]

            # Split train/val
            train_data, val_data = train_test_split(
                train_val_data,
                train_size=self.train_val_ratio,
                random_state=random_state,
                stratify=train_val_data['label']
            )

            print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

            # Featurize (derive categorical levels from training data).
            X_train, categorical_categories, feature_names = self.featurize(
                train_data
            )
            y_train = train_data['label'].values

            X_val, _, _ = self.featurize(
                val_data,
                categorical_categories=categorical_categories,
            )
            y_val = val_data['label'].values

            X_test, _, _ = self.featurize(
                test_data,
                categorical_categories=categorical_categories,
            )
            y_test = test_data['label'].values

            print(f"Features ({len(feature_names)}): {feature_names}")

            # Tune and train
            model, tuning_result = self.tune_and_train(
                X_train, y_train, X_val, y_val, C_candidates=C_candidates
            )

            # Evaluate on test set
            test_probs = model.predict_proba(X_test)[:, 1]
            test_auroc = roc_auc_score(y_test, test_probs)
            test_aupr = average_precision_score(y_test, test_probs)
            test_preds = (test_probs >= 0.5).astype(int)
            test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                  f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

            # Log model coefficients
            coefs = dict(zip(feature_names, model.coef_[0]))
            print(f"Model coefficients: {coefs}")

            # Build per-sample rows for output DataFrame
            for (_, row), score in zip(test_data.iterrows(), test_probs):
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': row['specimen_label'],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'method': 'Demographic_Features',
                    'disease_model': target_disease,
                    'model_score': float(score),
                    'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                })

            fold_results.append({
                'fold': test_fold,
                'n_train': len(train_data),
                'n_val': len(val_data),
                'n_test': len(test_data),
                'best_C': tuning_result['best_C'],
                'val_auroc': tuning_result['best_val_auroc'],
                'val_aupr': tuning_result['best_val_aupr'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
                'coefficients': coefs,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(y_test.tolist())

        # Overall metrics
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)
        overall_preds = (all_probs_arr >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(all_labels_arr, overall_preds)
        overall_f1 = f1_score(all_labels_arr, overall_preds)

        print(f"\n{'='*60}")
        print(f"OVERALL CROSS-VALIDATION RESULTS: {target_disease} vs Healthy")
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

        print(f"\nBest C per fold:")
        for r in fold_results:
            print(f"  Fold {r['fold']}: C={r['best_C']}")

        return pd.DataFrame(all_test_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demographic Features Disease Classification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv file')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Target disease to classify (e.g., Lupus, T1D, HIV)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--ext_metadata_path', type=str, default=None,
                        help='Optional external-cohort metadata TSV (MAL-ID column style). '
                             'Merges external samples by fold (no repertoire files needed).')
    parser.add_argument(
        '--features',
        nargs='+',
        default=['age', 'sex', 'ancestry'],
        help='Demographic columns to use. Age is raw numeric, sex is binary, '
             'and all other columns are one-hot encoded.',
    )
    parser.add_argument(
        '--complete_case_features',
        nargs='+',
        default=None,
        help='Columns required to be non-missing for every feature ablation. '
             'Defaults to --features.',
    )
    parser.add_argument(
        '--fold_col',
        default='malid_cross_validation_fold_id_when_in_test_set',
        help='Metadata column containing the pre-assigned test fold.',
    )
    args = parser.parse_args()

    print("Demographic Features Disease Classification Evaluation")
    print("=" * 60)

    evaluator = DemographicFeaturesEvaluator(
        train_val_ratio=0.9,
        feature_columns=args.features,
        complete_case_columns=args.complete_case_features,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        fold_col=args.fold_col,
        n_folds=3,
        random_state=7,
        C_candidates=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        ext_metadata_path=args.ext_metadata_path,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
