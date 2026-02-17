"""
Evaluation script for disease classification using demographic features only.

Uses age (raw), sex (binary), and ancestry (one-hot encoded) as features
with logistic regression. Evaluates whether demographic confounders alone
can predict disease status.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


class DemographicFeaturesEvaluator:
    """
    Evaluator for disease classification using demographic features only.

    Features:
    - age: raw numeric value
    - sex: binary (M=1, F=0)
    - ancestry: one-hot encoded over all categories observed in training data
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, train_val_ratio=0.9):
        self.train_val_ratio = train_val_ratio

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        # Drop rows with any missing demographic feature
        before = len(filtered)
        filtered = filtered.dropna(subset=['age', 'sex', 'ancestry'])
        # Also drop rows where ancestry is empty string
        filtered = filtered[filtered['ancestry'].str.strip() != '']
        after = len(filtered)

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()

        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease ({target_disease}): {n_disease} samples")
        print(f"  Healthy ({self.HEALTHY_LABEL}): {n_healthy} samples")
        print(f"  Total: {after} samples (dropped {before - after} with missing demographics)")

        return filtered

    def featurize(self, data, ancestry_categories=None):
        """
        Convert demographic columns into a numeric feature matrix.

        Args:
            data: DataFrame with 'age', 'sex', 'ancestry' columns
            ancestry_categories: List of ancestry categories for one-hot encoding.
                If None, derived from data (use for training). Pass training categories
                for test data to ensure consistent encoding.

        Returns:
            (feature_matrix as np.ndarray, ancestry_categories list)
        """
        # Age: raw numeric
        age = data['age'].values.astype(float)

        # Sex: binary (M=1, F=0)
        sex = (data['sex'] == 'M').astype(int).values

        # Ancestry: one-hot encoding
        if ancestry_categories is None:
            ancestry_categories = sorted(data['ancestry'].unique().tolist())

        ancestry_dummies = np.zeros((len(data), len(ancestry_categories)), dtype=float)
        for i, cat in enumerate(ancestry_categories):
            ancestry_dummies[:, i] = (data['ancestry'].values == cat).astype(float)

        # Stack all features: [age, sex, ancestry_0, ancestry_1, ...]
        features = np.column_stack([age, sex, ancestry_dummies])

        feature_names = ['age', 'sex'] + [f'ancestry_{cat}' for cat in ancestry_categories]

        return features, ancestry_categories, feature_names

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
            tuning_results.append({
                'C': C,
                'val_auroc': val_auroc,
                'val_aupr': val_aupr
            })
            print(f"  C={C}: Val AUROC={val_auroc:.4f}, Val AUPR={val_aupr:.4f}")

        best = max(tuning_results, key=lambda x: x['val_auroc'])
        best_C = best['C']
        print(f"\nBest C: {best_C} "
              f"(Val AUROC={best['val_auroc']:.4f}, Val AUPR={best['val_aupr']:.4f})")

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
        }

    def run_cross_validation(self, metadata_path, target_disease,
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7,
                              C_candidates=None):
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)

        results = {
            'target_disease': target_disease,
            'fold_results': [],
            'all_probs': [],
            'all_labels': []
        }

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

            # Featurize (derive ancestry categories from training data)
            X_train, ancestry_cats, feature_names = self.featurize(train_data)
            y_train = train_data['label'].values

            X_val, _, _ = self.featurize(val_data, ancestry_categories=ancestry_cats)
            y_val = val_data['label'].values

            X_test, _, _ = self.featurize(test_data, ancestry_categories=ancestry_cats)
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
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}")

            # Log model coefficients
            coefs = dict(zip(feature_names, model.coef_[0]))
            print(f"Model coefficients: {coefs}")

            fold_result = {
                'fold': test_fold,
                'n_train': len(train_data),
                'n_val': len(val_data),
                'n_test': len(test_data),
                'best_C': tuning_result['best_C'],
                'val_auroc': tuning_result['best_val_auroc'],
                'val_aupr': tuning_result['best_val_aupr'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_probs': test_probs.tolist(),
                'test_labels': y_test.tolist(),
                'coefficients': coefs,
                'tuning_result': tuning_result
            }
            results['fold_results'].append(fold_result)

            results['all_probs'].extend(test_probs.tolist())
            results['all_labels'].extend(y_test.tolist())

        # Overall metrics
        all_probs = np.array(results['all_probs'])
        all_labels = np.array(results['all_labels'])
        overall_auroc = roc_auc_score(all_labels, all_probs)
        overall_aupr = average_precision_score(all_labels, all_probs)

        results['overall_auroc'] = overall_auroc
        results['overall_aupr'] = overall_aupr

        print(f"\n{'='*60}")
        print(f"OVERALL CROSS-VALIDATION RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
        print(f"Mean Test AUROC: {np.mean([r['test_auroc'] for r in results['fold_results']]):.4f} "
              f"± {np.std([r['test_auroc'] for r in results['fold_results']]):.4f}")
        print(f"Mean Test AUPR:  {np.mean([r['test_aupr'] for r in results['fold_results']]):.4f} "
              f"± {np.std([r['test_aupr'] for r in results['fold_results']]):.4f}")
        print(f"Overall AUROC (all folds combined): {overall_auroc:.4f}")
        print(f"Overall AUPR (all folds combined):  {overall_aupr:.4f}")

        print(f"\nBest C per fold:")
        for r in results['fold_results']:
            print(f"  Fold {r['fold']}: C={r['best_C']}")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demographic Features Disease Classification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv file')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Target disease to classify (e.g., Lupus, T1D, HIV)')
    args = parser.parse_args()

    print("Demographic Features Disease Classification Evaluation")
    print("=" * 60)

    evaluator = DemographicFeaturesEvaluator(train_val_ratio=0.9)

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        n_folds=3,
        random_state=7,
        C_candidates=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    )

    print(f"\nOverall AUROC: {results['overall_auroc']:.4f}")
    print(f"Overall AUPR: {results['overall_aupr']:.4f}")
