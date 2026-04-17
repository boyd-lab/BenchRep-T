"""
MotifBoost wrapper conforming to the train / predict_diagnosis interface
used by motifboost_2021_disease_classification.py.

Loads AIRR .tsv.gz repertoire files into Repertoire objects and delegates
to MotifBoostClassifier for feature extraction and classification.
"""

import os
import numpy as np
import pandas as pd

from .motif import MotifBoostClassifier
from .repertoire import Repertoire


class MotifBoost:
    def __init__(
        self,
        sequence_col='cdr3_aa',
        count_col='duplicate_count',
        ngram_range=(3, 4),
        classifier_method='optuna-lightgbm',
        count_weight_mode=True,
        tfidf_mode=False,
        augmentation_times=5,
        augmentation_rate=0.5,
        n_jobs=None,
        subsample_fraction=1.0,
        subsample_seed=7,
    ):
        self.sequence_col = sequence_col
        self.count_col = count_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self._classifier = MotifBoostClassifier(
            count_weight_mode=count_weight_mode,
            tfidf_mode=tfidf_mode,
            augmentation_times=augmentation_times,
            augmentation_rate=augmentation_rate,
            ngram_range=ngram_range,
            classifier_method=classifier_method,
            n_jobs=n_jobs,
        )

    def _load_repertoire(self, file_path):
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        if self.subsample_fraction < 1.0:
            df = df.sample(frac=self.subsample_fraction, random_state=self.subsample_seed)

        if self.sequence_col not in df.columns:
            return None
        df = df.dropna(subset=[self.sequence_col])
        sequences = df[self.sequence_col].tolist()
        if not sequences:
            return None

        if self.count_col in df.columns:
            counts = pd.to_numeric(df[self.count_col], errors='coerce').fillna(1).clip(lower=1).astype(int).tolist()
        else:
            counts = [1] * len(sequences)

        sample_id = os.path.basename(file_path)
        return Repertoire(
            experiment_id='airr_bench',
            sample_id=sample_id,
            info={},
            sequences=sequences,
            counts=counts,
        )

    def train(self, train_files, train_labels):
        """
        Load repertoires from file paths and fit the classifier.

        Returns:
            Dict with n_train (number of successfully loaded repertoires).
        """
        repertoires = []
        labels = []
        for fp, label in zip(train_files, train_labels):
            rep = self._load_repertoire(fp)
            if rep is not None:
                repertoires.append(rep)
                labels.append(label)

        self._classifier.fit(repertoires, labels)
        return {'n_train': len(repertoires)}

    def predict_diagnosis(self, file_path):
        """
        Predict disease probability for a single repertoire file.

        Returns:
            Dict with probability_positive (float in [0, 1]).
        """
        rep = self._load_repertoire(file_path)
        if rep is None:
            return {'probability_positive': 0.5}

        proba = self._classifier.predict_proba([rep])
        prob = float(proba[0, 1])
        return {'probability_positive': prob}
