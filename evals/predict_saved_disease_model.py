"""Run inference from a saved AIRR-Bench disease model without retraining.

Supports the models with standalone Python save/load APIs. DeepRC and DeepTCR
use their native checkpoint/inference loaders; GIANA uses its saved reference
cluster and query workflow.
"""

import argparse
import os

import pandas as pd


def load_model(method, checkpoint, use_gpu):
    if method == 'emerson':
        from models.emerson_2017 import CMV_Immunosequencing_Model
        return CMV_Immunosequencing_Model.load(checkpoint)
    if method == 'ostmeyer':
        from models.ostmeyer_2019 import MIL_TCR_Classifier
        return MIL_TCR_Classifier.load(checkpoint)
    if method == 'ensemble_regression':
        from models.ensemble_regression import Gapped_4mer_VJgene
        return Gapped_4mer_VJgene.load(checkpoint)
    if method == 'ensemble_xgboost':
        from models.ensemble_xgboost import XGBoostKmer
        return XGBoostKmer.load(checkpoint)
    if method == 'abmil':
        from models.ensemble_abmil import ABMIL
        return ABMIL.load(checkpoint, use_gpu=use_gpu)
    raise ValueError(f'Unsupported method: {method}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=[
        'emerson', 'ostmeyer', 'ensemble_regression',
        'ensemble_xgboost', 'abmil'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--repertoire_paths', required=True, nargs='+')
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    model = load_model(args.method, args.checkpoint, not args.no_gpu)
    rows = []
    for path in args.repertoire_paths:
        result = model.predict_diagnosis(path)
        row = {'method': args.method, 'repertoire_path': os.path.abspath(path)}
        row.update({key: value for key, value in result.items()
                    if isinstance(value, (str, int, float, bool)) or value is None})
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f'Wrote {len(rows)} predictions to {args.output_csv}')


if __name__ == '__main__':
    main()
