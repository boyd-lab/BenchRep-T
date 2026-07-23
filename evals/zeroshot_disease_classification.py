"""Evaluate a saved Rawat T1D model on a labelled external cohort.

This script never trains or updates a model.  The target cohort's ``fold``
column is used only to report fold-stratified metrics in addition to metrics
over all available specimens.
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SIMPLE_METHODS = {
    "emerson", "ostmeyer", "ensemble_regression", "ensemble_xgboost", "abmil"
}
ALL_METHODS = sorted(SIMPLE_METHODS | {"deeprc", "deeptcr", "giana"})


def _load_metadata(args):
    metadata = pd.read_csv(args.metadata, sep="\t")
    required = {
        args.participant_col, args.specimen_col, args.disease_col, args.fold_col
    }
    absent = sorted(required - set(metadata.columns))
    if absent:
        raise ValueError(f"Metadata is missing required columns: {absent}")

    healthy_values = set(args.healthy_label)
    keep = (metadata[args.disease_col] == args.target_disease) | (
        metadata[args.disease_col].isin(healthy_values)
    )
    metadata = metadata.loc[keep].copy()
    metadata["label"] = (
        metadata[args.disease_col] == args.target_disease
    ).astype(int)
    metadata["sample_id"] = metadata[args.specimen_col].astype(str)

    if metadata["sample_id"].duplicated().any():
        dupes = metadata.loc[
            metadata["sample_id"].duplicated(keep=False), "sample_id"
        ].unique()
        raise ValueError(
            "Specimen identifiers must be unique for checkpoint inference; "
            f"duplicates include {dupes[:10].tolist()}"
        )

    def make_path(row):
        filename = args.file_template.format(
            participant_label=row[args.participant_col],
            specimen_label=row[args.specimen_col],
        )
        return os.path.abspath(os.path.join(args.repertoire_data_dir, filename))

    metadata["file_path"] = metadata.apply(make_path, axis=1)
    metadata["file_exists"] = metadata["file_path"].map(os.path.isfile)
    missing = metadata.loc[~metadata["file_exists"]].copy()
    os.makedirs(os.path.dirname(os.path.abspath(args.missing_csv)), exist_ok=True)
    missing.to_csv(args.missing_csv, index=False)
    if len(missing) and not args.allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} repertoire files are missing. See {args.missing_csv}"
        )
    metadata = metadata.loc[metadata["file_exists"]].drop(
        columns=["file_exists"]
    )

    if args.max_samples_per_class:
        metadata = (
            metadata.sort_values(
                [args.fold_col, "label", "sample_id"], kind="stable"
            )
            .groupby([args.fold_col, "label"], group_keys=False)
            .head(args.max_samples_per_class)
            .copy()
        )
    if metadata.empty:
        raise ValueError("No evaluable T1D/control specimens were found.")
    return metadata, missing


def _predict_simple(method, checkpoint, metadata, use_gpu):
    from evals.predict_saved_disease_model import load_model

    model = load_model(method, checkpoint, use_gpu)
    scores = {}
    for index, row in enumerate(metadata.itertuples(index=False), 1):
        result = model.predict_diagnosis(row.file_path)
        scores[row.sample_id] = float(result["probability_positive"])
        if index % 25 == 0 or index == len(metadata):
            print(f"  scored {index}/{len(metadata)} repertoires", flush=True)
    return scores


def _predict_deeprc(checkpoint, metadata, use_gpu, n_workers):
    import torch
    from torch.utils.data import DataLoader
    from evals.deeprc_2020_disease_classification import DeepRC2020Evaluator
    from dataset_readers import (
        AIRRRepertoireDataset,
        no_sequence_count_scaling,
        no_stack_collate_fn,
    )
    from task_definitions import BinaryTarget, TaskDefinition

    device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    evaluator = DeepRC2020Evaluator(device=device)
    task_definition = TaskDefinition(
        targets=[BinaryTarget(column_name="label", true_class_value="1")]
    )
    label_frame = pd.DataFrame({"label": metadata["label"].astype(str)})
    labels = task_definition.get_targets(label_frame)
    dataset = AIRRRepertoireDataset(
        file_paths=metadata["file_path"].tolist(),
        labels=labels,
        sample_ids=metadata["sample_id"].tolist(),
        sequence_col="cdr3_aa",
        count_col="duplicate_count",
        sample_n_sequences=None,
        sequence_counts_scaling_fn=no_sequence_count_scaling,
        keep_in_ram=False,
        verbose=True,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=n_workers,
        collate_fn=no_stack_collate_fn,
    )
    model = evaluator._build_model(task_definition)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    sample_ids, probabilities = evaluator._predict_proba(model, loader)
    return dict(zip(map(str, sample_ids), map(float, probabilities)))


def _predict_deeptcr(checkpoint, metadata, n_workers):
    from evals.deeptcr_2021_disease_classification import (
        DeepTCREvaluator,
        DeepTCR_WF,
    )

    evaluator = DeepTCREvaluator()
    scores = {}
    # Loading one target fold at a time bounds host-memory use.  These groups
    # do not alter the model and are only inference batches.
    for fold, fold_data in metadata.groupby("metric_fold", sort=True):
        print(f"  loading target fold {fold} ({len(fold_data)} repertoires)")
        collected = evaluator._collect_all_data(fold_data, "T1D")
        beta_sequences, sample_labels, _, counts, v_beta, j_beta, written = collected
        if beta_sequences is None:
            continue
        model = DeepTCR_WF(checkpoint, max_length=40, device=0)
        with mp.Pool(processes=n_workers) as pool:
            model.Sample_Inference(
                sample_labels=sample_labels,
                beta_sequences=beta_sequences,
                v_beta=v_beta,
                j_beta=j_beta,
                counts=counts,
                batch_size=10,
                models=["model_2"],
                p=pool,
            )
        if "T1D" not in model.Inference_Pred_Dict:
            raise RuntimeError(
                "DeepTCR checkpoint does not contain a T1D output class; "
                f"found {sorted(model.Inference_Pred_Dict)}"
            )
        pred = model.Inference_Pred_Dict["T1D"]
        scores.update(dict(zip(pred["Samples"].astype(str), pred["Pred"].astype(float))))
        del model
    return scores


def _predict_giana(checkpoint, metadata, work_dir, use_gpu, n_workers):
    from evals.giana_2021_disease_classification import (
        GIANAEvaluator,
        _load_giana_module,
    )

    os.makedirs(work_dir, exist_ok=True)
    train_file = os.path.join(checkpoint, "train_fold2.txt")
    ref_cluster = os.path.join(
        checkpoint, "train_fold2--RotationEncodingBL62.txt"
    )
    for path in (train_file, ref_cluster):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing GIANA checkpoint artifact: {path}")

    evaluator = GIANAEvaluator(
        results_dir=work_dir, use_gpu=use_gpu, n_threads=n_workers,
        max_seqs_per_specimen=10000,
    )
    lines = []
    readable = set()
    for index, row in enumerate(metadata.itertuples(index=False), 1):
        repertoire = evaluator._load_repertoire(row.file_path)
        if repertoire is not None and len(repertoire):
            lines.extend(
                evaluator._build_giana_rows(repertoire, "test", row.sample_id)
            )
            readable.add(row.sample_id)
        if index % 25 == 0 or index == len(metadata):
            print(f"  prepared {index}/{len(metadata)} repertoires", flush=True)
    if not lines:
        return {}
    query_file = os.path.join(work_dir, "external_query.txt")
    evaluator._write_giana_input(lines, query_file)
    giana = _load_giana_module()
    reference_data = giana.CreateReference(
        train_file, Vgene=evaluator.use_v_gene, ST=3
    )
    merged = evaluator._run_query(
        query_file, reference_data, ref_cluster, work_dir
    )
    assigned = (
        evaluator._compute_specimen_scores(merged, "T1D")
        if merged is not None else {}
    )
    # Match the method's prior cross-validation behavior: repertoires with no
    # reference-cluster assignment receive a score of zero.
    return {sample_id: float(assigned.get(sample_id, 0.0))
            for sample_id in readable}


def _metric_rows(predictions, method, cohort):
    rows = []
    groups = [
        (f"fold{fold}", frame)
        for fold, frame in predictions.groupby("metric_fold", sort=True)
    ]
    groups.append(("overall", predictions))
    for split, frame in groups:
        y_true = frame["label"].to_numpy(dtype=int)
        y_score = frame["model_score"].to_numpy(dtype=float)
        two_classes = np.unique(y_true).size == 2
        rows.append({
            "cohort": cohort,
            "method": method,
            "evaluation_split": split,
            "n": len(frame),
            "n_t1d": int(y_true.sum()),
            "n_healthy": int((y_true == 0).sum()),
            "auroc": roc_auc_score(y_true, y_score) if two_classes else np.nan,
            "aupr": average_precision_score(y_true, y_score)
            if two_classes else np.nan,
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_score >= 0.5
            ) if two_classes else np.nan,
            "f1": f1_score(y_true, y_score >= 0.5, zero_division=0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--repertoire_data_dir", required=True)
    parser.add_argument("--file_template", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output_predictions", required=True)
    parser.add_argument("--output_metrics", required=True)
    parser.add_argument("--missing_csv", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--target_disease", default="T1D")
    parser.add_argument(
        "--healthy_label", action="append",
        default=["Healthy/Background", "Healthy"],
    )
    parser.add_argument("--participant_col", default="participant_label")
    parser.add_argument("--specimen_col", default="specimen_label")
    parser.add_argument("--disease_col", default="disease")
    parser.add_argument("--fold_col", default="fold")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--max_samples_per_class", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    metadata, missing = _load_metadata(args)
    metadata["metric_fold"] = metadata[args.fold_col]
    print(
        f"Evaluating {args.method} on {len(metadata)} available specimens "
        f"({len(missing)} missing); no training will be performed."
    )

    use_gpu = not args.no_gpu
    if args.method in SIMPLE_METHODS:
        scores = _predict_simple(
            args.method, args.checkpoint, metadata, use_gpu
        )
    elif args.method == "deeprc":
        scores = _predict_deeprc(
            args.checkpoint, metadata, use_gpu, args.n_workers
        )
    elif args.method == "deeptcr":
        scores = _predict_deeptcr(
            args.checkpoint, metadata, args.n_workers
        )
    else:
        scores = _predict_giana(
            args.checkpoint, metadata, args.work_dir, use_gpu, args.n_workers
        )

    predictions = metadata.copy()
    predictions["model_score"] = predictions["sample_id"].map(scores)
    no_score = predictions["model_score"].isna()
    if no_score.any():
        failed_path = os.path.join(args.work_dir, "unscored_specimens.csv")
        os.makedirs(args.work_dir, exist_ok=True)
        predictions.loc[no_score].to_csv(failed_path, index=False)
        print(
            f"Warning: {int(no_score.sum())} available repertoires could not "
            f"be scored; see {failed_path}"
        )
        predictions = predictions.loc[~no_score].copy()
    if predictions.empty:
        raise RuntimeError("The checkpoint produced no specimen predictions.")

    predictions.insert(0, "method", args.method)
    predictions.insert(0, "cohort", args.cohort)
    predictions["checkpoint"] = os.path.abspath(args.checkpoint)
    metrics = _metric_rows(predictions, args.method, args.cohort)
    for output in (args.output_predictions, args.output_metrics):
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    predictions.to_csv(args.output_predictions, index=False)
    metrics.to_csv(args.output_metrics, index=False)
    print(metrics.to_string(index=False))
    print(f"Wrote predictions: {args.output_predictions}")
    print(f"Wrote metrics: {args.output_metrics}")


if __name__ == "__main__":
    main()
