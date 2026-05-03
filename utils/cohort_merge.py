"""
Runtime helper for merging an external cohort into an internal evaluator's
already-prepared metadata DataFrame.

Used by every disease-classification evaluator under ``evals/`` when
``--ext_metadata_path`` is provided. The external metadata is assumed to be
in MAL-ID column style (run
``external_data_process/harmonize_t1d_metadata.py`` once to convert legacy
external files).

The merged DataFrame contains both internal and external rows, each with a
populated ``file_path``, ``label``, and ``cohort`` column, so downstream
fold-based CV code can iterate over it without further branching.
"""

import os
import re

import pandas as pd


HEALTHY_LABEL = 'Healthy/Background'
DISEASE_COL = 'disease'
FOLD_COL = 'malid_cross_validation_fold_id_when_in_test_set'

# DenverT1D sample IDs in metadata are unpadded (DenverT1D-9) but the
# external repertoire files are zero-padded to width 3 (DenverT1D-009_TCRB.tsv).
_DENVER_PATTERN = re.compile(r'^(DenverT1D-)(\d+)$')


def _resolve_external_path(row, ext_data_dir, ext_file_template):
    """Construct an external file path, handling DenverT1D zero-padding."""
    values = row.to_dict()
    sample_name = values.get('sample_name') or values.get('participant_label')
    values.setdefault('participant_label', sample_name)
    values.setdefault('specimen_label', values.get('participant_label'))
    values.setdefault('sample_name', sample_name)

    plain = ext_file_template.format(**values)
    plain_path = os.path.join(ext_data_dir, plain)
    if os.path.exists(plain_path):
        return plain_path

    m = _DENVER_PATTERN.match(sample_name)
    if m:
        prefix, num = m.groups()
        padded = f"{prefix}{int(num):03d}"
        padded_values = values.copy()
        padded_values['participant_label'] = padded
        if padded_values.get('specimen_label') == sample_name:
            padded_values['specimen_label'] = padded
        if padded_values.get('sample_name') == sample_name:
            padded_values['sample_name'] = padded
        padded_name = ext_file_template.format(**padded_values)
        padded_path = os.path.join(ext_data_dir, padded_name)
        if os.path.exists(padded_path):
            return padded_path

    return plain_path  # return non-existent path so caller can drop it uniformly


def prepare_merged_cohort(internal_metadata,
                          ext_metadata_path,
                          ext_data_dir,
                          target_disease,
                          ext_file_template='{participant_label}_TCRB.tsv',
                          healthy_label=HEALTHY_LABEL,
                          fold_col=FOLD_COL,
                          disease_col=DISEASE_COL):
    """Merge external samples into an evaluator's internal metadata DataFrame.

    Args:
        internal_metadata: DataFrame already filtered to disease + healthy rows
            with ``label`` populated. If repertoire files are needed (i.e. the
            caller already populated ``file_path``), pass ``ext_data_dir`` to
            resolve external paths; pass ``None`` for metadata-only evaluators
            (e.g. demographic-features).
        ext_metadata_path: Path to external metadata TSV (MAL-ID column style).
        ext_data_dir: Directory containing external repertoire files. Pass
            ``None`` to skip file-path resolution.
        target_disease: Disease label being classified (e.g. 'T1D').
        ext_file_template: Format string for external filenames. Supported
            placeholders: ``{participant_label}``, ``{specimen_label}``,
            ``{sample_name}`` (alias of ``{participant_label}``).
        healthy_label, fold_col, disease_col: Column / value conventions
            (defaults match MAL-ID).

    Returns:
        Concatenated DataFrame with internal + external rows. Each row has a
        ``cohort`` column (``'internal'`` or ``'external'``) plus the same
        ``file_path``, ``label``, and ``fold_col`` columns the evaluator
        already expects.
    """
    internal = internal_metadata.copy()
    if 'cohort' not in internal.columns:
        internal['cohort'] = 'internal'

    ext = pd.read_csv(ext_metadata_path, sep='\t')

    required_cols = {disease_col, fold_col, 'participant_label'}
    missing = required_cols - set(ext.columns)
    if missing:
        raise ValueError(
            f"External metadata at {ext_metadata_path} is missing required "
            f"columns: {sorted(missing)}. Run "
            f"external_data_process/harmonize_t1d_metadata.py first."
        )

    mask = ext[disease_col].isin([target_disease, healthy_label])
    ext = ext[mask].copy()
    ext = ext[ext[fold_col].notna()].copy()
    ext[fold_col] = ext[fold_col].astype(int)

    ext['label'] = (ext[disease_col] == target_disease).astype(int)

    if ext_data_dir is not None:
        ext['file_path'] = ext.apply(
            lambda row: _resolve_external_path(row, ext_data_dir, ext_file_template),
            axis=1,
        )
        before = len(ext)
        ext = ext[ext['file_path'].apply(os.path.exists)].copy()
        missing = before - len(ext)
        if missing > 0:
            print(f"  External cohort: {missing} of {before} repertoire files "
                  f"not found in {ext_data_dir}; proceeding with {len(ext)}.")

    ext['cohort'] = 'external'

    n_int_pos = int((internal['label'] == 1).sum())
    n_int_neg = int((internal['label'] == 0).sum())
    n_ext_pos = int((ext['label'] == 1).sum())
    n_ext_neg = int((ext['label'] == 0).sum())
    print(f"\nMerged cohort for '{target_disease}' classification:")
    print(f"  Internal: {n_int_pos} {target_disease} + {n_int_neg} healthy "
          f"= {len(internal)}")
    print(f"  External: {n_ext_pos} {target_disease} + {n_ext_neg} healthy "
          f"= {len(ext)}")
    print(f"  Total:    {len(internal) + len(ext)}")

    merged = pd.concat([internal, ext], ignore_index=True, sort=False)
    return merged
