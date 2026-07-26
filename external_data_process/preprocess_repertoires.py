"""
Preprocess external ImmunoSEQ repertoire files so they match internal (AIRR/IMGT) conventions
used by Mal-ID and Mal-ID-Lite.

Transformations applied:
  1.  Normalize camelCase and snake_case immunoSEQ export schemas.
  2.  Filter to productive sequences: keep only rows with sequenceStatus == "In".
  3.  Rename columns: aminoAcid → cdr3_aa, vGeneName → v_call, jGeneName → j_call.
  4.  Drop rows with "unresolved" v_call, j_call, or cdr3_aa (case-insensitive).
  5.  Remap V/J gene names from Adaptive to AIRR and strip alleles.
  6.  Harmonize orphon names: "-or9_2" → "/OR9-2".
  7.  Collapse indistinguishable V genes (TRBV12-4 → TRBV12-3, TRBV6-3 → TRBV6-2),
      mirroring preprocessing/clean_tcr_data.py
  8.  Call collapse_imgt_singleton() to strip "-1" from singleton TRBV families
      (e.g. TRBV13-1 → TRBV13), matching the IgBLAST/Mal-ID convention.
  9.  Trim CDR3 sequences: remove first and last amino acid (conserved C and F/W).
  10. Drop rows with missing/empty v_call, j_call, or cdr3_aa.
  11. Drop rows whose CDR3 contains non-standard AA characters (*, X, gaps, etc.).
  12. Drop rows lacking gene-level V/J resolution: family-only calls (e.g. TRBV20,
      TRBJ2) and "unknown". A V call is gene-level if it has a subgroup hyphen
      (TRBV7-2), is an orphon (TRBV20/OR9-2), or is a recognized singleton family
      (TRBV28); a J call is gene-level if it has a subgroup (TRBJ2-3).
  13. Add "sequence" and "num_reads" standardized columns.
  15. If --metadata_file is provided:
      a. Add "repertoire_id" column derived from output filename (matches specimen_label
         in metadata)
      b. Add "participant_label" column looked up from metadata
  16. If --strip_filename_TCRB_suffix is set, remove "_TCRB" from output filenames

Arguments:
  --input_dir               Directory containing raw external repertoire TSV files
  --output_dir              Output directory (default: data/external_processed_v2/)
  --file_glob               Glob pattern for input files (default: *_TCRB.tsv)
  --metadata_file           Path to metadata TSV with specimen_label and participant_label
                            columns. Enables repertoire_id/participant_label columns and
                            metadata-based file filtering.
  --process_all             Process all files even without metadata match (default: False).
                            Unmatched files get empty repertoire_id/participant_label.
                            No effect if --metadata_file is not provided.
  --strip_filename_TCRB_suffix  Remove "_TCRB" from output filenames (for T1D)
  --emerson_cmv             Apply the memory-safe Emerson CMV preset.
  --read_chunksize          Bound input rows held in memory at once.
  --compare_gene_dir        Compare processed V/J names with a reference dataset.
  --n_jobs                  Number of parallel processes (default: 4). Set to 1
                            for serial execution.

Usage examples for each disease:

  # T1D (strip _TCRB from filenames so they match specimen_label in metadata):
  python external_data_process/preprocess_repertoires.py \\
      --input_dir data/external_raw/T1D/ \\
      --file_glob "*_TCRB.tsv" \\
      --strip_filename_TCRB_suffix \\
      --metadata_file data/external_metadata/metadata_T1D_final.tsv

  # Rheumatoid Arthritis:
  python external_data_process/preprocess_repertoires.py \\
      --input_dir data/external_raw/rheumatoid_arthritis/ \\
      --file_glob "*.tsv" \\
      --metadata_file data/external_metadata/metadata_RA_final.tsv

  # Tuberculosis:
  python external_data_process/preprocess_repertoires.py \\
      --input_dir data/external_raw/tuberculosis/ \\
      --file_glob "*_TCRB.tsv" \\
      --metadata_file data/external_metadata/metadata_Tb_final.tsv

  # Emerson CMV (memory-safe preset; IDs are derived from each filename):
  python external_data_process/preprocess_repertoires.py --emerson_cmv

  # Emerson CMV plus a V/J vocabulary comparison against Mal-ID:
  python external_data_process/preprocess_repertoires.py \\
      --emerson_cmv \\
      --compare_gene_dir data/malid_clean/TCR
"""

import os
import re
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.gene_harmonization import (
    IMGT_SINGLETON_TRBV,
    adaptive_to_airr,
    collapse_imgt_singleton,
    strip_allele,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Adaptive → AIRR column name mapping
COLUMN_RENAME = {
    'aminoAcid': 'cdr3_aa',
    'vGeneName': 'v_call',
    'jGeneName': 'j_call',
}

# Some Adaptive/immunoSEQ exports use snake_case column names (the "v2" / sample
# export variant, e.g. Rawat_T1D) instead of the camelCase names the rest of this
# script expects. Normalize snake_case → camelCase up front (Step 0) so the
# downstream pipeline is unchanged. Only applied when the camelCase target is
# absent, so it is a no-op on the original camelCase exports.
# For V/J we map the gene-level columns (v_gene/j_gene), matching how vGeneName/
# jGeneName behave; alleles are stripped downstream regardless.
INPUT_COLUMN_NORMALIZE = {
    'amino_acid':    'aminoAcid',
    'v_gene':        'vGeneName',
    'j_gene':        'jGeneName',
    'frame_type':    'sequenceStatus',
    'rearrangement': 'nucleotide',
    'templates':     'count (templates/reads)',
}

# Indistinguishable V genes due to FR3 primers — mirrors GENE_FIXES in
# preprocessing/clean_tcr_data.py. Keys are gene names *after* allele stripping.
GENE_COLLAPSES = {
    'TRBV12-4': 'TRBV12-3',
    'TRBV6-3': 'TRBV6-2',
}

VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# The Emerson immuneACCESS files repeat dozens of sample-level fields on every
# clonotype row and total hundreds of GB. The preset reads only fields needed
# by the repertoire models and writes the compact standardized schema.
MODEL_INPUT_COLUMNS = {
    # Camel-case Adaptive schema.
    'aminoAcid', 'vGeneName', 'jGeneName', 'sequenceStatus',
    'nucleotide', 'count (templates/reads)',
    # Snake-case/sample-export schema used by Emerson.
    'amino_acid', 'v_gene', 'j_gene', 'frame_type',
    'rearrangement', 'templates',
}

COMPACT_OUTPUT_COLUMNS = [
    'cdr3_aa', 'v_call', 'j_call', 'sequence', 'num_reads',
    'repertoire_id', 'participant_label',
]

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EMERSON_CMV_INPUT_DIR = os.path.join(
    REPO_DIR, 'data', 'external_datasets', 'CMV', 'raw'
)
EMERSON_CMV_OUTPUT_DIR = os.path.join(
    REPO_DIR, 'data', 'external_datasets', 'CMV', 'processed'
)


# ---------------------------------------------------------------------------
# Gene name helpers
# ---------------------------------------------------------------------------

def trim_cdr3(seq):
    """Remove the first and last amino acid from a CDR3 sequence."""
    if not isinstance(seq, str) or len(seq) <= 2:
        return seq
    return seq[1:-1]


def harmonize_gene(gene_name):
    """Convert Adaptive gene name to AIRR format, then strip allele."""
    return strip_allele(adaptive_to_airr(gene_name))


def collapse_indistinguishable_v(gene_name):
    """Collapse V genes that FR3 primers cannot disambiguate."""
    if not isinstance(gene_name, str):
        return gene_name
    return GENE_COLLAPSES.get(gene_name, gene_name)


def harmonize_orphon(gene_name):
    """Convert Adaptive orphon naming to IMGT format.

    Adaptive uses "-or9_2" suffix, IMGT uses "/OR9-2".
    Examples:
        TRBV20-or9_2 → TRBV20/OR9-2
        TRBVA-or9_2  → TRBVA/OR9-2
        TRBV7-9      → TRBV7-9  (not an orphon, untouched)
    """
    if not isinstance(gene_name, str):
        return gene_name
    return re.sub(
        r'-or9[_-]2$', '/OR9-2', gene_name, flags=re.IGNORECASE
    )


def cdr3_is_valid(seq):
    """True if seq is a non-empty string of standard amino acids."""
    if not isinstance(seq, str) or seq == '':
        return False
    return set(seq.upper()).issubset(VALID_AMINO_ACIDS)


def is_unresolved(value):
    """True for an unresolved sentinel, ignoring whitespace and case."""
    return isinstance(value, str) and value.strip().casefold() == 'unresolved'


def is_gene_level_v(gene_name):
    """True if a harmonized v_call is resolved to a specific gene.

    Gene-level V calls carry a subgroup hyphen (e.g. TRBV7-2), are an orphon
    (e.g. TRBV20/OR9-2, which also contains a hyphen), or are a recognized
    singleton family whose gene and family coincide (e.g. TRBV28, TRBV13).
    Family-only calls (TRBV20, TRBV4, ...) and 'unknown' are not gene-level.
    """
    if not isinstance(gene_name, str):
        return False
    return '-' in gene_name or gene_name in IMGT_SINGLETON_TRBV


def is_gene_level_j(gene_name):
    """True if a harmonized j_call is resolved to a specific gene.

    All functional TRBJ genes carry a subgroup (TRBJ1-x / TRBJ2-x), so a call
    without a hyphen (e.g. TRBJ2, 'unknown') is family-only / unresolved.
    """
    return isinstance(gene_name, str) and '-' in gene_name


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata(metadata_path):
    """Load metadata file and build specimen_label → participant_label mapping.

    Args:
        metadata_path: Path to a TSV metadata file with at least
            'specimen_label' and 'participant_label' columns.

    Returns:
        (specimen_labels, specimen_to_participant) where specimen_labels is a
        set of valid specimen labels, and specimen_to_participant is a dict
        mapping specimen_label → participant_label.
    """
    df = pd.read_csv(metadata_path, sep='\t')

    for col in ('specimen_label', 'participant_label'):
        if col not in df.columns:
            raise ValueError(
                f"Metadata file is missing required column '{col}'. "
                f"Available columns: {list(df.columns)}"
            )

    # Check for NaN values in required columns
    for col in ('specimen_label', 'participant_label'):
        n_na = df[col].isna().sum()
        if n_na > 0:
            raise ValueError(
                f"Metadata column '{col}' has {n_na} NaN values. "
                f"All entries must be non-null."
            )

    # Check for duplicate specimen_labels
    dupes = df['specimen_label'].duplicated()
    if dupes.any():
        dupe_vals = df.loc[dupes, 'specimen_label'].unique().tolist()
        raise ValueError(
            f"Metadata has {dupes.sum()} duplicate specimen_label entries: "
            f"{dupe_vals[:10]}{'...' if len(dupe_vals) > 10 else ''}"
        )

    specimen_labels = set(df['specimen_label'].astype(str))
    specimen_to_participant = dict(
        zip(df['specimen_label'].astype(str), df['participant_label'].astype(str))
    )

    return specimen_labels, specimen_to_participant


def derive_specimen_label(filename):
    """Derive specimen_label from an output filename by removing the .tsv extension.

    The _TCRB suffix (if any) should already be stripped from the filename before
    calling this function (via --strip_filename_TCRB_suffix handling in main()).

    Args:
        filename: Output filename (e.g. "310101.tsv", "HC1.tsv",
            "01-0935_D0_TCRB.tsv").

    Returns:
        Specimen label string (e.g. "310101", "HC1", "01-0935_D0_TCRB").
    """
    if filename.endswith('.tsv'):
        return filename[:-4]
    return filename


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df, input_basename, repertoire_id=None,
                         participant_label=None, compact_output=False):
    """Apply the repertoire transformations to one in-memory dataframe/chunk."""
    n_in = len(df)

    # --- Step 0: Normalize snake_case Adaptive columns to camelCase ---
    # Handles the "v2" / sample-export variant (amino_acid/v_gene/j_gene/
    # frame_type/rearrangement/templates). No-op for camelCase exports.
    normalize_map = {
        src: dst for src, dst in INPUT_COLUMN_NORMALIZE.items()
        if src in df.columns and dst not in df.columns
    }
    if normalize_map:
        df = df.rename(columns=normalize_map)

    # --- Step 1: Filter to productive sequences (sequenceStatus == "In") ---
    n_non_productive = 0
    if 'sequenceStatus' in df.columns:
        mask = df['sequenceStatus'] == 'In'
        n_non_productive = int((~mask).sum())
        df = df[mask].copy()
    else:
        print(f"  WARNING: 'sequenceStatus' column not found in {input_basename}, "
              f"skipping productive filter")

    # --- Step 2: Rename columns ---
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    required = {'cdr3_aa', 'v_call', 'j_call'}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(
            f"{input_basename} is missing required repertoire columns after "
            f"schema normalization: {missing_required}. Available columns: "
            f"{list(df.columns)}"
        )

    # --- Step 3: Drop unresolved calls before transforming their strings ---
    # This is case-insensitive and row-based, so a row with both unresolved V
    # and J calls is counted once. Doing this before CDR3 trimming also avoids
    # turning a literal "unresolved" CDR3 into "nresolve".
    unresolved_mask = pd.Series(False, index=df.index)
    for col in ('v_call', 'j_call', 'cdr3_aa'):
        unresolved_mask |= df[col].apply(is_unresolved)
    n_unresolved = int(unresolved_mask.sum())
    if n_unresolved:
        df = df[~unresolved_mask].copy()

    # --- Steps 4-7: Harmonize V/J calls ---
    # Orphon conversion happens first so both Adaptive TRBV20-or9_2 and already
    # canonical TRBV20/OR9-2 survive adaptive_to_airr(). The shared
    # collapse_imgt_singleton() helper is called explicitly to keep this logic
    # identical across preprocessing entry points.
    df['v_call'] = (
        df['v_call']
        .apply(harmonize_orphon)
        .apply(harmonize_gene)
        .apply(collapse_indistinguishable_v)
        .apply(collapse_imgt_singleton)
    )
    df['j_call'] = df['j_call'].apply(harmonize_gene)

    # --- Step 8: Trim CDR3 sequences (strip conserved C and F/W) ---
    df['cdr3_aa'] = df['cdr3_aa'].apply(trim_cdr3)

    # --- Step 9: Drop rows with missing/empty required fields ---
    missing_mask = pd.Series(False, index=df.index)
    for col in ('cdr3_aa', 'v_call', 'j_call'):
        missing_mask |= df[col].isna() | (df[col].astype(str).str.strip() == '')
    n_missing = int(missing_mask.sum())
    if n_missing:
        df = df[~missing_mask].copy()

    # --- Step 10: Drop rows whose CDR3 contains non-standard AA characters ---
    # Explicit bool conversion is required for empty chunks. With pandas 3,
    # Series.apply() can preserve the empty source column's string dtype, whose
    # sum is "", causing int((~mask).sum()) to fail.
    mask = df['cdr3_aa'].map(cdr3_is_valid).astype(bool)
    n_invalid_cdr3 = int((~mask).sum())
    df = df[mask].copy()

    # --- Step 11: Drop rows lacking gene-level V/J resolution ---
    # Family-only calls (e.g. TRBV20, TRBJ2) and unknown calls cannot be
    # matched to a specific gene. Recognized singleton V families are valid.
    n_family_only = 0
    mask = df['v_call'].map(is_gene_level_v).astype(bool)
    n_family_only += int((~mask).sum())
    df = df[mask].copy()
    mask = df['j_call'].map(is_gene_level_j).astype(bool)
    n_family_only += int((~mask).sum())
    df = df[mask].copy()

    # --- Step 12: Add standardized sequence/read columns ---
    if 'nucleotide' in df.columns:
        df['sequence'] = df['nucleotide']
    else:
        print(f"  WARNING: 'nucleotide' column not found in {input_basename}, "
              f"cannot create 'sequence' column")

    count_col = 'count (templates/reads)'
    if count_col in df.columns:
        df['num_reads'] = df[count_col]
    else:
        print(f"  WARNING: '{count_col}' column not found in {input_basename}, "
              f"cannot create 'num_reads' column")

    # --- Step 13: Add repertoire_id and participant_label ---
    if repertoire_id is not None:
        df['repertoire_id'] = repertoire_id
    if participant_label is not None:
        df['participant_label'] = participant_label

    if compact_output:
        missing_compact = [c for c in COMPACT_OUTPUT_COLUMNS if c not in df.columns]
        if missing_compact:
            raise ValueError(
                f"{input_basename} cannot produce compact output; missing "
                f"standardized columns: {missing_compact}"
            )
        df = df[COMPACT_OUTPUT_COLUMNS]

    return df, {
        'n_in': n_in,
        'n_out': len(df),
        'n_non_productive': n_non_productive,
        'n_unresolved': n_unresolved,
        'n_missing_field': n_missing,
        'n_invalid_cdr3': n_invalid_cdr3,
        'n_family_only': n_family_only,
    }


def preprocess_file(input_path, output_path, repertoire_id=None,
                    participant_label=None, read_chunksize=None,
                    model_columns_only=False, compact_output=False):
    """Preprocess a single repertoire file.

    Reads a tab-separated Adaptive-format file, applies filtering, column
    renaming, V/J gene remapping, CDR3 trimming, quality filtering, and
    adds extra columns. Writes the result as TSV.

    Args:
        input_path: Path to input TSV file.
        output_path: Path to write output TSV file.
        repertoire_id: If provided, added as 'repertoire_id' column.
            Should match the specimen_label from metadata.
        participant_label: If provided, added as 'participant_label' column.
        read_chunksize: Optional number of input rows per chunk. This bounds
            memory use for very large immuneACCESS files.
        model_columns_only: Read only the raw columns needed by repertoire
            models instead of retaining all source/export metadata columns.
        compact_output: Write only the standardized model columns.

    Returns:
        Stats dict with per-step row counts.
    """
    # pandas 3's C parser can raise IndexError in _concatenate_chunks while
    # performing its own low-memory type-inference passes with callable
    # usecols (observed in Emerson Keck0080_MC1.tsv). Our explicit outer
    # chunksize already bounds memory, so disable the parser's nested
    # low-memory pass.
    read_kwargs = {'sep': '\t', 'low_memory': False}
    if model_columns_only:
        read_kwargs['usecols'] = lambda column: column in MODEL_INPUT_COLUMNS
    if read_chunksize is not None:
        read_kwargs['chunksize'] = read_chunksize

    data = pd.read_csv(input_path, **read_kwargs)
    chunks = data if read_chunksize is not None else [data]
    input_basename = os.path.basename(input_path)
    temp_output = f"{output_path}.tmp.{os.getpid()}"
    totals = {
        'file': os.path.basename(input_path),
        'output_file': os.path.basename(output_path),
        'n_in': 0,
        'n_out': 0,
        'n_non_productive': 0,
        'n_unresolved': 0,
        'n_missing_field': 0,
        'n_invalid_cdr3': 0,
        'n_family_only': 0,
    }
    first_chunk = True

    try:
        for chunk in chunks:
            processed, stats = preprocess_dataframe(
                chunk,
                input_basename=input_basename,
                repertoire_id=repertoire_id,
                participant_label=participant_label,
                compact_output=compact_output,
            )
            processed.to_csv(
                temp_output,
                sep='\t',
                index=False,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
            )
            first_chunk = False
            for key in (
                'n_in', 'n_out', 'n_non_productive', 'n_unresolved',
                'n_missing_field', 'n_invalid_cdr3', 'n_family_only',
            ):
                totals[key] += stats[key]

        if first_chunk:
            raise ValueError(f"{input_basename} contained no readable rows")
        os.replace(temp_output, output_path)
    except Exception:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise

    return totals


# ---------------------------------------------------------------------------
# Gene-vocabulary comparison
# ---------------------------------------------------------------------------

def collect_gene_vocabulary(paths, chunksize=250_000):
    """Collect canonical allele-free V/J names from processed TSV/TSV.GZ files."""
    vocabulary = {'V': set(), 'J': set()}
    for path in paths:
        try:
            chunks = pd.read_csv(
                path,
                sep='\t',
                usecols=['v_call', 'j_call'],
                chunksize=chunksize,
            )
            for chunk in chunks:
                v_calls = (
                    chunk['v_call'].dropna().astype(str)
                    .map(harmonize_orphon)
                    .map(harmonize_gene)
                    .map(collapse_indistinguishable_v)
                    .map(collapse_imgt_singleton)
                )
                j_calls = (
                    chunk['j_call'].dropna().astype(str)
                    .map(harmonize_gene)
                )
                vocabulary['V'].update(
                    gene for gene in v_calls
                    if gene and not is_unresolved(gene)
                )
                vocabulary['J'].update(
                    gene for gene in j_calls
                    if gene and not is_unresolved(gene)
                )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Could not collect v_call/j_call vocabulary from {path}: {exc}"
            ) from exc
    return vocabulary


def compare_gene_vocabularies(external_paths, reference_paths, output_path,
                              chunksize=250_000):
    """Write shared/external-only/reference-only V/J gene names to a TSV."""
    external = collect_gene_vocabulary(external_paths, chunksize=chunksize)
    reference = collect_gene_vocabulary(reference_paths, chunksize=chunksize)
    rows = []
    summaries = {}

    for locus in ('V', 'J'):
        shared = external[locus] & reference[locus]
        external_only = external[locus] - reference[locus]
        reference_only = reference[locus] - external[locus]
        summaries[locus] = {
            'shared': sorted(shared),
            'external_only': sorted(external_only),
            'reference_only': sorted(reference_only),
        }
        for gene in sorted(external[locus] | reference[locus]):
            if gene in shared:
                status = 'shared'
            elif gene in external_only:
                status = 'external_only'
            else:
                status = 'reference_only'
            rows.append({
                'locus': locus,
                'gene': gene,
                'status': status,
                'in_external': gene in external[locus],
                'in_reference': gene in reference[locus],
            })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep='\t', index=False)

    print("\nGene vocabulary comparison:")
    for locus in ('V', 'J'):
        summary = summaries[locus]
        print(
            f"  {locus}: {len(summary['shared'])} shared, "
            f"{len(summary['external_only'])} external-only, "
            f"{len(summary['reference_only'])} reference-only"
        )
        print(f"    external-only: {summary['external_only']}")
        print(f"    reference-only: {summary['reference_only']}")
    print(f"  Full shared/unique report: {output_path}")

    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess external repertoire files to AIRR/IMGT conventions."
    )
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing raw external repertoire files. '
                             'Optional with --emerson_cmv.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to write processed files '
                             '(default: data/external_processed_v2/)')
    parser.add_argument('--file_glob', type=str, default=None,
                        help='Glob pattern for repertoire files (default: *_TCRB.tsv)')
    parser.add_argument('--emerson_cmv', action='store_true', default=False,
                        help='Use the Emerson CMV preset: default raw/processed CMV '
                             'directories, *.tsv input, filename-derived participant '
                             'and repertoire IDs, 100,000-row chunked reads, and '
                             'compact model columns.')
    parser.add_argument('--metadata_file', type=str, default=None,
                        help='Path to metadata TSV file with specimen_label and '
                             'participant_label columns. If provided, adds '
                             'repertoire_id and participant_label columns to output, '
                             'and by default only processes files with matching '
                             'metadata entries (see --process_all)')
    parser.add_argument('--process_all', action='store_true', default=False,
                        help='Process all input files even if they have no matching '
                             'metadata entry. Files without metadata get empty '
                             'repertoire_id and participant_label. Has no effect '
                             'if --metadata_file is not provided (all files are '
                             'always processed in that case).')
    parser.add_argument('--strip_filename_TCRB_suffix', action='store_true', default=False,
                        help='Remove "_TCRB" suffix from output filenames '
                             '(e.g. 310101_TCRB.tsv → 310101.tsv). Use for T1D '
                             'to match specimen_label in metadata.')
    parser.add_argument('--participant_from_filename', action='store_true', default=False,
                        help='Set both repertoire_id and participant_label from the '
                             'output filename stem. Used automatically by '
                             '--emerson_cmv, where each file is one subject.')
    parser.add_argument('--read_chunksize', type=int, default=None,
                        help='Read each input file in chunks of this many rows. '
                             'Recommended for large immuneACCESS exports; '
                             '--emerson_cmv defaults to 100000.')
    parser.add_argument('--model_columns_only', action='store_true', default=False,
                        help='Read only raw sequence, V/J, productivity, and count '
                             'columns. Used automatically by --emerson_cmv.')
    parser.add_argument('--compact_output', action='store_true', default=False,
                        help='Write only cdr3_aa, v_call, j_call, sequence, '
                             'num_reads, repertoire_id, and participant_label. '
                             'Used automatically by --emerson_cmv.')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='Resume a prior run by skipping non-empty output '
                             'files. Outputs are written atomically, so non-empty '
                             'files from this script are complete.')
    parser.add_argument('--compare_gene_dir', type=str, default=None,
                        help='Optional processed reference repertoire directory '
                             '(for example data/malid_clean/TCR). After processing, '
                             'write shared/external-only/reference-only V/J names.')
    parser.add_argument('--compare_gene_glob', type=str, default='*.tsv*',
                        help='Glob within --compare_gene_dir (default: *.tsv*).')
    parser.add_argument('--gene_comparison_output', type=str, default=None,
                        help='Gene comparison TSV path (default: '
                             '<output_dir>/gene_vocabulary_comparison.tsv).')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel processes (default: 4). '
                             'Set to 1 for serial execution.')

    args = parser.parse_args()

    if args.emerson_cmv:
        args.input_dir = args.input_dir or EMERSON_CMV_INPUT_DIR
        args.output_dir = args.output_dir or EMERSON_CMV_OUTPUT_DIR
        args.file_glob = args.file_glob or '*.tsv'
        args.read_chunksize = args.read_chunksize or 100_000
        args.participant_from_filename = True
        args.model_columns_only = True
        args.compact_output = True
    else:
        if args.input_dir is None:
            parser.error("--input_dir is required unless --emerson_cmv is used")
        args.output_dir = args.output_dir or 'data/external_processed_v2/'
        args.file_glob = args.file_glob or '*_TCRB.tsv'

    if args.n_jobs < 1:
        raise ValueError(f"--n_jobs must be >= 1, got {args.n_jobs}")
    if args.read_chunksize is not None and args.read_chunksize < 1:
        raise ValueError(
            f"--read_chunksize must be >= 1, got {args.read_chunksize}"
        )
    if args.metadata_file is not None and args.participant_from_filename:
        parser.error(
            "--metadata_file and --participant_from_filename are mutually exclusive"
        )

    # --- Discover input files ---
    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.file_glob)))
    if not input_files:
        print(f"No files matching '{args.file_glob}' found in {args.input_dir}")
        return

    # --- Load metadata if provided ---
    metadata_specimen_labels = None
    specimen_to_participant = None
    if args.metadata_file is not None:
        if not os.path.exists(args.metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found: {args.metadata_file}"
            )
        metadata_specimen_labels, specimen_to_participant = load_metadata(
            args.metadata_file
        )
        print(f"Loaded metadata: {len(metadata_specimen_labels)} specimen entries "
              f"from {args.metadata_file}")

    # --- Build list of files to process ---
    # For each input file, compute the output filename and specimen_label
    files_to_process = []
    files_skipped = []
    files_skipped_existing = []
    expected_outputs = []
    expected_output_sources = {}
    for input_path in input_files:
        input_filename = os.path.basename(input_path)

        # Compute output filename (optionally strip _TCRB suffix)
        output_filename = input_filename
        if args.strip_filename_TCRB_suffix:
            stem, ext = os.path.splitext(output_filename)
            if stem.endswith('_TCRB'):
                output_filename = stem[:-5] + ext

        # Derive specimen_label from output filename
        specimen_label = derive_specimen_label(output_filename)

        # Determine if this file should be processed
        if metadata_specimen_labels is not None and not args.process_all:
            # Only process files with matching metadata entries
            if specimen_label not in metadata_specimen_labels:
                files_skipped.append((input_filename, specimen_label))
                continue

        # Look up participant_label from metadata (None if no metadata or no match)
        participant_label = None
        if specimen_to_participant is not None:
            participant_label = specimen_to_participant.get(specimen_label)

        # If metadata is provided, set repertoire_id = specimen_label for matched files,
        # or empty string for unmatched files (when --process_all)
        repertoire_id = None
        if metadata_specimen_labels is not None:
            if specimen_label in metadata_specimen_labels:
                repertoire_id = specimen_label
            else:
                # --process_all is True but file has no metadata match
                repertoire_id = ''
                participant_label = ''
        elif args.participant_from_filename:
            repertoire_id = specimen_label
            participant_label = specimen_label

        output_path = os.path.join(args.output_dir, output_filename)
        if output_path in expected_output_sources:
            raise ValueError(
                f"Output filename collision: '{output_filename}' would be "
                f"produced by both "
                f"'{os.path.basename(expected_output_sources[output_path])}' and "
                f"'{input_filename}'. Check --strip_filename_TCRB_suffix and "
                f"input files."
            )
        expected_output_sources[output_path] = input_path
        expected_outputs.append(output_path)

        if (args.skip_existing and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0):
            files_skipped_existing.append(output_path)
            continue

        files_to_process.append({
            'input_path': input_path,
            'output_path': output_path,
            'repertoire_id': repertoire_id,
            'participant_label': participant_label,
        })

    if not files_to_process and not files_skipped_existing:
        print("No files to process after metadata filtering.")
        if files_skipped:
            print(f"  {len(files_skipped)} files skipped (no metadata match)")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(files_to_process)} files from {args.input_dir}")
    if files_skipped:
        print(f"  Skipping {len(files_skipped)} files (no metadata match)")
    if files_skipped_existing:
        print(
            f"  Resuming: skipping {len(files_skipped_existing)} existing "
            f"non-empty outputs"
        )
    print(f"Output directory: {args.output_dir}")
    if args.strip_filename_TCRB_suffix:
        print("  Stripping '_TCRB' suffix from output filenames")
    if metadata_specimen_labels is not None:
        print(f"  Adding repertoire_id and participant_label from metadata")
    if args.n_jobs > 1:
        print(f"  Parallel processing with {args.n_jobs} workers")

    # --- Process files ---
    n_total = len(files_to_process)
    all_stats = []
    if n_total == 0:
        print("  No new files need processing.")
    elif args.n_jobs == 1:
        # Serial execution
        for i, file_info in enumerate(files_to_process, 1):
            stats = preprocess_file(
                file_info['input_path'],
                file_info['output_path'],
                repertoire_id=file_info['repertoire_id'],
                participant_label=file_info['participant_label'],
                read_chunksize=args.read_chunksize,
                model_columns_only=args.model_columns_only,
                compact_output=args.compact_output,
            )
            all_stats.append(stats)
            if i % 10 == 0 or i == n_total:
                print(f"  Processed {i}/{n_total} files "
                      f"({stats['n_in']:,} -> {stats['n_out']:,} rows in last file)")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(
                    preprocess_file,
                    file_info['input_path'],
                    file_info['output_path'],
                    repertoire_id=file_info['repertoire_id'],
                    participant_label=file_info['participant_label'],
                    read_chunksize=args.read_chunksize,
                    model_columns_only=args.model_columns_only,
                    compact_output=args.compact_output,
                ): file_info
                for file_info in files_to_process
            }
            for future in as_completed(futures):
                stats = future.result()
                all_stats.append(stats)
                n_done = len(all_stats)
                if n_done % 10 == 0 or n_done == n_total:
                    print(f"  Processed {n_done}/{n_total} files")

    # --- Summary ---
    total_in = sum(s['n_in'] for s in all_stats)
    total_out = sum(s['n_out'] for s in all_stats)
    total_non_productive = sum(s['n_non_productive'] for s in all_stats)
    total_unresolved = sum(s['n_unresolved'] for s in all_stats)
    total_missing = sum(s['n_missing_field'] for s in all_stats)
    total_invalid = sum(s['n_invalid_cdr3'] for s in all_stats)
    total_family_only = sum(s['n_family_only'] for s in all_stats)
    total_dropped = total_in - total_out

    print(f"\nDone. {len(all_stats)} files processed.")
    print(f"  Total rows: {total_in:,} -> {total_out:,} (dropped {total_dropped:,})")
    print(f"  Dropped breakdown:")
    print(f"    sequenceStatus != 'In': {total_non_productive:,}")
    print(f"    unresolved v/j/cdr3:    {total_unresolved:,}")
    print(f"    missing v/j/cdr3:       {total_missing:,}")
    print(f"    invalid CDR3 chars:     {total_invalid:,}")
    print(f"    family-only v/j (+unk): {total_family_only:,}")

    # --- Validate output files ---
    missing_files = [p for p in expected_outputs if not os.path.exists(p)]
    empty_files = [p for p in expected_outputs
                   if os.path.exists(p) and os.path.getsize(p) == 0]

    if missing_files or empty_files:
        print(f"\n  VALIDATION FAILED:")
        if missing_files:
            print(f"    {len(missing_files)} output files missing:")
            for p in missing_files[:10]:
                print(f"      {os.path.basename(p)}")
            if len(missing_files) > 10:
                print(f"      ... and {len(missing_files) - 10} more")
        if empty_files:
            print(f"    {len(empty_files)} output files are empty (0 bytes):")
            for p in empty_files[:10]:
                print(f"      {os.path.basename(p)}")
            if len(empty_files) > 10:
                print(f"      ... and {len(empty_files) - 10} more")
    else:
        print(f"\n  Validation: all {len(expected_outputs)} output files exist and are non-empty.")

    if args.compare_gene_dir is not None:
        if missing_files or empty_files:
            raise RuntimeError(
                "Skipping gene vocabulary comparison because output validation failed"
            )
        reference_files = sorted(glob.glob(os.path.join(
            args.compare_gene_dir, args.compare_gene_glob
        )))
        if not reference_files:
            raise FileNotFoundError(
                f"No reference files matching '{args.compare_gene_glob}' found "
                f"in {args.compare_gene_dir}"
            )
        comparison_output = (
            args.gene_comparison_output
            or os.path.join(args.output_dir, 'gene_vocabulary_comparison.tsv')
        )
        compare_gene_vocabularies(
            expected_outputs,
            reference_files,
            comparison_output,
            chunksize=args.read_chunksize or 250_000,
        )


if __name__ == '__main__':
    main()
