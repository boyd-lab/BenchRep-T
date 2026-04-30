"""
Preprocess external ImmunoSEQ repertoire files so they match internal (AIRR/IMGT) conventions
used by Mal-ID and Mal-ID-Lite.

Transformations applied:
  1.  Filter to productive sequences: keep only rows with sequenceStatus == "In"
  2.  Rename columns: aminoAcid → cdr3_aa, vGeneName → v_call, jGeneName → j_call
  3.  Remap V/J gene names from Adaptive to AIRR format (e.g. TCRBV07-02 → TRBV7-2)
  4.  Strip alleles from V/J gene names (e.g. TRBV7-2*01 → TRBV7-2)
  5.  Collapse indistinguishable V genes (TRBV12-4 → TRBV12-3, TRBV6-3 → TRBV6-2),
      mirroring preprocessing/clean_tcr_data.py
  6.  Collapse singleton TRBV families: strip "-1" suffix on families with only one
      functional member (e.g. TRBV13-1 → TRBV13), matching IMGT/Mal-ID convention.
      Derived from IMGT reference (tcrb_v_gene_cdrs.generated.tsv).
  7.  Harmonize orphon gene names: convert Adaptive "-or9_2" suffix to IMGT "/OR9-2"
      format (e.g. TRBV20-or9_2 → TRBV20/OR9-2)
  8.  Trim CDR3 sequences: remove first and last amino acid (conserved C and F/W)
  9.  Drop rows with "unresolved" v_call, j_call, or cdr3_aa
  10. Drop rows with missing/empty v_call, j_call, or cdr3_aa
  11. Drop rows whose CDR3 contains non-standard AA characters (*, X, gaps, etc.)
  12. Add "sequence" column (copy of "nucleotide")
  13. Add "num_reads" column (copy of "count (templates/reads)")
  14. If --metadata_file is provided:
      a. Add "repertoire_id" column derived from output filename (matches specimen_label
         in metadata)
      b. Add "participant_label" column looked up from metadata
  15. If --strip_filename_TCRB_suffix is set, remove "_TCRB" from output filenames

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
"""

import os
import re
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.gene_harmonization import adaptive_to_airr, strip_allele


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Adaptive → AIRR column name mapping
COLUMN_RENAME = {
    'aminoAcid': 'cdr3_aa',
    'vGeneName': 'v_call',
    'jGeneName': 'j_call',
}

# Indistinguishable V genes due to FR3 primers — mirrors GENE_FIXES in
# preprocessing/clean_tcr_data.py. Keys are gene names *after* allele stripping.
GENE_COLLAPSES = {
    'TRBV12-4': 'TRBV12-3',
    'TRBV6-3': 'TRBV6-2',
}

# IMGT TRBV singleton families: families with only one functional gene member.
# Adaptive/immunoSEQ names these as "TRBV<N>-1"; IMGT/Mal-ID drops the "-1".
# Derived from IMGT reference file tcrb_v_gene_cdrs.generated.tsv: these are
# all v_gene entries that have no hyphen-number suffix (excluding orphon genes).
SINGLETON_TRBV = {
    'TRBV1', 'TRBV2', 'TRBV9', 'TRBV13', 'TRBV14', 'TRBV15',
    'TRBV16', 'TRBV17', 'TRBV18', 'TRBV19', 'TRBV26', 'TRBV27',
    'TRBV28', 'TRBV30',
}

VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


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


def collapse_singleton_v(gene_name):
    """Strip "-1" suffix on IMGT singleton TRBV families.

    Examples:
        TRBV13-1 → TRBV13   (singleton family)
        TRBV2-1  → TRBV2    (singleton family)
        TRBV7-1  → TRBV7-1  (multi-member family, untouched)
    """
    if not isinstance(gene_name, str) or not gene_name.endswith('-1'):
        return gene_name
    base = gene_name[:-2]
    return base if base in SINGLETON_TRBV else gene_name


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
    return re.sub(r'-or9_2$', '/OR9-2', gene_name)


def cdr3_is_valid(seq):
    """True if seq is a non-empty string of standard amino acids."""
    if not isinstance(seq, str) or seq == '':
        return False
    return set(seq.upper()).issubset(VALID_AMINO_ACIDS)


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

def preprocess_file(input_path, output_path, repertoire_id=None,
                    participant_label=None):
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

    Returns:
        Stats dict with per-step row counts.
    """
    df = pd.read_csv(input_path, sep='\t')
    n_in = len(df)

    # --- Step 1: Filter to productive sequences (sequenceStatus == "In") ---
    n_non_productive = 0
    if 'sequenceStatus' in df.columns:
        mask = df['sequenceStatus'] == 'In'
        n_non_productive = int((~mask).sum())
        df = df[mask].copy()
    else:
        print(f"  WARNING: 'sequenceStatus' column not found in {os.path.basename(input_path)}, "
              f"skipping productive filter")

    # --- Step 2: Rename columns ---
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # --- Steps 3-4: Remap V/J gene names (Adaptive → AIRR, strip alleles) ---
    # --- Step 5: Collapse indistinguishable V genes ---
    # --- Step 6: Collapse singleton V gene families ---
    # --- Step 7: Harmonize orphon gene names ---
    if 'v_call' in df.columns:
        df['v_call'] = (
            df['v_call']
            .apply(harmonize_gene)
            .apply(collapse_indistinguishable_v)
            .apply(collapse_singleton_v)
            .apply(harmonize_orphon)
        )
    if 'j_call' in df.columns:
        df['j_call'] = df['j_call'].apply(harmonize_gene)

    # --- Step 8: Trim CDR3 sequences (strip conserved C and F/W) ---
    if 'cdr3_aa' in df.columns:
        df['cdr3_aa'] = df['cdr3_aa'].apply(trim_cdr3)

    # --- Step 9: Drop rows with "unresolved" v_call, j_call, or cdr3_aa ---
    n_unresolved = 0
    for col in ('v_call', 'j_call', 'cdr3_aa'):
        if col in df.columns:
            mask = df[col] != 'unresolved'
            n_unresolved += int((~mask).sum())
            df = df[mask]

    # --- Step 10: Drop rows with missing/empty required fields ---
    n_missing = 0
    for col in ('cdr3_aa', 'v_call', 'j_call'):
        if col in df.columns:
            mask = df[col].notna() & (df[col].astype(str).str.len() > 0)
            n_missing += int((~mask).sum())
            df = df[mask]

    # --- Step 11: Drop rows whose CDR3 contains non-standard AA characters ---
    n_invalid_cdr3 = 0
    if 'cdr3_aa' in df.columns:
        mask = df['cdr3_aa'].apply(cdr3_is_valid)
        n_invalid_cdr3 = int((~mask).sum())
        df = df[mask]

    # --- Step 12: Add "sequence" column (copy of "nucleotide") ---
    if 'nucleotide' in df.columns:
        df['sequence'] = df['nucleotide']
    else:
        print(f"  WARNING: 'nucleotide' column not found in {os.path.basename(input_path)}, "
              f"cannot create 'sequence' column")

    # --- Step 13: Add "num_reads" column (copy of "count (templates/reads)") ---
    count_col = 'count (templates/reads)'
    if count_col in df.columns:
        df['num_reads'] = df[count_col]
    else:
        print(f"  WARNING: '{count_col}' column not found in {os.path.basename(input_path)}, "
              f"cannot create 'num_reads' column")

    # --- Step 14: Add repertoire_id and participant_label ---
    if repertoire_id is not None:
        df['repertoire_id'] = repertoire_id
    if participant_label is not None:
        df['participant_label'] = participant_label

    df.to_csv(output_path, sep='\t', index=False)

    return {
        'file': os.path.basename(input_path),
        'output_file': os.path.basename(output_path),
        'n_in': n_in,
        'n_out': len(df),
        'n_non_productive': n_non_productive,
        'n_unresolved': n_unresolved,
        'n_missing_field': n_missing,
        'n_invalid_cdr3': n_invalid_cdr3,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess external repertoire files to AIRR/IMGT conventions."
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw external repertoire files')
    parser.add_argument('--output_dir', type=str, default='data/external_processed_v2/',
                        help='Directory to write processed files '
                             '(default: data/external_processed_v2/)')
    parser.add_argument('--file_glob', type=str, default='*_TCRB.tsv',
                        help='Glob pattern for repertoire files (default: *_TCRB.tsv)')
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
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel processes (default: 4). '
                             'Set to 1 for serial execution.')

    args = parser.parse_args()

    if args.n_jobs < 1:
        raise ValueError(f"--n_jobs must be >= 1, got {args.n_jobs}")

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

        output_path = os.path.join(args.output_dir, output_filename)
        files_to_process.append({
            'input_path': input_path,
            'output_path': output_path,
            'repertoire_id': repertoire_id,
            'participant_label': participant_label,
        })

    if not files_to_process:
        print("No files to process after metadata filtering.")
        if files_skipped:
            print(f"  {len(files_skipped)} files skipped (no metadata match)")
        return

    # Check for duplicate output filenames (e.g. both FOO_TCRB.tsv and FOO.tsv
    # mapping to FOO.tsv when --strip_filename_TCRB_suffix is set)
    output_filenames = [os.path.basename(f['output_path']) for f in files_to_process]
    seen = {}
    for i, name in enumerate(output_filenames):
        if name in seen:
            raise ValueError(
                f"Output filename collision: '{name}' would be produced by both "
                f"'{os.path.basename(files_to_process[seen[name]]['input_path'])}' and "
                f"'{os.path.basename(files_to_process[i]['input_path'])}'. "
                f"Check --strip_filename_TCRB_suffix and input files."
            )
        seen[name] = i

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(files_to_process)} files from {args.input_dir}")
    if files_skipped:
        print(f"  Skipping {len(files_skipped)} files (no metadata match)")
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
    if args.n_jobs == 1:
        # Serial execution
        for i, file_info in enumerate(files_to_process, 1):
            stats = preprocess_file(
                file_info['input_path'],
                file_info['output_path'],
                repertoire_id=file_info['repertoire_id'],
                participant_label=file_info['participant_label'],
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
    total_dropped = total_in - total_out

    print(f"\nDone. {len(all_stats)} files processed.")
    print(f"  Total rows: {total_in:,} -> {total_out:,} (dropped {total_dropped:,})")
    print(f"  Dropped breakdown:")
    print(f"    sequenceStatus != 'In': {total_non_productive:,}")
    print(f"    unresolved v/j/cdr3:    {total_unresolved:,}")
    print(f"    missing v/j/cdr3:       {total_missing:,}")
    print(f"    invalid CDR3 chars:     {total_invalid:,}")

    # --- Validate output files ---
    expected_outputs = [f['output_path'] for f in files_to_process]
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


if __name__ == '__main__':
    main()
