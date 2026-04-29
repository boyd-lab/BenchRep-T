"""
Preprocess external repertoire files so they match internal (AIRR) conventions.

Transformations applied:
  1. Rename columns: aminoAcid → cdr3_aa, vGeneName → v_call, jGeneName → j_call
  2. Remap V/J gene names from Adaptive to AIRR format (e.g. TCRBV07-02 → TRBV7-2)
  3. Strip alleles from V/J gene names (e.g. TRBV7-2*01 → TRBV7-2)
  4. Collapse indistinguishable V genes (TRBV12-4 → TRBV12-3, TRBV6-3 → TRBV6-2),
     mirroring preprocessing/clean_tcr_data.py
  5. Trim CDR3 sequences: remove first and last amino acid
  6. Drop rows with missing/empty v_call, j_call, or cdr3_aa
  7. Drop rows whose CDR3 contains non-standard AA characters (*, X, gaps, etc.)

Usage:
    python external_data_process/preprocess_repertoires.py \
        --input_dir data/external_raw/ \
        --output_dir data/external_processed/ \
        --file_glob "*_TCRB.tsv"
"""

import os
import argparse
import glob

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.gene_harmonization import adaptive_to_airr, strip_allele


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

VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


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


def cdr3_is_valid(seq):
    """True if seq is a non-empty string of standard amino acids."""
    if not isinstance(seq, str) or seq == '':
        return False
    return set(seq.upper()).issubset(VALID_AMINO_ACIDS)


def preprocess_file(input_path, output_path):
    """Preprocess a single repertoire file.

    Reads a tab-separated Adaptive-format file, applies column renaming,
    V/J gene remapping, CDR3 trimming, and quality filtering, then writes
    the result. Returns a stats dict with row counts.
    """
    df = pd.read_csv(input_path, sep='\t')
    n_in = len(df)

    # Rename columns that exist in the file
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Remap V/J gene names; collapse indistinguishable V genes after harmonization
    if 'v_call' in df.columns:
        df['v_call'] = df['v_call'].apply(harmonize_gene).apply(collapse_indistinguishable_v)
    if 'j_call' in df.columns:
        df['j_call'] = df['j_call'].apply(harmonize_gene)

    # Trim CDR3 sequences (strip conserved C and F/W)
    if 'cdr3_aa' in df.columns:
        df['cdr3_aa'] = df['cdr3_aa'].apply(trim_cdr3)

    # Drop rows missing required fields
    n_missing = 0
    for col in ('cdr3_aa', 'v_call', 'j_call'):
        if col in df.columns:
            mask = df[col].notna() & (df[col].astype(str).str.len() > 0)
            n_missing += int((~mask).sum())
            df = df[mask]

    # Drop rows whose CDR3 contains non-standard AA characters
    n_invalid_cdr3 = 0
    if 'cdr3_aa' in df.columns:
        mask = df['cdr3_aa'].apply(cdr3_is_valid)
        n_invalid_cdr3 = int((~mask).sum())
        df = df[mask]

    df.to_csv(output_path, sep='\t', index=False)

    return {
        'file': os.path.basename(input_path),
        'n_in': n_in,
        'n_out': len(df),
        'n_missing_field': n_missing,
        'n_invalid_cdr3': n_invalid_cdr3,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess external repertoire files to AIRR conventions."
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw external repertoire files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write processed files')
    parser.add_argument('--file_glob', type=str, default='*_TCRB.tsv',
                        help='Glob pattern for repertoire files (default: *_TCRB.tsv)')

    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.file_glob)))
    if not input_files:
        print(f"No files matching '{args.file_glob}' found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(input_files)} files from {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    all_stats = []
    for input_path in tqdm(input_files, desc="Preprocessing"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)
        all_stats.append(preprocess_file(input_path, output_path))

    total_in = sum(s['n_in'] for s in all_stats)
    total_out = sum(s['n_out'] for s in all_stats)
    total_missing = sum(s['n_missing_field'] for s in all_stats)
    total_invalid = sum(s['n_invalid_cdr3'] for s in all_stats)
    print(f"Done. {len(input_files)} files written to {args.output_dir}")
    print(f"  Total rows: {total_in:,} → {total_out:,} (dropped {total_in - total_out:,})")
    print(f"  Dropped — missing v/j/cdr3: {total_missing:,}, invalid CDR3 chars: {total_invalid:,}")


if __name__ == '__main__':
    main()
