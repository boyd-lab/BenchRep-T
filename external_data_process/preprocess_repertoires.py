"""
Preprocess external repertoire files so they match internal (AIRR) conventions.

Transformations applied:
  1. Rename columns: aminoAcid → cdr3_aa, vGeneName → v_call, jGeneName → j_call
  2. Remap V/J gene names from Adaptive to AIRR format (e.g. TCRBV07-02 → TRBV7-2)
  3. Strip alleles from V/J gene names (e.g. TRBV7-2*01 → TRBV7-2)
  4. Trim CDR3 sequences: remove first and last amino acid

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


def trim_cdr3(seq):
    """Remove the first and last amino acid from a CDR3 sequence."""
    if not isinstance(seq, str) or len(seq) <= 2:
        return seq
    return seq[1:-1]


def harmonize_gene(gene_name):
    """Convert Adaptive gene name to AIRR format, then strip allele."""
    return strip_allele(adaptive_to_airr(gene_name))


def preprocess_file(input_path, output_path):
    """Preprocess a single repertoire file.

    Reads a tab-separated Adaptive-format file, applies column renaming,
    V/J gene remapping, and CDR3 trimming, then writes the result.
    """
    df = pd.read_csv(input_path, sep='\t')

    # Rename columns that exist in the file
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Remap V/J gene names
    if 'v_call' in df.columns:
        df['v_call'] = df['v_call'].apply(harmonize_gene)
    if 'j_call' in df.columns:
        df['j_call'] = df['j_call'].apply(harmonize_gene)

    # Trim CDR3 sequences
    if 'cdr3_aa' in df.columns:
        df['cdr3_aa'] = df['cdr3_aa'].apply(trim_cdr3)

    df.to_csv(output_path, sep='\t', index=False)


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

    for input_path in tqdm(input_files, desc="Preprocessing"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)
        preprocess_file(input_path, output_path)

    print(f"Done. {len(input_files)} files written to {args.output_dir}")


if __name__ == '__main__':
    main()
