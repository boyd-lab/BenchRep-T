"""
Gene name harmonization between Adaptive/immunoSEQ and AIRR/IMGT formats.

Adaptive format:  TCRBV07-02, TCRBJ02-03  (no allele in gene name column)
AIRR/IMGT format: TRBV7-2*01, TRBJ2-3*01 (allele included)

Key differences:
  - Prefix: TCRBV/TCRBJ vs TRBV/TRBJ (extra 'C')
  - Number padding: leading zeros in Adaptive (07, 02) vs none in AIRR (7, 2)
  - Allele: separate column in Adaptive, appended with '*' in AIRR
"""

import re


def adaptive_to_airr(gene_name):
    """Convert an Adaptive/immunoSEQ gene name to AIRR/IMGT format (without allele).

    Examples:
        TCRBV07-02   → TRBV7-2
        TCRBJ02-03   → TRBJ2-3
        TCRBV14      → TRBV14
        TCRBV03-01/03-02 → TRBV3-1  (takes first gene for ambiguous calls)

    Args:
        gene_name: Adaptive-format gene name string.

    Returns:
        AIRR-format gene name without allele, or the original value if
        it cannot be parsed (e.g. NaN, empty string, unrecognised format).
    """
    if not isinstance(gene_name, str) or gene_name.strip() == '':
        return gene_name

    name = gene_name.strip()

    # Take the first gene for ambiguous calls (e.g. TCRBV03-01/03-02)
    if '/' in name:
        name = name.split('/')[0]

    # Remove the extra 'C': TCRBV → TRBV, TCRBJ → TRBJ
    name = re.sub(r'^TCRB([VDJ])', r'TRB\1', name)

    # Strip leading zeros from each number group while preserving hyphens
    # e.g. TRBV07-02 → TRBV7-2,  TRBJ02-03 → TRBJ2-3
    name = re.sub(r'(?<=\D)0+(\d)', r'\1', name)

    return name


def strip_allele(gene_name):
    """Remove allele designation from an AIRR gene name.

    Examples:
        TRBV7-2*01 → TRBV7-2
        TRBJ2-3*02 → TRBJ2-3
        TRBV14*01  → TRBV14
        TRBV7-2    → TRBV7-2  (no-op if no allele)

    Args:
        gene_name: AIRR-format gene name, possibly with ``*allele`` suffix.

    Returns:
        Gene name with allele stripped, or the original value if not a string.
    """
    if not isinstance(gene_name, str):
        return gene_name
    return gene_name.split('*')[0]


# IMGT TRBV families with a single functional member. The Adaptive convention
# names these as "TRBV<N>-1"; IMGT canonical drops the "-1". Required when
# merging cohorts processed under each convention (e.g. internal MAL-ID vs
# external Adaptive-derived T1D files).
_IMGT_SINGLETON_TRBV = {
    'TRBV2', 'TRBV9', 'TRBV13', 'TRBV14', 'TRBV15',
    'TRBV18', 'TRBV19', 'TRBV27', 'TRBV28', 'TRBV30',
}


def collapse_imgt_singleton(gene_name):
    """Collapse Adaptive-style "-1" suffix on IMGT singleton TRBV families.

    Examples:
        TRBV13-1 → TRBV13
        TRBV2-1  → TRBV2
        TRBV7-9  → TRBV7-9   (multi-member family, untouched)
        TRBV13   → TRBV13    (no-op)
    """
    if not isinstance(gene_name, str) or not gene_name.endswith('-1'):
        return gene_name
    base = gene_name[:-2]
    return base if base in _IMGT_SINGLETON_TRBV else gene_name


def canonicalize_gene(gene_name):
    """Strip allele and collapse Adaptive singletons to IMGT canonical form.

    Use this to align V/J gene labels across cohorts that mix IMGT (no "-1"
    on singleton families) and Adaptive ("-1" on every gene) conventions.
    """
    return collapse_imgt_singleton(strip_allele(gene_name))
