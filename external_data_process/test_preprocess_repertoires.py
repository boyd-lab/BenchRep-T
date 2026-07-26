import unittest

import pandas as pd

from airr_bench.external_data_process.preprocess_repertoires import (
    preprocess_dataframe,
)


class PreprocessRepertoiresTest(unittest.TestCase):
    def test_external_gene_harmonization_and_unresolved_filtering(self):
        rows = [
            ('CASSAAAQYF', 'TCRBV13-01', 'TCRBJ02-03', 'In'),
            ('CASSGGGQYF', 'TRBV20-or9_2', 'TCRBJ02-03', 'In'),
            ('CASSCCCQYF', 'TRBV29/OR9-2', 'TCRBJ02-03', 'In'),
            ('CASSDDDQYF', 'TCRBV07-01', 'TCRBJ02-03', 'In'),
            ('CASSEEEQYF', ' unresolved ', 'TCRBJ02-03', 'In'),
            ('CASSFFFQYF', 'TCRBV13-01', 'Unresolved', 'In'),
            ('unresolved', 'TCRBV13-01', 'TCRBJ02-03', 'In'),
        ]
        df = pd.DataFrame(
            rows,
            columns=['amino_acid', 'v_gene', 'j_gene', 'frame_type'],
        )
        df['rearrangement'] = 'ACGT'
        df['templates'] = 1

        output, stats = preprocess_dataframe(
            df,
            input_basename='synthetic.tsv',
            repertoire_id='sample',
            participant_label='person',
            compact_output=True,
        )

        self.assertEqual(output['v_call'].tolist(), [
            'TRBV13',
            'TRBV20/OR9-2',
            'TRBV29/OR9-2',
            'TRBV7-1',
        ])
        self.assertEqual(output['j_call'].tolist(), ['TRBJ2-3'] * 4)
        self.assertEqual(stats['n_unresolved'], 3)
        self.assertEqual(stats['n_out'], 4)

    def test_chunk_that_becomes_empty_after_filtering(self):
        df = pd.DataFrame({
            'amino_acid': ['CASSAAAQYF', 'CASSGGGQYF'],
            'v_gene': ['unresolved', 'TRBV13-1'],
            'j_gene': ['TRBJ2-3', 'unresolved'],
            'frame_type': ['In', 'In'],
            'rearrangement': ['ACGT', 'ACGT'],
            'templates': [1, 1],
        })

        output, stats = preprocess_dataframe(
            df,
            input_basename='empty_chunk.tsv',
            repertoire_id='sample',
            participant_label='person',
            compact_output=True,
        )

        self.assertTrue(output.empty)
        self.assertEqual(stats['n_unresolved'], 2)
        self.assertEqual(stats['n_out'], 0)


if __name__ == '__main__':
    unittest.main()
