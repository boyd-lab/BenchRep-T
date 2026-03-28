# -*- coding: utf-8 -*-
"""
Reading repertoire datasets from hdf5 containers.

See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
See `deeprc/examples/` for examples.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
import numpy as np
import h5py
import pandas as pd
from typing import Tuple, Callable, Union
import torch
from torch.utils.data import Dataset, DataLoader
from deeprc.dataset_converters import DatasetToHDF5
from deeprc.task_definitions import TaskDefinition


def log_sequence_count_scaling(seq_counts: np.ndarray):
    """Scale sequence counts `seq_counts` using a natural element-wise logarithm. Values `< 1` are set to `1`.
    To be used for `deeprc.dataset_readers.make_dataloaders`.
    
    Parameters
    ----------
    seq_counts
        Sequence counts as numpy array.
    
    Returns
    ---------
    scaled_seq_counts
        Scaled sequence counts as numpy array.
    """
    return np.log(np.maximum(seq_counts, 1))


def no_sequence_count_scaling(seq_counts: np.ndarray):
    """No scaling of sequence counts `seq_counts`. Values `< 0` are set to `0`.
    To be used for `deeprc.dataset_readers.make_dataloaders`.
    
    Parameters
    ----------
    seq_counts
        Sequence counts as numpy array.
    
    Returns
    ---------
    scaled_seq_counts
        Scaled sequence counts as numpy array.
    """
    return np.maximum(seq_counts, 0)


def make_dataloaders(task_definition: TaskDefinition, metadata_file: str, repertoiresdata_path: str,
                     split_inds: list = None, n_splits: int = 5, cross_validation_fold: int = 0, rnd_seed: int = 0,
                     n_worker_processes: int = 4, batch_size: int = 4,
                     inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                     sample_n_sequences: int = 10000,
                     metadata_file_id_column: str = 'ID', metadata_file_column_sep: str = '\t',
                     sequence_column: str = 'amino_acid', sequence_counts_column: str = 'templates',
                     repertoire_files_column_sep: str = '\t', filename_extension: str = '.tsv', h5py_dict: dict = None,
                     sequence_counts_scaling_fn: Callable = no_sequence_count_scaling, verbose: bool = True) \
        -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get data loaders for a dataset
    
    Get data loaders for training set in training mode (with random subsampling) and training-, validation-, and
    test-set in evaluation mode (without random subsampling).
    Creates PyTorch data loaders for hdf5 containers or `.tsv`/`.csv` files, which will be converted to hdf5 containers
    on-the-fly (see dataset_converters.py).
    If provided,`set_inds` will determine which sample belongs to which split, otherwise random assignment of 
    3/5, 1/5, and 1/5 samples to the three splits is performed. Indices in `set_inds` correspond to line indices
    (excluding the header line) in `metadata_file`.
    
    See `deeprc/examples/` for examples with custom datasets and datasets used in papers.
    See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
    
    Parameters
    ----------
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
    metadata_file : str
        Filepath of metadata .tsv file with targets.
    repertoiresdata_path : str
        Filepath of hdf5 file containing repertoire sequence data or filepath of folder containing the repertoire
        `.tsv`/`.csv` files. `.tsv`/`.csv` will be converted to a hdf5 file.
    split_inds : list of iterable
        Optional: List of iterables of repertoire indices. Each iterable in `split_inds` represents a dataset split.
        For 5-fold cross-validation, `split_inds` should contain 5 lists of repertoire indices, with non-overlapping
        repertoire indices.
        Indices in `set_inds` correspond to line indices (excluding the header line) in `metadata_file`.
        If None, the repertoire indices will be assigned to `n_splits` different splits randomly using `rnd_seed`.
    n_splits
        Optional: If `split_inds` is None, `n_splits` random dataset splits for the cross-validation are created.
    cross_validation_fold : int
        Specify the fold of the cross-validation the dataloaders should be computed for.
    rnd_seed : int
        Seed for the random generator to create the random dataset splits. Only used if `split_inds=None`.
    n_worker_processes : int
        Number of background processes to use for converting dataset to hdf5 container and trainingset dataloader.
    batch_size : int
        Number of repertoires per minibatch during training.
    inputformat : 'NCL' or 'NLC'
        Format of input feature array;
        'NCL' -> (batchsize, channels, seq.length);
        'LNC' -> (seq.length, batchsize, channels);
    keep_dataset_in_ram : bool
        It is faster to load the full hdf5 file into the RAM instead of keeping it on the disk.
        If False, the hdf5 file will be read from the disk and consume less RAM.
    sample_n_sequences : int
        Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
        Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
        random indices have been drawn multiple times.
        If None, all sequences will be loaded for each repertoire.
    metadata_file_id_column : str
        Name of column holding the repertoire names in`metadata_file`.
    metadata_file_column_sep : str
        The column separator in `metadata_file`.
    sequence_column : str
        Optional: The name of the column that includes the AA sequences (only for hdf5-conversion).
    sequence_counts_column : str
        Optional: The name of the column that includes the sequence counts (only for hdf5-conversion).
    repertoire_files_column_sep : str
        Optional: The column separator in the repertoire files (only for hdf5-conversion).
    filename_extension : str
        Filename extension of the metadata and repertoire files. (For repertoire files only for hdf5-conversion.)
    h5py_dict : dict ot None
        Dictionary with kwargs for creating h5py datasets;
        Defaults to `gzip` compression at level `4` if None; (only for hdf5-conversion)
    sequence_counts_scaling_fn
        Scaling function for sequence counts. E.g. `deeprc.dataset_readers.log_sequence_count_scaling` or
        `deeprc.dataset_readers.no_sequence_count_scaling`.
    verbose : bool
        Activate verbose mode
        
    Returns
    ---------
    trainingset_dataloader: DataLoader
        Dataloader for trainingset with active `sample_n_sequences` (=random subsampling/dropout of repertoire
        sequences)
    trainingset_eval_dataloader: DataLoader
        Dataloader for trainingset with deactivated `sample_n_sequences`
    validationset_eval_dataloader: DataLoader
        Dataloader for validationset with deactivated `sample_n_sequences`
    testset_eval_dataloader: DataLoader
        Dataloader for testset with deactivated `sample_n_sequences`
    """
    #
    # Convert dataset to hdf5 container if no hdf5 container was specifies
    #
    try:
        with h5py.File(repertoiresdata_path, 'r') as hf:
            n_repertoires = hf['metadata']['n_samples'][()]
        hdf5_file = repertoiresdata_path
    except Exception:
        # Convert to hdf5 container if no hdf5 container was given
        hdf5_file = repertoiresdata_path + ".hdf5"
        user_input = None
        while user_input != 'y':
            user_input = input(f"Path {repertoiresdata_path} is not a hdf container. "
                               f"Should I create an hdf5 container {hdf5_file}? (y/n)")
            if user_input == 'n':
                print("Process aborted by user")
                exit()
        if verbose:
            print(f"Converting: {repertoiresdata_path}\n->\n{hdf5_file} @{n_worker_processes} processes")
        converter = DatasetToHDF5(
                repertoiresdata_directory=repertoiresdata_path, sequence_column=sequence_column,
                sequence_counts_column=sequence_counts_column, column_sep=repertoire_files_column_sep,
                filename_extension=filename_extension, h5py_dict=h5py_dict, verbose=verbose)
        converter.save_data_to_file(output_file=hdf5_file, n_workers=n_worker_processes)
        with h5py.File(hdf5_file, 'r') as hf:
            n_repertoires = hf['metadata']['n_samples'][()]
        if verbose:
            print(f"\tSuccessfully created {hdf5_file}!")
    
    #
    # Create dataset
    #
    if verbose:
        print(f"Creating dataloader from repertoire files in {hdf5_file}")
    full_dataset = RepertoireDataset(metadata_filepath=metadata_file, hdf5_filepath=hdf5_file,
                                     sample_id_column=metadata_file_id_column,
                                     metadata_file_column_sep=metadata_file_column_sep,
                                     task_definition=task_definition, keep_in_ram=keep_dataset_in_ram,
                                     inputformat=inputformat, sequence_counts_scaling_fn=sequence_counts_scaling_fn)
    n_samples = len(full_dataset)
    if verbose:
        print(f"\tFound and loaded a total of {n_samples} samples")
    
    #
    # Create dataset split indices
    #
    if split_inds is None:
        if verbose:
            print("Computing random split indices")
        n_repertoires_per_split = int(n_repertoires / n_splits)
        rnd_gen = np.random.RandomState(rnd_seed)
        shuffled_repertoire_inds = rnd_gen.permutation(n_repertoires)
        split_inds = [shuffled_repertoire_inds[s_i*n_repertoires_per_split:(s_i+1)*n_repertoires_per_split]
                      if s_i != n_splits-1 else
                      shuffled_repertoire_inds[s_i*n_repertoires_per_split:]  # Remaining repertoires to last split
                      for s_i in range(n_splits)]
    else:
        split_inds = [np.array(split_ind, dtype=int) for split_ind in split_inds]
    
    if cross_validation_fold >= len(split_inds):
        raise ValueError(f"Demanded `cross_validation_fold` {cross_validation_fold} but only {len(split_inds)} splits "
                         f"exist in `split_inds`.")
    testset_inds = split_inds.pop(cross_validation_fold)
    validationset_inds = split_inds.pop(cross_validation_fold-1)
    trainingset_inds = np.concatenate(split_inds)
    
    #
    # Create datasets and dataloaders for splits
    #
    if verbose:
        print("Creating dataloaders for dataset splits")
    
    training_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=trainingset_inds, sample_n_sequences=sample_n_sequences)
    trainingset_dataloader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker_processes,
            collate_fn=no_stack_collate_fn)

    training_eval_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=trainingset_inds, sample_n_sequences=None)
    trainingset_eval_dataloader = DataLoader(
            training_eval_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=no_stack_collate_fn)
    
    validationset_eval_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=validationset_inds, sample_n_sequences=None)
    validationset_eval_dataloader = DataLoader(
            validationset_eval_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=no_stack_collate_fn)
    
    testset_eval_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=testset_inds, sample_n_sequences=None)
    testset_eval_dataloader = DataLoader(
            testset_eval_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=no_stack_collate_fn)
    
    if verbose:
        print("\tDone!")
    
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader


def no_stack_collate_fn(batch_as_list: list):
    """Function to be passed to `torch.utils.data.DataLoader` as `collate_fn`
    
    Instead of stacking the samples in a minibatch into one torch.tensor object, sample entries will be individually
    converted to torch.tensor objects and packed into a list instead.
    Objects that could not be converted to torch.tensor objects are packed into a list without conversion.
    """
    # Go through all samples, convert entries that are numpy to tensor and put entries in lists
    list_batch = [[torch.from_numpy(sample[entry_i]) for sample in batch_as_list]
                  if isinstance(batch_as_list[0][entry_i], np.ndarray) else
                  [sample[entry_i] for sample in batch_as_list]
                  for entry_i in range(len(batch_as_list[0]))]
    return list_batch


def str_or_byte_to_str(str_or_byte: Union[str, bytes], decoding: str = 'utf8') -> str:
    """Convenience function to increase compatibility with different h5py versions"""
    return str_or_byte.decode(decoding) if isinstance(str_or_byte, bytes) else str_or_byte


class RepertoireDataset(Dataset):
    def __init__(self, metadata_filepath: str, hdf5_filepath: str, inputformat: str = 'NCL',
                 sample_id_column: str = 'ID', metadata_file_column_sep: str = '\t',
                 task_definition: TaskDefinition = None,
                 keep_in_ram: bool = True, sequence_counts_scaling_fn: Callable = no_sequence_count_scaling,
                 sample_n_sequences: int = None, verbose: bool = True):
        """PyTorch Dataset class for reading repertoire dataset from metadata file and hdf5 file
        
        See `deeprc.dataset_readers.make_dataloaders` for simple loading of datasets via PyTorch data loader.
        See `deeprc.dataset_readers.make_dataloaders` or `dataset_converters.py` for conversion of `.tsv`/`.csv` files
         to hdf5 container.
        
        Parameters
        ----------
        metadata_filepath : str
            Filepath of metadata `.tsv`/`.csv` file with targets used by `task_definition`.
        hdf5_filepath : str
            Filepath of hdf5 file containing repertoire sequence data.
        inputformat : 'NCL' or 'NLC'
            Format of input feature array;
            'NCL' -> (batchsize, channels, seq_length);
            'LNC' -> (seq_length, batchsize, channels);
        task_definition: TaskDefinition
            TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for
             examples.
        keep_in_ram : bool
            It is faster to load the hdf5 file into the RAM as dictionary instead of keeping it on the disk.
            If False, the hdf5 file will be read from the disk dynamically, which is slower but consume less RAM.
        sequence_counts_scaling_fn
            Scaling function for sequence counts. E.g. `deeprc.dataset_readers.log_sequence_count_scaling` or
            `deeprc.dataset_readers.no_sequence_count_scaling`.
        sample_n_sequences : int
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, all sequences will be loaded for each repertoire.
            Can be set for individual samples using `sample_n_sequences` parameter of __getitem__() method.
        verbose : bool
            Activate verbose mode
        """
        self.metadata_filepath = metadata_filepath
        self.filepath = hdf5_filepath
        self.inputformat = inputformat
        self.task_definition = task_definition
        self.sample_id_column = sample_id_column
        self.keep_in_ram = keep_in_ram
        self.sequence_counts_scaling_fn = sequence_counts_scaling_fn
        self.metadata_file_column_sep = metadata_file_column_sep
        self.sample_n_sequences = sample_n_sequences
        self.sequence_counts_hdf5_key = 'sequence_counts'
        self.sequences_hdf5_key = 'sequences'
        self.verbose = verbose
        
        if self.inputformat not in ['NCL', 'LNC']:
            raise ValueError(f"Unsupported input format {self.inputformat}")
        
        # Read target data from csv file
        self.metadata = pd.read_csv(self.metadata_filepath, sep=self.metadata_file_column_sep, header=0, dtype=str)
        self.metadata.index = self.metadata[self.sample_id_column].values
        self.sample_keys = np.array([os.path.splitext(k)[0] for k in self.metadata[self.sample_id_column].values])
        self.n_samples = len(self.sample_keys)
        self.target_features = self.task_definition.get_targets(self.metadata)
        
        # Read sequence data from hdf5 file
        with h5py.File(self.filepath, 'r') as hf:
            metadata = hf['metadata']
            # Add characters for 3 position features to list of AAs
            self.aas = str_or_byte_to_str(metadata['aas'][()])
            self.aas += ''.join(['<', '>', '^'])
            self.n_features = len(self.aas)
            self.stats = str_or_byte_to_str(metadata['stats'][()])
            self.n_samples = metadata['n_samples'][()]
            hdf5_sample_keys = [str_or_byte_to_str(os.path.splitext(k)[0]) for k in metadata['sample_keys'][:]]
            
            # Mapping metadata sample indices -> hdf5 file sample indices
            unfound_samples = np.array([sk not in hdf5_sample_keys for sk in self.sample_keys], dtype=bool)
            if np.any(unfound_samples):
                raise KeyError(f"Samples {self.sample_keys[unfound_samples]} "
                               f"could not be found in hdf5 file. Please add the samples and re-create the hdf5 file "
                               f"or remove the sample keys from the used samples of the metadata file.")
            self.hdf5_inds = np.array([hdf5_sample_keys.index(sk) for sk in self.sample_keys], dtype=int)
            
            # Support old hdf5 format and check for missing hdf5 keys
            if self.sequence_counts_hdf5_key not in hf['sampledata'].keys():
                if 'duplicates_per_sequence' in hf['sampledata'].keys():
                    self.sequence_counts_hdf5_key = 'duplicates_per_sequence'
                elif 'counts_per_sequence' in hf['sampledata'].keys():
                    self.sequence_counts_hdf5_key = 'counts_per_sequence'
                else:
                    raise KeyError(f"Could not locate entry {self.sequence_counts_hdf5_key}, which should contains "
                                   f"sequence counts, in hdf5 file. Only found keys {list(hf['sampledata'].keys())}.")
            if self.sequences_hdf5_key not in hf['sampledata'].keys():
                if 'amino_acid_sequences' in hf['sampledata'].keys():
                    self.sequences_hdf5_key = 'amino_acid_sequences'
                else:
                    raise KeyError(f"Could not locate entry {self.sequences_hdf5_key}, which should contains "
                                   f"sequence counts, in hdf5 file. Only found keys {list(hf['sampledata'].keys())}.")
            
            if keep_in_ram:
                sampledata = dict()
                sampledata['seq_lens'] = hf['sampledata']['seq_lens'][:]
                sampledata[self.sequence_counts_hdf5_key] =\
                    np.array(hf['sampledata'][self.sequence_counts_hdf5_key][:], dtype=np.float32)
                if np.any(sampledata[self.sequence_counts_hdf5_key] <= 0):
                    print(f"Warning: Found {(sampledata[self.sequence_counts_hdf5_key] <= 0).sum()} sequences with "
                          f"counts <= 0. They will be handled as specified in the sequence_counts_scaling_fn "
                          f"{sequence_counts_scaling_fn} passed to RepertoireDataset.")
                sampledata[self.sequences_hdf5_key] = hf['sampledata'][self.sequences_hdf5_key][:]
                self.sampledata = sampledata
            else:
                self.sampledata = None
            
            sample_sequences_start_end = hf['sampledata']['sample_sequences_start_end'][:]
            self.sample_sequences_start_end = sample_sequences_start_end[self.hdf5_inds]
            
        self._vprint("File Stats:")
        self._vprint("  " + "  \n".join(self.stats.split('; ')))
        self._vprint(f"Used samples: {self.n_samples}")
    
    def get_sample(self, idx: int, sample_n_sequences: Union[None, int] = None):
        """ Return repertoire with index idx from dataset, randomly sub-/up-sampled to `sample_n_sequences` sequences
        
        Parameters
        ----------
        idx: int
            Index of repertoire to return
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, will use `sample_n_sequences` as specified when creating `RepertoireDataset` instance.
        
        Returns
        ---------
        aa_sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts per sequence in repertoire.
        """
        sample_sequences_start_end = self.sample_sequences_start_end[idx]
        if sample_n_sequences:
            rnd_gen = np.random.RandomState()  # TODO: Add shared memory integer random seed for dropout
            sample_sequence_inds = np.unique(rnd_gen.randint(
                    low=sample_sequences_start_end[0], high=sample_sequences_start_end[1],
                    size=sample_n_sequences))
            if self.sampledata is None:
                # Compatibility for indexing hdf5 file
                sample_sequence_inds = list(sample_sequence_inds)
        else:
            sample_sequence_inds = slice(sample_sequences_start_end[0], sample_sequences_start_end[1])
    
        with h5py.File(self.filepath, 'r') as hf:
            if self.sampledata is not None:
                sampledata = self.sampledata
            else:
                sampledata = hf['sampledata']
            
            seq_lens = sampledata['seq_lens'][sample_sequence_inds]
            sample_max_seq_len = seq_lens.max()
            aa_sequences = sampledata[self.sequences_hdf5_key][sample_sequence_inds, :sample_max_seq_len]
            counts_per_sequence = \
                self.sequence_counts_scaling_fn(sampledata[self.sequence_counts_hdf5_key][sample_sequence_inds])
    
        if self.inputformat.startswith('LN'):
            aa_sequences = np.swapaxes(aa_sequences, 0, 1)
        return aa_sequences, seq_lens, counts_per_sequence
    
    def inds_to_aa(self, inds: np.array):
        """Convert array of AA indices to character array (see also `self.inds_to_aa_ignore_negative()`)"""
        lookup = np.chararray(shape=(len(self.aas),))
        lookup[:] = list(self.aas)
        char_array = lookup[inds]
        return char_array
    
    def inds_to_aa_ignore_negative(self, inds: np.array):
        """Convert array of AA indices to character array, ignoring '-1'-padding to equal sequence length"""
        lookup = np.chararray(shape=(len(self.aas),))
        lookup[:] = list(self.aas)
        char_array = lookup[inds[inds >= 0]].tostring().decode('utf8')
        return char_array
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx, sample_n_sequences: Union[None, int] = None):
        """ Return repertoire with index idx from dataset, randomly sub-/up-sampled to `sample_n_sequences` sequences
        
        Parameters
        ----------
        idx: int
            Index of repertoire to return
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, will use `sample_n_sequences` as specified when creating `RepertoireDataset` instance.
        
        Returns
        ---------
        target_features: numpy float32 array
            Target feature vector.
        sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts per sequence in repertoire.
        sample_id: str
            Sample ID.
        """
        target_features = self.target_features[idx]
        sample_id = str(self.sample_keys[idx])
        if sample_n_sequences is None:
            sample_n_sequences = self.sample_n_sequences
        sequences, seq_lens, counts_per_sequence = self.get_sample(idx, sample_n_sequences)
        return target_features, sequences, seq_lens, counts_per_sequence, sample_id
    
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class AIRRRepertoireDataset(Dataset):
    """PyTorch Dataset that reads AIRR-format .tsv/.tsv.gz repertoire files directly.

    Eliminates the need for a pre-built HDF5 container.  Each repertoire file
    is read on demand from ``__getitem__``.  The interface (return values,
    ``aas`` attribute, ``inds_to_aa*`` helpers) is identical to
    ``RepertoireDataset`` so it works transparently with
    ``RepertoireDatasetSubset`` and the existing ``train`` / ``evaluate``
    functions.
    """

    # Standard 20 amino acids used by DeepRC (same order as DatasetToHDF5)
    _AAS = 'ACDEFGHIKLMNPQRSTVWY'

    def __init__(self, file_paths: list, labels: np.ndarray,
                 sample_ids: list,
                 sequence_col: str = 'cdr3_aa',
                 count_col: str = 'duplicate_count',
                 inputformat: str = 'NCL',
                 sample_n_sequences: int = None,
                 sequence_counts_scaling_fn: Callable = no_sequence_count_scaling,
                 keep_in_ram: bool = True,
                 prebuilt_cache: list = None,
                 indices_map: dict = None,
                 verbose: bool = False):
        """
        Parameters
        ----------
        file_paths : list of str
            One .tsv or .tsv.gz AIRR file per repertoire, in the same order as
            ``labels`` and ``sample_ids``.
        labels : np.ndarray, shape (n_samples, n_output_features), float32
            Pre-computed target feature vectors (from ``TaskDefinition.get_targets``).
        sample_ids : list of str
            Identifier string for each repertoire (used in output rows).
        sequence_col : str
            AIRR column containing CDR3 amino acid sequences.
        count_col : str
            AIRR column containing sequence counts.  If absent, every sequence
            gets a count of 1.
        inputformat : 'NCL' or 'LNC'
            Array layout expected by the model.
        sample_n_sequences : int or None
            If set, randomly sub-sample this many sequences per repertoire
            during ``__getitem__``.  None means use all sequences.
        sequence_counts_scaling_fn : callable
            Scaling function applied to raw counts.
        keep_in_ram : bool
            If True (default), all repertoire files are read and encoded once
            during ``__init__`` and kept in RAM.  This matches the behaviour of
            ``RepertoireDataset(keep_in_ram=True)`` and prevents the GPU from
            being starved by repeated disk I/O during training.
        prebuilt_cache : list or None
            Optional pre-built cache (list of ``(encoded, seq_lens, counts)``
            tuples, one per repertoire).  If supplied, ``keep_in_ram`` and disk
            reads are skipped entirely — useful for sharing the cache between
            the training and training-eval dataloaders.
        indices_map : dict or None
            Optional mapping from repertoire ID (filename without extension,
            e.g. ``'part_table_P1_S1'``) to a list of integer row indices.
            When provided, only those rows are used from each repertoire file.
            This is used for sequencing-depth sub-sampling experiments.
        verbose : bool
            Print per-file warnings.
        """
        if inputformat not in ('NCL', 'LNC'):
            raise ValueError(f"Unsupported inputformat '{inputformat}'")

        self.file_paths = list(file_paths)
        self.target_features = labels          # shape (n_samples, n_out)
        self.sample_keys = np.array(sample_ids, dtype=object)
        self.n_samples = len(file_paths)
        self.sequence_col = sequence_col
        self.count_col = count_col
        self.inputformat = inputformat
        self.sample_n_sequences = sample_n_sequences
        self.sequence_counts_scaling_fn = sequence_counts_scaling_fn
        self.indices_map = indices_map
        self.verbose = verbose

        # AA alphabet + 3 positional tokens (identical to RepertoireDataset)
        self.aas = self._AAS + '<>^'
        self.n_features = len(self.aas)
        self._aa_to_idx = {c: i for i, c in enumerate(self._AAS)}

        # Byte-level lookup table for fast vectorised encoding (ord -> aa_idx)
        self._byte_lookup = np.full(256, -1, dtype=np.int8)
        for char, idx in self._aa_to_idx.items():
            self._byte_lookup[ord(char)] = idx

        # Pre-load all repertoires into RAM so __getitem__ never touches disk.
        # A prebuilt_cache (list of (encoded, seq_lens, counts) tuples) can be
        # passed in to skip the disk read entirely — used to share the cache
        # between the training and training-eval dataloaders.
        if prebuilt_cache is not None:
            self._ram_cache = prebuilt_cache
        elif keep_in_ram:
            if verbose:
                print(f"AIRRRepertoireDataset: pre-loading {self.n_samples} repertoires into RAM...")
            self._ram_cache = [self._read_repertoire(fp) for fp in self.file_paths]
            if verbose:
                print("  Done.")
        else:
            self._ram_cache = None

    # ------------------------------------------------------------------
    # Helpers (same API as RepertoireDataset)
    # ------------------------------------------------------------------

    def inds_to_aa(self, inds: np.ndarray):
        lookup = np.chararray(shape=(len(self.aas),))
        lookup[:] = list(self.aas)
        return lookup[inds]

    def inds_to_aa_ignore_negative(self, inds: np.ndarray):
        lookup = np.chararray(shape=(len(self.aas),))
        lookup[:] = list(self.aas)
        return lookup[inds[inds >= 0]].tostring().decode('utf8')

    # ------------------------------------------------------------------
    # Core reading logic
    # ------------------------------------------------------------------

    def _read_repertoire(self, file_path: str):
        """Read one AIRR file and return encoded sequences, lengths, and counts."""
        df = pd.read_csv(file_path, sep='\t', low_memory=False, keep_default_na=False)

        if self.indices_map is not None:
            rep_id = os.path.basename(file_path).replace('.tsv.gz', '').replace('.tsv', '')
            indices = self.indices_map.get(rep_id)
            if indices is not None:
                df = df.iloc[indices].reset_index(drop=True)

        if self.sequence_col not in df.columns:
            raise ValueError(
                f"Column '{self.sequence_col}' not found in {file_path}. "
                f"Available: {list(df.columns)}"
            )

        seqs = df[self.sequence_col].astype(str).values

        # Raw counts
        if self.count_col in df.columns:
            counts = pd.to_numeric(df[self.count_col], errors='coerce').fillna(1).values
        else:
            counts = np.ones(len(seqs), dtype=np.float32)

        # Filter: keep only non-empty sequences composed entirely of valid AAs.
        # Uses the byte lookup table to avoid a Python character loop per sequence.
        def _all_valid(s):
            if not s:
                return False
            b = np.frombuffer(s.encode('ascii', errors='replace'), dtype=np.uint8)
            return bool(np.all(self._byte_lookup[b] >= 0))

        valid_mask = np.array([_all_valid(s) for s in seqs], dtype=bool)
        seqs = seqs[valid_mask]
        counts = counts[valid_mask].astype(np.float32)

        if len(seqs) == 0:
            # Return a single dummy sequence so downstream code never sees empty
            seqs = np.array(['A'])
            counts = np.array([1.0], dtype=np.float32)
            if self.verbose:
                print(f"Warning: no valid sequences found in {file_path}; using dummy.")

        seq_lens = np.array([len(s) for s in seqs], dtype=np.int64)
        max_len = int(seq_lens.max())

        # Encode as int8 padded array (-1 = padding) using vectorised byte lookup
        encoded = np.full((len(seqs), max_len), fill_value=-1, dtype=np.int8)
        for i, s in enumerate(seqs):
            b = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
            encoded[i, :len(b)] = self._byte_lookup[b]

        return encoded, seq_lens, counts

    def get_sample(self, idx: int, sample_n_sequences=None):
        if self._ram_cache is not None:
            encoded, seq_lens, counts = self._ram_cache[idx]
            # Work on copies so subsampling doesn't mutate the cache
            encoded = encoded.copy()
            seq_lens = seq_lens.copy()
            counts = counts.copy()
        else:
            encoded, seq_lens, counts = self._read_repertoire(self.file_paths[idx])

        n_seq = len(seq_lens)
        if sample_n_sequences and sample_n_sequences < n_seq:
            rnd = np.random.RandomState()
            chosen = np.unique(rnd.randint(0, n_seq, size=sample_n_sequences))
            encoded = encoded[chosen]
            seq_lens = seq_lens[chosen]
            counts = counts[chosen]

        # Trim encoded columns to actual max length of the (possibly subsampled)
        # sequences so that encoded.shape[1] == seq_lens.max().
        # reduce_and_stack_minibatch derives max_mb_seq_len from seq_lens.max()
        # across the batch; if encoded.shape[1] ever exceeds that value the
        # subsequent slice clips silently and the reshape sizes diverge.
        encoded = encoded[:, :int(seq_lens.max())]

        counts = self.sequence_counts_scaling_fn(counts)

        if self.inputformat.startswith('LN'):
            encoded = np.swapaxes(encoded, 0, 1)

        return encoded, seq_lens, counts

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx, sample_n_sequences=None):
        if sample_n_sequences is None:
            sample_n_sequences = self.sample_n_sequences
        sequences, seq_lens, counts = self.get_sample(idx, sample_n_sequences)
        target_features = self.target_features[idx]
        sample_id = str(self.sample_keys[idx])
        return target_features, sequences, seq_lens, counts, sample_id

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


def make_dataloaders_from_airr(
        task_definition: TaskDefinition,
        train_metadata: pd.DataFrame,
        val_metadata: pd.DataFrame,
        test_metadata: pd.DataFrame,
        file_path_col: str = 'file_path',
        label_col: str = 'label',
        sample_id_col: str = 'specimen_label',
        sequence_col: str = 'cdr3_aa',
        count_col: str = 'duplicate_count',
        inputformat: str = 'NCL',
        sample_n_sequences: int = 10000,
        batch_size: int = 4,
        n_worker_processes: int = 4,
        sequence_counts_scaling_fn: Callable = no_sequence_count_scaling,
        keep_in_ram: bool = True,
        indices_map: dict = None,
        verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders directly from AIRR .tsv/.tsv.gz files (no HDF5 needed).

    Parameters
    ----------
    task_definition : TaskDefinition
    train_metadata, val_metadata, test_metadata : pd.DataFrame
        DataFrames with at least ``file_path_col``, ``label_col``, and
        ``sample_id_col`` columns.
    file_path_col : str
        Column containing the path to each AIRR repertoire file.
    label_col : str
        Column with integer (0/1) labels; must match the column expected by
        ``task_definition``.
    sample_id_col : str
        Column with specimen / sample identifiers.
    sequence_col : str
        AIRR column name for CDR3 amino acid sequences.
    count_col : str
        AIRR column name for duplicate counts.
    inputformat : str
        'NCL' or 'LNC'.
    sample_n_sequences : int or None
        Sequences randomly sub-sampled per repertoire during training.
        None → use all.
    batch_size : int
        Repertoires per mini-batch for the training DataLoader.
    n_worker_processes : int
        Worker processes for the training DataLoader.
    sequence_counts_scaling_fn : callable
        Count scaling function (e.g. ``log_sequence_count_scaling``).
    indices_map : dict or None
        Optional mapping from repertoire ID (filename without extension) to a
        list of integer row indices for depth sub-sampling experiments.
    verbose : bool

    Returns
    -------
    trainingset_dataloader, trainingset_eval_dataloader,
    validationset_eval_dataloader, testset_eval_dataloader : DataLoader
    """

    def _build_dataset(meta: pd.DataFrame, subsample,
                       prebuilt_cache=None) -> AIRRRepertoireDataset:
        # Build a tiny DataFrame with the label column so get_targets works
        label_df = meta[[sample_id_col, label_col]].copy()
        label_df = label_df.rename(columns={sample_id_col: 'ID', label_col: 'label'})
        label_df['label'] = label_df['label'].astype(str)
        label_df = label_df.set_index('ID')
        targets = task_definition.get_targets(label_df)

        return AIRRRepertoireDataset(
            file_paths=meta[file_path_col].tolist(),
            labels=targets,
            sample_ids=meta[sample_id_col].tolist(),
            sequence_col=sequence_col,
            count_col=count_col,
            inputformat=inputformat,
            sample_n_sequences=subsample,
            sequence_counts_scaling_fn=sequence_counts_scaling_fn,
            keep_in_ram=keep_in_ram,
            prebuilt_cache=prebuilt_cache,
            indices_map=indices_map,
            verbose=verbose,
        )

    if verbose:
        print(f"Building AIRR dataloaders: "
              f"train={len(train_metadata)}, val={len(val_metadata)}, test={len(test_metadata)}")

    # Build train dataset first, then reuse its cache for the eval view of the
    # same split — avoids reading the same 155+ files twice.
    train_ds      = _build_dataset(train_metadata, sample_n_sequences)
    train_eval_ds = _build_dataset(train_metadata, None,
                                   prebuilt_cache=train_ds._ram_cache)
    val_ds        = _build_dataset(val_metadata,   None)
    test_ds       = _build_dataset(test_metadata,  None)

    trainingset_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_worker_processes, collate_fn=no_stack_collate_fn)
    trainingset_eval_dataloader = DataLoader(
        train_eval_ds, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=no_stack_collate_fn)
    validationset_eval_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=no_stack_collate_fn)
    testset_eval_dataloader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=no_stack_collate_fn)

    if verbose:
        print("  Done building dataloaders.")

    return (trainingset_dataloader, trainingset_eval_dataloader,
            validationset_eval_dataloader, testset_eval_dataloader)


class RepertoireDatasetSubset(Dataset):
    def __init__(self, dataset: RepertoireDataset, indices: Union[list, np.ndarray], sample_n_sequences: int = None):
        """Create subset of `deeprc.dataset_readers.RepertoireDataset` dataset
        
        Parameters
        ----------
        dataset
            A `deeprc.dataset_readers.RepertoireDataset` dataset instance
        indices
            List of indices that the subset of `dataset` should contain
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, all sequences will be loaded as specified in `dataset`.
            Can be set for individual samples using `sample_n_sequences` parameter of __getitem__() method.
        """
        self.indices = np.asarray(indices, dtype=int)
        self.sample_n_sequences = sample_n_sequences
        self.repertoire_reader = dataset
        
        self.inds_to_aa = self.repertoire_reader.inds_to_aa
        self.aas = self.repertoire_reader.aas
        self.inds_to_aa_ignore_negative = self.repertoire_reader.inds_to_aa_ignore_negative
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx, sample_n_sequences: Union[None, int] = None):
        """ Return repertoire with index idx from dataset, randomly sub-/up-sampled to `sample_n_sequences` sequences
        
        Parameters
        ----------
        idx: int
            Index of repertoire to return
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, will use `sample_n_sequences` as specified when creating `RepertoireDatasetSubset` instance.
        
        Returns
        ---------
        target_features: numpy float32 array
            Target feature vector.
        sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts per sequence in repertoire.
        sample_id: str
            Sample ID.
        """
        if sample_n_sequences is None:
            sample_n_sequences = self.sample_n_sequences
        target_features, sequences, seq_lens, counts_per_sequence, sample_id = \
            self.repertoire_reader.__getitem__(self.indices[idx], sample_n_sequences=sample_n_sequences)
        return target_features, sequences, seq_lens, counts_per_sequence, sample_id
