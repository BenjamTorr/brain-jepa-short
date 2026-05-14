"""ABCD fMRI dataset for Brain-JEPA pretraining.

Loads a monolithic .npy of shape (N, n_rois, seq_length) together with a
metadata CSV (one row per sample, same order as the array). Splits are built at
the *subject* level so that all visits/events for a given src_subject_id land
in exactly one of train/val/test -- prevents leakage from longitudinal scans.

Sample output: {'fmri': torch.float32 tensor of shape (1, n_rois, seq_length)}.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from logging import getLogger

logger = getLogger()


def subject_split_indices(
    metadata_path,
    subject_col='src_subject_id',
    val_frac=0.1,
    test_frac=0.1,
    seed=0,
):
    """Deterministically partition row indices into train/val/test by subject.

    All rows sharing a subject ID go to the same split. Returns a dict mapping
    split name -> 1-D np.int64 array of row indices into the .npy/metadata.
    """
    md = pd.read_csv(metadata_path, low_memory=False)
    assert subject_col in md.columns, f'metadata missing column {subject_col}'

    subjects = md[subject_col].to_numpy()
    unique_subjects = np.array(sorted(pd.unique(subjects)))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_subjects))
    shuffled = unique_subjects[perm]

    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    assert n_train > 0, 'split fractions leave no training subjects'

    train_subj = set(shuffled[:n_train].tolist())
    val_subj = set(shuffled[n_train:n_train + n_val].tolist())
    test_subj = set(shuffled[n_train + n_val:].tolist())

    def rows_for(subj_set):
        mask = np.fromiter((s in subj_set for s in subjects), dtype=bool, count=len(subjects))
        return np.nonzero(mask)[0].astype(np.int64)

    return {
        'train': rows_for(train_subj),
        'val': rows_for(val_subj),
        'test': rows_for(test_subj),
    }


class ABCDfMRIDataset(Dataset):
    def __init__(
        self,
        ts_path,
        metadata_path,
        split='train',
        n_rois=100,
        seq_length=1500,
        val_frac=0.1,
        test_frac=0.1,
        split_seed=0,
        subject_col='src_subject_id',
        use_standatdization=False,
    ):
        assert split in ('train', 'val', 'test')
        self.ts_path = ts_path
        self.metadata_path = metadata_path
        self.split = split
        self.n_rois = n_rois
        self.seq_length = seq_length
        self.use_standatdization = use_standatdization

        self._arr = np.load(ts_path, mmap_mode='r')
        assert self._arr.ndim == 3, f'expected (N, R, T), got {self._arr.shape}'
        assert self._arr.shape[1] == n_rois, \
            f'ts has {self._arr.shape[1]} ROIs, config expects {n_rois}'
        assert self._arr.shape[2] >= seq_length, \
            f'ts has {self._arr.shape[2]} timepoints, config expects >= {seq_length}'

        md = pd.read_csv(metadata_path, low_memory=False)
        assert len(md) == self._arr.shape[0], \
            f'metadata rows ({len(md)}) != ts samples ({self._arr.shape[0]})'

        splits = subject_split_indices(
            metadata_path,
            subject_col=subject_col,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=split_seed,
        )
        self._indices = splits[split]

        n_subj_split = md.iloc[self._indices][subject_col].nunique()
        n_subj_total = md[subject_col].nunique()
        logger.info(
            f'ABCDfMRIDataset[{split}] samples={len(self._indices)} '
            f'subjects={n_subj_split}/{n_subj_total} (seed={split_seed})'
        )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        row = self._indices[idx]
        ts = np.asarray(self._arr[row], dtype=np.float32)

        if ts.shape[1] > self.seq_length:
            ts = ts[:, :self.seq_length]
        elif ts.shape[1] < self.seq_length:
            pad = np.zeros((ts.shape[0], self.seq_length - ts.shape[1]), dtype=np.float32)
            ts = np.concatenate([ts, pad], axis=1)

        ts = torch.from_numpy(ts).unsqueeze(0).to(torch.float32)  # (1, R, T)

        if self.use_standatdization:
            ts = (ts - ts.mean()) / (ts.std() + 1e-6)

        return {'fmri': ts}


def make_abcd(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    downsample=False,
    use_standatdization=False,
    ts_path=None,
    metadata_path=None,
    split='train',
    val_frac=0.1,
    test_frac=0.1,
    split_seed=0,
    subject_col='src_subject_id',
    n_rois=100,
    seq_length=1500,
):
    # `downsample` is accepted for API parity with the UKB loader; ABCD's first
    # run uses no temporal downsampling so this flag is ignored.
    del downsample

    assert ts_path is not None and metadata_path is not None, \
        'ts_path and metadata_path are required for ABCD'

    dataset = ABCDfMRIDataset(
        ts_path=ts_path,
        metadata_path=metadata_path,
        split=split,
        n_rois=n_rois,
        seq_length=seq_length,
        val_frac=val_frac,
        test_frac=test_frac,
        split_seed=split_seed,
        subject_col=subject_col,
        use_standatdization=use_standatdization,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info(f'ABCD {split} loader ready')

    return dataset, data_loader, dist_sampler
