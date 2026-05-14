"""Compute Brain Gradient Positioning from fMRI time-series data.

Implements the procedure described in Brain-JEPA (NeurIPS 2024) Section 3.1:

  1. Compute functional connectivity (Pearson correlation across ROIs) per
     subject, then average across subjects to form a group-level FC C.
  2. Treat each ROI's row of C as its feature vector c_i and form the
     non-negative angular affinity matrix
            A(i,j) = 1 - (1/pi) * arccos( <c_i,c_j> / (||c_i|| * ||c_j||) )
  3. Diffusion-map normalization (alpha = 0.5, BrainSpace convention):
            L      = D^{-alpha} A D^{-alpha}
            M      = D_L^{-1} L                   (row-stochastic)
            M_sym  = D_L^{-1/2} L D_L^{-1/2}      (symmetric, same eigenvalues)
  4. Eigendecompose M_sym, drop the trivial constant eigenvector, keep the
     top-m components -> gradient matrix G of shape (n_rois, m).
  5. Save G as a CSV consumable by Brain-JEPA's gradient_pos_embed input.

Input data layouts supported (matches src/datasets/abcd.py):
    - A single .npy file of shape (N, R, T)
    - A directory containing one .npy per subject of shape (R, T)

Usage:
    python scripts/compute_gradient.py \\
        --data /path/to/abcd.npy \\
        --out data/gradient_mapping_100.csv \\
        --n_components 30
"""

import argparse
import glob
import os
import sys

import numpy as np


def _iter_subjects(data_path, max_subjects=None, indices=None):
    if os.path.isfile(data_path) and data_path.endswith('.npy'):
        arr = np.load(data_path, mmap_mode='r')
        assert arr.ndim == 3, f'expected (N, R, T), got {arr.shape}'
        if indices is None:
            indices = np.arange(arr.shape[0])
        if max_subjects is not None:
            indices = indices[:max_subjects]
        for i in indices:
            yield np.asarray(arr[int(i)], dtype=np.float64)
    elif os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
        if indices is not None:
            files = [files[int(i)] for i in indices]
        if max_subjects is not None:
            files = files[:max_subjects]
        assert files, f'no .npy files in {data_path}'
        for f in files:
            ts = np.load(f).astype(np.float64)
            assert ts.ndim == 2, f'{f}: expected (R, T), got {ts.shape}'
            yield ts
    else:
        raise FileNotFoundError(data_path)


def group_fc(data_path, max_subjects=None, indices=None, verbose=True):
    acc = None
    n = 0
    for ts in _iter_subjects(data_path, max_subjects, indices=indices):
        fc = np.corrcoef(ts)
        np.nan_to_num(fc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if acc is None:
            acc = np.zeros_like(fc)
        acc += fc
        n += 1
        if verbose and n % 200 == 0:
            print(f'  accumulated FC over {n} subjects', file=sys.stderr)
    assert n > 0, 'no subjects loaded'
    if verbose:
        print(f'group FC over {n} subjects, shape {acc.shape}', file=sys.stderr)
    return acc / n


def angular_affinity(C):
    # Cosine similarity over ROI feature vectors (rows of FC).
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    sim = (C @ C.T) / (norms @ norms.T)
    sim = np.clip(sim, -1.0, 1.0)
    A = 1.0 - np.arccos(sim) / np.pi  # in [0, 1]
    np.fill_diagonal(A, 0.0)
    return A


def diffusion_embedding(A, n_components=30, alpha=0.5):
    d = A.sum(axis=1)
    d = np.where(d == 0, 1e-12, d)
    d_inv_alpha = 1.0 / np.power(d, alpha)
    L = d_inv_alpha[:, None] * A * d_inv_alpha[None, :]

    dL = L.sum(axis=1)
    dL = np.where(dL == 0, 1e-12, dL)
    dL_sqrt = np.sqrt(dL)

    # Symmetric form shares eigenvalues with the row-stochastic M = D_L^{-1} L;
    # eigenvectors of M are recovered as psi_k = v_k / dL_sqrt.
    M_sym = L / (dL_sqrt[:, None] * dL_sqrt[None, :])
    M_sym = 0.5 * (M_sym + M_sym.T)  # enforce symmetry against fp drift

    evals, evecs = np.linalg.eigh(M_sym)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    psi = evecs / dL_sqrt[:, None]
    # Drop the trivial first eigenvector (constant, eigenvalue ~= 1).
    psi = psi[:, 1:n_components + 1]
    evals = evals[1:n_components + 1]
    return psi, evals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True,
                    help='Path to .npy (N,R,T) file or directory of (R,T) .npy files.')
    ap.add_argument('--out', default='data/gradient_mapping_100.csv',
                    help='Output CSV path (no header, no index).')
    ap.add_argument('--n_components', type=int, default=30,
                    help='Number of gradient axes m to keep (paper uses 30).')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='Diffusion-map normalization parameter (paper: 0.5).')
    ap.add_argument('--max_subjects', type=int, default=None,
                    help='Optional cap on number of subjects used.')
    # Optional: restrict to training subjects (no leakage from val/test).
    ap.add_argument('--metadata', default=None,
                    help='Metadata CSV; if given, gradient is computed only on '
                         'rows whose subject falls in --split.')
    ap.add_argument('--split', default='train', choices=('train', 'val', 'test'),
                    help='Which split to use when --metadata is supplied.')
    ap.add_argument('--subject_col', default='src_subject_id')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_frac', type=float, default=0.1)
    ap.add_argument('--split_seed', type=int, default=0)
    args = ap.parse_args()

    indices = None
    if args.metadata is not None:
        # Import here so the script remains usable without the project on PYTHONPATH.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.datasets.abcd import subject_split_indices
        splits = subject_split_indices(
            args.metadata,
            subject_col=args.subject_col,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.split_seed,
        )
        indices = splits[args.split]
        print(f'Restricting to split={args.split!r}: {len(indices)} rows',
              file=sys.stderr)

    print(f'Computing group-mean functional connectivity from {args.data} ...',
          file=sys.stderr)
    C = group_fc(args.data, args.max_subjects, indices=indices)

    print('Computing angular affinity ...', file=sys.stderr)
    A = angular_affinity(C)

    print(f'Computing diffusion embedding (m={args.n_components}, alpha={args.alpha}) ...',
          file=sys.stderr)
    G, evals = diffusion_embedding(A, n_components=args.n_components, alpha=args.alpha)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savetxt(args.out, G, delimiter=',', fmt='%.18e')
    print(f'Wrote gradient matrix {G.shape} -> {args.out}', file=sys.stderr)
    print(f'Top eigenvalues: {evals[:5]}', file=sys.stderr)


if __name__ == '__main__':
    main()
