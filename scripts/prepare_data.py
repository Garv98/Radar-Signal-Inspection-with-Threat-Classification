#!/usr/bin/env python3
"""
Data Preparation Script

This script handles:
1. Validating real radar dataset availability
2. Verifying required file structure and basic data contracts
3. Printing actionable setup guidance for missing datasets

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --dataset mafat
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.real_datasets import MAFATLoader, BistaticUAVLoader, print_dataset_instructions


def validate_mafat_dataset(data_root: Path, max_samples: int) -> dict:
    """Validate MAFAT directory structure and load a sample batch."""
    mafat_dir = data_root / 'mafat'
    required_files = [
        mafat_dir / 'train' / 'MAFAT_RADAR_Train_Segments.pkl',
        mafat_dir / 'train' / 'MAFAT_RADAR_Train_Metadata.csv'
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        return {
            'ok': False,
            'dataset': 'mafat',
            'message': f"Missing required files: {missing}"
        }

    loader = MAFATLoader(str(mafat_dir))
    loaded = loader.load_split('train', max_samples=max_samples)

    iq = np.asarray(loaded['iq_matrices'])
    labels = np.asarray(loaded['labels'])

    if iq.ndim != 3 or iq.shape[1:] != (32, 128):
        return {
            'ok': False,
            'dataset': 'mafat',
            'message': f"Unexpected IQ shape: {iq.shape}. Expected [N, 32, 128]."
        }

    labeled_count = int(np.sum(labels >= 0))
    return {
        'ok': True,
        'dataset': 'mafat',
        'samples_checked': int(iq.shape[0]),
        'labeled_samples': labeled_count,
        'message': 'MAFAT data contract is valid.'
    }


def validate_bistatic_dataset(data_root: Path, max_samples: int) -> dict:
    """Validate Bistatic UAV structure and load a sample batch."""
    bistatic_dir = data_root / 'bistatic_uav'
    images_dir = bistatic_dir / 'trainval' / 'images'

    if not images_dir.exists():
        return {
            'ok': False,
            'dataset': 'bistatic_uav',
            'message': f"Missing images directory: {images_dir}"
        }

    if not list(images_dir.glob('*.png')):
        return {
            'ok': False,
            'dataset': 'bistatic_uav',
            'message': f"No PNG files found in {images_dir}"
        }

    loader = BistaticUAVLoader(str(bistatic_dir))
    loaded = loader.load_split('trainval', max_samples=max_samples)

    images = np.asarray(loaded['images'])
    labels = np.asarray(loaded['labels'])
    if images.ndim != 3:
        return {
            'ok': False,
            'dataset': 'bistatic_uav',
            'message': f"Unexpected image tensor shape: {images.shape}. Expected [N, H, W]."
        }

    return {
        'ok': True,
        'dataset': 'bistatic_uav',
        'samples_checked': int(images.shape[0]),
        'unique_labels': sorted(np.unique(labels).tolist()),
        'message': 'Bistatic UAV data contract is valid.'
    }


def main():
    parser = argparse.ArgumentParser(description='Validate real dataset readiness')
    parser.add_argument('--source', type=str, choices=['real'], default='real',
                        help='Data source policy (real-only)')
    parser.add_argument('--dataset', type=str, choices=['mafat', 'bistatic_uav', 'all'],
                        default='all', help='Dataset to validate')
    parser.add_argument('--data-root', type=str, default='data/real',
                        help='Root directory for real datasets')
    parser.add_argument('--max-samples', type=int, default=32,
                        help='Number of samples to load per dataset for validation')
    parser.add_argument('--help-data', action='store_true',
                        help='Show download instructions for real datasets')

    args = parser.parse_args()

    if args.help_data:
        print_dataset_instructions('all')
        return

    print("=" * 60)
    print("Radar Real-Data Readiness Check")
    print("=" * 60)
    print(f"Source policy: {args.source}")
    print(f"Dataset selection: {args.dataset}")

    data_root = Path(args.data_root)
    validators = {
        'mafat': validate_mafat_dataset,
        'bistatic_uav': validate_bistatic_dataset
    }
    selected = list(validators.keys()) if args.dataset == 'all' else [args.dataset]

    results = []
    for dataset_name in selected:
        print(f"\nValidating {dataset_name}...")
        result = validators[dataset_name](data_root, args.max_samples)
        results.append(result)
        status = "OK" if result['ok'] else "FAIL"
        print(f"[{status}] {result['message']}")

    ok_count = sum(1 for r in results if r['ok'])
    print("\n" + "=" * 60)
    print(f"Validation Summary: {ok_count}/{len(results)} dataset checks passed")

    for result in results:
        print(f"- {result['dataset']}: {'PASS' if result['ok'] else 'FAIL'}")

    if ok_count != len(results):
        print("\nSome real datasets are not ready. Run with --help-data for setup instructions.")
        raise SystemExit(1)

    print("All selected real datasets are ready for training.")
    print("=" * 60)


if __name__ == '__main__':
    main()
