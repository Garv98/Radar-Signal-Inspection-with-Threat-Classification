"""
MAFAT Radar Challenge Dataset Loader

Loads real radar data from the MAFAT challenge for threat classification.
Handles IQ matrices (32x128), Doppler vectors, and metadata.

Dataset format:
- Pickle files containing segment data
- CSV files containing metadata
- Classes: Human (1), Animal (0)
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MAFATLoader:
    """
    Loader for MAFAT Radar Challenge dataset.

    Maps MAFAT classes to our 5-class taxonomy:
    - Human -> Drone/Aircraft (requires additional features to distinguish)
    - Animal -> Bird
    - Background/Low SNR -> Clutter/Noise
    """

    # Class mapping from MAFAT to our taxonomy
    CLASS_MAPPING = {
        'human': 'Drone',      # Conservative: treat humans as potential threats
        'animal': 'Bird',
        'background': 'Clutter',
        'low_snr': 'Noise'
    }

    # Our class indices
    CLASS_TO_IDX = {
        'Drone': 0,
        'Aircraft': 1,
        'Bird': 2,
        'Clutter': 3,
        'Noise': 4
    }

    def __init__(
        self,
        data_dir: str,
        map_humans_to: str = 'Drone',
        snr_threshold: float = 0.3
    ):
        """
        Initialize MAFAT loader.

        Args:
            data_dir: Path to MAFAT data directory
            map_humans_to: Map human class to 'Drone' or 'Aircraft'
            snr_threshold: Threshold for categorizing low SNR as Noise
        """
        self.data_dir = Path(data_dir)
        self.map_humans_to = map_humans_to
        self.snr_threshold = snr_threshold

        # Update class mapping
        self.CLASS_MAPPING['human'] = map_humans_to

        logger.info(f"Initialized MAFATLoader with data_dir={data_dir}")

    def _verify_data_structure(self) -> bool:
        """Verify expected data structure exists."""
        expected_files = ['MAFAT_RADAR_Train_Segments.pkl', 'MAFAT_RADAR_Train_Metadata.csv']

        for f in expected_files:
            if not (self.data_dir / 'train' / f).exists():
                logger.warning(f"Expected file not found: {f}")
                return False
        return True

    def load_split(
        self,
        split: str = 'train',
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a data split.

        Args:
            split: 'train', 'auxiliary', or 'test'
            max_samples: Optional limit on samples to load

        Returns:
            Dictionary with 'iq_matrices', 'doppler', 'labels', 'metadata'
        """
        split_dir = self.data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Load segments pickle
        segments_file = list(split_dir.glob('*Segments*.pkl'))
        if not segments_file:
            raise FileNotFoundError(f"No segments file found in {split_dir}")

        logger.info(f"Loading segments from {segments_file[0]}")
        with open(segments_file[0], 'rb') as f:
            segments = pickle.load(f)

        # Load metadata CSV
        metadata_file = list(split_dir.glob('*Metadata*.csv'))
        if metadata_file:
            logger.info(f"Loading metadata from {metadata_file[0]}")
            metadata = pd.read_csv(metadata_file[0])
        else:
            metadata = None
            logger.warning("No metadata file found")

        # Process segments
        iq_matrices = []
        doppler_vectors = []
        labels = []
        segment_ids = []

        for i, seg in enumerate(segments):
            if max_samples and i >= max_samples:
                break

            iq_matrices.append(seg['iq_sweep_burst'])
            doppler_vectors.append(seg.get('doppler_burst', np.zeros(32)))
            segment_ids.append(seg['segment_id'])

            # Get label from metadata if available
            if metadata is not None and 'target_type' in metadata.columns:
                seg_meta = metadata[metadata['segment_id'] == seg['segment_id']]
                if len(seg_meta) > 0:
                    target_type = seg_meta['target_type'].values[0]
                    snr_type = seg_meta.get('snr_type', pd.Series(['high'])).values[0]

                    # Map to our classes
                    if snr_type == 'low' and np.random.random() < self.snr_threshold:
                        mapped_class = 'Noise'
                    elif target_type == 'human':
                        mapped_class = self.map_humans_to
                    elif target_type == 'animal':
                        mapped_class = 'Bird'
                    else:
                        mapped_class = 'Clutter'

                    labels.append(self.CLASS_TO_IDX[mapped_class])
                else:
                    labels.append(-1)  # Unknown
            else:
                labels.append(-1)

        # Convert to numpy arrays
        iq_matrices = np.array(iq_matrices)
        doppler_vectors = np.array(doppler_vectors)
        labels = np.array(labels)

        logger.info(f"Loaded {len(iq_matrices)} samples from {split}")
        logger.info(f"IQ matrix shape: {iq_matrices.shape}")
        logger.info(f"Label distribution: {np.bincount(labels[labels >= 0])}")

        return {
            'iq_matrices': iq_matrices,
            'doppler': doppler_vectors,
            'labels': labels,
            'segment_ids': segment_ids,
            'metadata': metadata
        }

    def load_all(self, include_auxiliary: bool = True) -> Dict[str, Any]:
        """
        Load all available data.

        Args:
            include_auxiliary: Whether to include auxiliary set

        Returns:
            Combined data dictionary
        """
        all_data = {
            'iq_matrices': [],
            'doppler': [],
            'labels': [],
            'segment_ids': [],
            'split_ids': []
        }

        # Load training data
        if (self.data_dir / 'train').exists():
            train_data = self.load_split('train')
            all_data['iq_matrices'].append(train_data['iq_matrices'])
            all_data['doppler'].append(train_data['doppler'])
            all_data['labels'].append(train_data['labels'])
            all_data['segment_ids'].extend(train_data['segment_ids'])
            all_data['split_ids'].extend(['train'] * len(train_data['labels']))

        # Load auxiliary data
        if include_auxiliary and (self.data_dir / 'auxiliary').exists():
            aux_data = self.load_split('auxiliary')
            all_data['iq_matrices'].append(aux_data['iq_matrices'])
            all_data['doppler'].append(aux_data['doppler'])
            all_data['labels'].append(aux_data['labels'])
            all_data['segment_ids'].extend(aux_data['segment_ids'])
            all_data['split_ids'].extend(['auxiliary'] * len(aux_data['labels']))

        # Concatenate
        all_data['iq_matrices'] = np.concatenate(all_data['iq_matrices'])
        all_data['doppler'] = np.concatenate(all_data['doppler'])
        all_data['labels'] = np.concatenate(all_data['labels'])

        return all_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'data_dir': str(self.data_dir),
            'available_splits': [],
            'total_samples': 0
        }

        for split_name in ['train', 'auxiliary', 'test']:
            split_dir = self.data_dir / split_name
            if split_dir.exists():
                stats['available_splits'].append(split_name)

                # Count samples
                segments_file = list(split_dir.glob('*Segments*.pkl'))
                if segments_file:
                    with open(segments_file[0], 'rb') as f:
                        segments = pickle.load(f)
                    stats[f'{split_name}_samples'] = len(segments)
                    stats['total_samples'] += len(segments)

        return stats


def download_mafat_instructions():
    """Print instructions for downloading MAFAT dataset."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                 MAFAT RADAR CHALLENGE DATASET                       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  This dataset requires manual download due to licensing.            ║
    ║                                                                      ║
    ║  Steps to obtain the data:                                          ║
    ║                                                                      ║
    ║  1. Go to: https://mafatchallenge.mod.gov.il/#ApplicationForm       ║
    ║                                                                      ║
    ║  2. Fill out the application form with:                             ║
    ║     - Your institutional affiliation                                 ║
    ║     - Research purpose (academic/defense research)                   ║
    ║     - Intended use case                                              ║
    ║                                                                      ║
    ║  3. Wait for approval (typically 2-5 business days)                 ║
    ║                                                                      ║
    ║  4. Once approved, register on CodaLab:                             ║
    ║     https://competitions.codalab.org/competitions/25389             ║
    ║                                                                      ║
    ║  5. Download from the "Participate" tab:                            ║
    ║     - MAFAT_RADAR_Train_Segments.pkl                                ║
    ║     - MAFAT_RADAR_Train_Metadata.csv                                ║
    ║     - MAFAT_RADAR_Aux_Segments.pkl                                  ║
    ║     - MAFAT_RADAR_Aux_Metadata.csv                                  ║
    ║                                                                      ║
    ║  6. Extract to: data/real/mafat/train/ and data/real/mafat/auxiliary/║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)
    return instructions
