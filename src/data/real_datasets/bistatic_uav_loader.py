"""
Bistatic Radar UAV Dataset Loader

Loads real bistatic radar Range-Doppler images for UAV detection.
Dataset contains PNG images with binary segmentation masks.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BistaticUAVLoader:
    """
    Loader for Bistatic Radar UAV Target RD Dataset.

    Dataset structure:
    - trainval/images/*.png - Range-Doppler images
    - trainval/masks/*.png - Binary segmentation masks
    - test/images/*.png
    - test/masks/*.png

    Maps to our classes:
    - UAV present (mask > 0) -> Drone
    - Background only -> Clutter
    """

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
        image_size: Tuple[int, int] = (128, 128)
    ):
        """
        Initialize BistaticUAV loader.

        Args:
            data_dir: Path to bistatic UAV data directory
            image_size: Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        logger.info(f"Initialized BistaticUAVLoader with data_dir={data_dir}")

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess a single image."""
        img = Image.open(path).convert('L')  # Grayscale
        img = img.resize(self.image_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load a binary mask."""
        mask = Image.open(path).convert('L')
        mask = mask.resize(self.image_size, Image.NEAREST)
        return np.array(mask, dtype=np.float32) / 255.0

    def _has_uav(self, mask: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if mask contains UAV (non-zero region)."""
        return np.mean(mask > 0.5) > threshold

    def load_split(
        self,
        split: str = 'trainval',
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a data split.

        Args:
            split: 'trainval' or 'test'
            max_samples: Optional limit on samples

        Returns:
            Dictionary with 'images', 'masks', 'labels', 'filenames'
        """
        split_dir = self.data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Get image files
        image_files = sorted(images_dir.glob('*.png'))
        if max_samples:
            image_files = image_files[:max_samples]

        images = []
        masks = []
        labels = []
        filenames = []

        for img_path in image_files:
            # Load image
            img = self._load_image(img_path)
            images.append(img)
            filenames.append(img_path.name)

            # Try to load corresponding mask
            mask_name = img_path.name.replace('rd_', 'label_')
            mask_path = masks_dir / mask_name

            if mask_path.exists():
                mask = self._load_mask(mask_path)
                masks.append(mask)

                # Determine label based on mask content
                if self._has_uav(mask):
                    labels.append(self.CLASS_TO_IDX['Drone'])
                else:
                    labels.append(self.CLASS_TO_IDX['Clutter'])
            else:
                masks.append(np.zeros(self.image_size, dtype=np.float32))
                labels.append(self.CLASS_TO_IDX['Clutter'])

        images = np.array(images)
        masks = np.array(masks)
        labels = np.array(labels)

        logger.info(f"Loaded {len(images)} samples from {split}")
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"UAV samples: {np.sum(labels == 0)}, Background: {np.sum(labels == 3)}")

        return {
            'images': images,
            'masks': masks,
            'labels': labels,
            'filenames': filenames
        }

    def load_all(self) -> Dict[str, Any]:
        """Load all available data."""
        all_data = {
            'images': [],
            'masks': [],
            'labels': [],
            'filenames': [],
            'split_ids': []
        }

        for split in ['trainval', 'test']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                data = self.load_split(split)
                all_data['images'].append(data['images'])
                all_data['masks'].append(data['masks'])
                all_data['labels'].append(data['labels'])
                all_data['filenames'].extend(data['filenames'])
                all_data['split_ids'].extend([split] * len(data['labels']))

        if all_data['images']:
            all_data['images'] = np.concatenate(all_data['images'])
            all_data['masks'] = np.concatenate(all_data['masks'])
            all_data['labels'] = np.concatenate(all_data['labels'])

        return all_data

    def convert_to_iq_format(
        self,
        images: np.ndarray,
        num_pulses: int = 32,
        num_samples: int = 128
    ) -> np.ndarray:
        """
        Convert RD images to pseudo-IQ format for model compatibility.

        This creates a synthetic IQ representation from the RD images
        to work with our CNN-LSTM model designed for IQ inputs.

        Args:
            images: RD images [N, H, W]
            num_pulses: Target number of pulses (slow time)
            num_samples: Target number of range samples

        Returns:
            Pseudo-IQ matrices [N, num_pulses, num_samples] as complex
        """
        from scipy.ndimage import zoom

        n_samples = len(images)
        iq_matrices = np.zeros((n_samples, num_pulses, num_samples), dtype=np.complex64)

        for i, img in enumerate(images):
            # Resize to target dimensions
            if img.shape != (num_pulses, num_samples):
                zoom_factors = (num_pulses / img.shape[0], num_samples / img.shape[1])
                img_resized = zoom(img, zoom_factors, order=1)
            else:
                img_resized = img

            # Convert magnitude image to pseudo-IQ
            # Assume random phase (since we only have magnitude)
            phase = np.random.uniform(-np.pi, np.pi, img_resized.shape)
            iq_matrices[i] = img_resized * np.exp(1j * phase)

        return iq_matrices

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'data_dir': str(self.data_dir),
            'available_splits': [],
            'total_samples': 0
        }

        for split in ['trainval', 'test']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                stats['available_splits'].append(split)
                images_dir = split_dir / 'images'
                if images_dir.exists():
                    n_images = len(list(images_dir.glob('*.png')))
                    stats[f'{split}_samples'] = n_images
                    stats['total_samples'] += n_images

        return stats


def download_bistatic_instructions():
    """Print instructions for downloading Bistatic UAV dataset."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              BISTATIC RADAR UAV TARGET RD DATASET                   ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  This dataset requires IEEE DataPort subscription.                  ║
    ║                                                                      ║
    ║  Steps to obtain the data:                                          ║
    ║                                                                      ║
    ║  1. Go to IEEE DataPort:                                            ║
    ║     https://ieee-dataport.org/documents/bistatic-radar-uav-target-rd-dataset ║
    ║                                                                      ║
    ║  2. Login with IEEE account or create one                           ║
    ║                                                                      ║
    ║  3. Subscribe to IEEE DataPort (check if institutional access)      ║
    ║                                                                      ║
    ║  4. Download: image.zip (201.4 MB)                                  ║
    ║                                                                      ║
    ║  5. Extract to: data/real/bistatic_uav/                             ║
    ║     Structure should be:                                            ║
    ║     - data/real/bistatic_uav/trainval/images/*.png                  ║
    ║     - data/real/bistatic_uav/trainval/masks/*.png                   ║
    ║     - data/real/bistatic_uav/test/images/*.png                      ║
    ║     - data/real/bistatic_uav/test/masks/*.png                       ║
    ║                                                                      ║
    ║  DOI: 10.21227/81p6-te37                                            ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)
    return instructions
