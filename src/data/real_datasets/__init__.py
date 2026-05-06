"""
Real Radar Dataset Loaders

Provides unified interface to load data from multiple real radar datasets:
- MAFAT Radar Challenge
- Bistatic Radar UAV
- DroneRF (coming soon)
"""

from .mafat_loader import MAFATLoader, download_mafat_instructions
from .bistatic_uav_loader import BistaticUAVLoader, download_bistatic_instructions

__all__ = [
    'MAFATLoader',
    'BistaticUAVLoader',
    'download_mafat_instructions',
    'download_bistatic_instructions',
    'get_available_datasets',
    'print_dataset_instructions'
]


def get_available_datasets() -> dict:
    """
    Get information about available real datasets.

    Returns:
        Dictionary with dataset info
    """
    return {
        'mafat': {
            'name': 'MAFAT Radar Challenge',
            'loader': MAFATLoader,
            'classes': ['Human', 'Animal'],
            'format': 'IQ matrices (32x128)',
            'samples': '~55,000',
            'access': 'Registration required',
            'url': 'https://mafatchallenge.mod.gov.il/'
        },
        'bistatic_uav': {
            'name': 'Bistatic Radar UAV RD Dataset',
            'loader': BistaticUAVLoader,
            'classes': ['UAV', 'Background'],
            'format': 'Range-Doppler images',
            'samples': '~540',
            'access': 'IEEE DataPort subscription',
            'url': 'https://ieee-dataport.org/documents/bistatic-radar-uav-target-rd-dataset'
        },
        'dronerf': {
            'name': 'DroneRF Dataset',
            'loader': None,  # Coming soon
            'classes': ['Multiple drone types', 'Background'],
            'format': 'RF I/Q signals',
            'samples': 'Variable',
            'access': 'Contact authors',
            'url': 'https://www.sciencedirect.com/science/article/pii/S2352340919306675'
        }
    }


def print_dataset_instructions(dataset: str = 'all'):
    """
    Print download instructions for specified dataset.

    Args:
        dataset: 'mafat', 'bistatic_uav', 'dronerf', or 'all'
    """
    if dataset in ['mafat', 'all']:
        download_mafat_instructions()

    if dataset in ['bistatic_uav', 'all']:
        download_bistatic_instructions()

    if dataset in ['dronerf', 'all']:
        print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                      DRONERF DATASET                                ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  Paper: "DroneRF dataset: A dataset of drones for RF-based          ║
    ║         detection, classification and identification"               ║
    ║                                                                      ║
    ║  To obtain:                                                          ║
    ║  1. Read the paper at ScienceDirect                                 ║
    ║  2. Contact authors for data access                                 ║
    ║  3. Check IEEE DataPort for alternative versions                    ║
    ║                                                                      ║
    ║  Paper URL: https://doi.org/10.1016/j.dib.2019.104313               ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
        """)
