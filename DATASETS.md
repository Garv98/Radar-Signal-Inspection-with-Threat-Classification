# Real Radar Datasets for Threat Classification

This document provides comprehensive information about publicly available radar datasets suitable for the RL-Driven Radar Signal Inspection project.

---

## 1. MAFAT Radar Challenge Dataset (RECOMMENDED)

**Access:** https://mafatchallenge.mod.gov.il/
**Registration:** Required (Israeli Ministry of Defense)

### Data Specifications
| Property | Value |
|----------|-------|
| Format | Pickle (.pkl) + CSV metadata |
| IQ Matrix | 32 x 128 (pulses x samples) |
| Data Type | Complex I/Q values |
| Training Samples | 6,656 segments |
| Auxiliary Samples | 49,071 segments |
| Classes | Human (1), Animal (0) |
| SNR Types | High SNR, Low SNR |

### Metadata Fields
- `segment_id`: Unique identifier
- `track_id`: Track identifier
- `geolocation_type`: Environment type
- `sensor_id`: Radar sensor ID
- `snr_type`: Signal-to-noise ratio category
- `target_type`: Ground truth label

### How to Access
1. Fill application form at https://mafatchallenge.mod.gov.il/#ApplicationForm
2. Register on CodaLab (competition page)
3. Wait for approval (typically 2-5 days)
4. Download from competition "participate" tab

### Mapping to Our Classes
| MAFAT Class | Our Class Mapping |
|-------------|-------------------|
| Human | Aircraft/Drone (similar motion) |
| Animal | Bird |
| Background | Clutter |
| Low SNR samples | Noise-like |

---

## 2. Bistatic Radar UAV Dataset

**Access:** https://ieee-dataport.org/documents/bistatic-radar-uav-target-rd-dataset
**DOI:** 10.21227/81p6-te37
**Subscription:** IEEE DataPort subscription required

### Data Specifications
| Property | Value |
|----------|-------|
| Format | PNG images + binary masks |
| Type | Range-Doppler (RD) images |
| Train/Val Samples | ~460 images |
| Test Samples | ~81 images |
| Total Size | 201.4 MB |
| Task | Binary UAV detection |

### Advantages
- Real bistatic radar data
- Low signal-to-clutter conditions
- Segmentation masks for precise localization

---

## 3. DroneRF Dataset

**Access:** https://www.sciencedirect.com/science/article/pii/S2352340919306675 (Paper)
**Download:** Contact authors or check IEEE DataPort

### Data Specifications
| Property | Value |
|----------|-------|
| Drones | DJI Phantom 3, Parrot Bebop, etc. (10+ models) |
| Format | RF I/Q signals |
| Frequency | 2.4 GHz / 5.8 GHz bands |
| Scenarios | Hovering, flying, with/without payload |
| Classes | Multiple drone types + background |

### Use Case
- Drone-specific classification
- RF fingerprinting
- Multi-drone scenario detection

---

## 4. RADDet / CARRADA Dataset

**Access:** https://github.com/valeoai/CARRADA_RADDet
**License:** Open source

### Data Specifications
| Property | Value |
|----------|-------|
| Type | Automotive radar |
| Format | Range-Doppler-Angle tensors |
| Annotations | 2D/3D bounding boxes |
| Scenarios | Urban driving |

### Relevance
- Multi-class object detection
- Real radar data with ground truth
- Good for transfer learning

---

## 5. RaDICaL Dataset

**Access:** https://publish.illinois.edu/radicaldata/
**License:** Academic use

### Data Specifications
| Property | Value |
|----------|-------|
| Radar | TI mmWave (77 GHz) |
| Format | Raw ADC data, point clouds |
| Scenes | Indoor/outdoor |
| Objects | Humans, vehicles, objects |

---

## Dataset Integration Priority

For DRDO-level implementation, recommend this acquisition order:

### Phase 1: Quick Start (1-2 days)
1. **Bistatic UAV Dataset** - Small, easy to download with IEEE subscription
2. Continue using **synthetic data** for underrepresented classes

### Phase 2: Primary Dataset (1 week)
3. **MAFAT Challenge** - Apply for access immediately (takes a few days)
4. Map MAFAT classes to our 5-class taxonomy

### Phase 3: Drone Specialization (2 weeks)
5. **DroneRF** or similar drone-specific RF dataset
6. Fine-tune model for drone vs. aircraft discrimination

---

## Data Loading Code

Once you have the datasets, use the loaders in `src/data/real_datasets/`:

```python
from src.data.real_datasets import MAFATLoader, BistaticUAVLoader

# Load MAFAT data
mafat = MAFATLoader('data/real/mafat/')
train_data = mafat.load_split('train')

# Load Bistatic UAV data
bistatic = BistaticUAVLoader('data/real/bistatic_uav/')
uav_data = bistatic.load_all()
```

---

## Manual Download Instructions

### MAFAT Dataset
```bash
# After approval, download from CodaLab
# Files: MAFAT_RADAR_TRAIN.zip, MAFAT_RADAR_AUX.zip

# Extract
unzip MAFAT_RADAR_TRAIN.zip -d data/real/mafat/train/
unzip MAFAT_RADAR_AUX.zip -d data/real/mafat/auxiliary/
```

### Bistatic UAV Dataset
```bash
# Download from IEEE DataPort (requires subscription)
# File: image.zip (201.4 MB)

# Extract
unzip image.zip -d data/real/bistatic_uav/
```

---

## Class Mapping Strategy

| Original Dataset | Original Class | Mapped Class | Confidence |
|-----------------|----------------|--------------|------------|
| MAFAT | Human | Drone/Aircraft | Medium |
| MAFAT | Animal | Bird | High |
| MAFAT | Low SNR | Noise | High |
| Bistatic UAV | UAV Target | Drone | High |
| Bistatic UAV | Background | Clutter | High |
| DroneRF | Drone RF | Drone | High |
| DroneRF | No Drone | Clutter/Noise | Medium |
| Synthetic | All 5 classes | Direct | Full control |

---

## Recommended Hybrid Approach

For the best DRDO-level model:

1. **Use MAFAT as primary dataset** (55,727 total samples)
2. **Augment with Bistatic UAV** for drone-specific patterns
3. **Generate synthetic samples** for Aircraft class (rare in public datasets)
4. **Mix synthetic Clutter/Noise** to balance classes

Final dataset composition target:
- Drone: 3,000+ samples (real + synthetic)
- Aircraft: 2,000+ samples (mostly synthetic)
- Bird: 3,000+ samples (MAFAT animals)
- Clutter: 3,000+ samples (real + synthetic)
- Noise: 2,000+ samples (real low-SNR + synthetic)
