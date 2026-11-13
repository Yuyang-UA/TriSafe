# Test Case Generation

Run every command from the repository root.

## 1. Motion Dictionary

- **Option 1:** Use the default structure in `trajectory_candidate/dictionary_structure/`.
- **Option 2:** Rebuild the structure with your own settings:

```bash
# Determine the range of the coarse group number
python trajectory_candidate/kmeans_group_number_confirmation.py

# Build the hierarchical dictionary
python trajectory_candidate/build_hierarchical_dictionary.py
```

## 2. Hierarchical Greedy Search

### 2.1 Explore Different Weight Combinations

```bash
bash scripts/weight_combination_comparison.sh
```

### 2.2 Search with One Weight Combination

```bash
bash scripts/greedy_search.sh
```

## 3. Scene Reconstruction

Scene reconstruction follows [MTGS](https://github.com/OpenDriveLab/MTGS). The
reconstructed data is not shipped with this repository. Steps 4 and 6 read it
from the `data_root` of each config in `data/auto_reconstruction/config/`.

### 3.1 Environment Setting

```bash
source scripts/dev.sh
mtgs_setup mtgs/config/MTGS.py
```

### 3.2 Generate Configs from the NAVSIM Token List

```bash
python -m nuplan_scripts.auto_reconstruction.generate_configs_from_navsim_filter \
    --navsim_config navsim/planning/script/config/common/train_test_split/navtest.yaml \
    --data_root path/to/reconstruction_data \
    --output_dir path/to/config_dir \
    --split test \
    --num_workers 32
```

### 3.3 Preprocessing and Training

```bash
bash scripts/run_construction.sh
```

> **Note:** `scripts/run_construction.sh` runs `scripts/construction_batch_1.sh`, `scripts/construction_batch_2.sh`, `scripts/construction_batch_3.sh` and `scripts/construction_batch_4.sh` in sequence. You can also run them one by one in this order.

## 4. Risky Scene Construction

```bash
bash scripts/risky_case_construction.sh
```

## 5. Solvability Verification

```bash
bash scripts/solvability_verification.sh
```

## 6. Multi-View Sensor Data Rendering

```bash
bash scripts/sensor_data_rendering.sh
```

## 7. Driving Command Generation

```bash
RENDERED_DATA_ROOT=path/to/rendered_data bash scripts/generate_driving_command.sh
```
