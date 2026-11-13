# Data Preparation

## 1. Download nuPlan Data

Download the nuPlan **test** split with [NuPlan-Download-CLI](https://github.com/Syzygianinfern0/NuPlan-Download-CLI).

## 2. Arrange the Directory

Arrange the downloaded nuPlan data exactly as follows:

```text
nuplan/
├── dataset/
│   ├── data/
│   │   └── cache/
│   │       └── test/
│   ├── maps/
│   │   ├── .maplocks/
│   │   ├── nuplan-maps-v1.0.json
│   │   ├── sg-one-north/
│   │   │   └── 9.17.1964/
│   │   ├── us-ma-boston/
│   │   │   └── 9.12.1817/
│   │   ├── us-nv-las-vegas-strip/
│   │   │   └── 9.15.1915/
│   │   └── us-pa-pittsburgh-hazelwood/
│   │       └── 9.17.1937/
│   ├── nuplan-v1.1/
│   │   ├── sensor_blobs/
│   │   │   └── <scene_token>/
│   │   │       ├── CAM_B0/
│   │   │       ├── CAM_F0/
│   │   │       ├── CAM_L0~L2/
│   │   │       ├── CAM_R0~R2/
│   │   │       └── MergedPointCloud/
│   │   └── splits/
│   │       └── test/
```

## 3. Prepare NAVSIM Data

Prepare the NAVSIM data (test split) following the [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).

Set the following environment variables:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```
