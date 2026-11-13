# Model Testing

Run every command from the repository root. `path/to/rendered_data` is the
package produced in [Test Case Generation](test_case_generation.md).

## 1. Train-Test Split Generation

```bash
python nuplan_scripts/auto_reconstruction/build_navsim_package_split.py \
    --package-root path/to/rendered_data \
    --split-name navhazard \
    --overwrite
```

## 2. Metric Caching

```bash
PACKAGE_ROOT=path/to/rendered_data bash scripts/metric_cache.sh
```

## 3. Testing E2EAD Models

```bash
# Constant velocity (no checkpoint)
PACKAGE_ROOT=path/to/rendered_data bash scripts/test_cv.sh

# Latent TransFuser
PACKAGE_ROOT=path/to/rendered_data CHECKPOINT_PATH=path/to/ltf_seed_0.ckpt bash scripts/test_ltf.sh

# Ego-status MLP
PACKAGE_ROOT=path/to/rendered_data CHECKPOINT_PATH=path/to/ego_status_mlp_seed_0.ckpt bash scripts/test_mlp.sh
```
