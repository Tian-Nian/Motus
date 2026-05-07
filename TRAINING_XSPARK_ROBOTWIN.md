# Motus RobotWin Training Flow (xspark_shared)

This setup assumes:

- Conda environment: `motus`
- Pretrained weights root: `/vepfs-cnbje63de6fae220/xspark_shared/model_weights`
- Raw RobotWin data root: `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/raw`
- Converted Motus-format data root: `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/motus_processed`

## 1. Activate environment

```bash
conda activate motus
cd /vepfs-cnbje63de6fae220/niantian/Motus
```

## 2. Convert RobotWin raw data to Motus format

Training does not read the raw `robotwin_data/raw` tree directly. It expects the converted structure:

```text
motus_processed/
  clean/<task>/videos
  clean/<task>/qpos
  clean/<task>/umt5_wan
  randomized/<task>/videos
  randomized/<task>/qpos
  randomized/<task>/umt5_wan
```

The dataset loader requires `umt5_wan`, so T5 embedding generation must stay enabled during conversion.

```bash
cd /vepfs-cnbje63de6fae220/niantian/Motus/data/robotwin2/robotwin_data_convert
cp config_xspark.yml config.yml
bash run_conversion.sh
```

## 3. Verify converted data

At minimum, make sure the converted root contains both `clean` and/or `randomized`, and each task has `videos`, `qpos`, and `umt5_wan`.

## 4. Start fine-tuning

The training config is already wired to:

- WAN: `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B`
- VLM: `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-2B-Instruct`
- Motus stage-2 finetune init: `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Motus`
- Dataset: `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/motus_processed`

Run:

```bash
cd /vepfs-cnbje63de6fae220/niantian/Motus
bash scripts/train_robotwin_xspark.sh
```

## 5. Common overrides

Use environment variables when needed:

```bash
NPROC_PER_NODE=4 \
MASTER_PORT=29501 \
RUN_NAME=robotwin_xspark_debug \
CHECKPOINT_DIR=/vepfs-cnbje63de6fae220/niantian/Motus/checkpoints_xspark \
bash scripts/train_robotwin_xspark.sh
```

## 6. Resume training

Edit `configs/robotwin_xspark.yaml` and set:

```yaml
resume:
  checkpoint_path: /path/to/checkpoint_step_xxx
```

Then rerun the same launch script.