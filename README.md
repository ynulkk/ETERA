# ETERA
# Run Flow

Use `configs/SEED.yaml` for SEED and `configs/SEED_IV.yaml` for SEED-IV.

## 1. Preprocess

```bash
python preprocess_eeg_datasets.py \
  --dataset SEED_IV \
  --num_classes 4 \
  --num_trials 24 \
  --eeg_raw_data_path <RAW_SEED_IV_DIR> \
  --eeg_datasets_path <PROCESSED_SEED_IV_DIR>
```

For SEED, use `--dataset SEED --num_classes 3 --num_trials 15`.

## 2. Pretrain EEG-TopoEncoder

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pretrain_EEG_TopoEncoder.py \
  --config configs/SEED_IV.yaml
```

## 3. Train EEG-TopoEncoder

Cross-session:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_cross_session.py \
  --config configs/SEED_IV.yaml \
  --method EEG_TopoEncoder \
  --subject <SUBJECT_ID> \
  --train-session <TRAIN_SESSION> \
  --test-session <TEST_SESSION> \
  --manifest <MANIFEST_JSON>
```

Cross-subject:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_cross_subject.py \
  --config configs/SEED_IV.yaml \
  --method EEG_TopoEncoder \
  --session <SESSION_ID> \
  --test-subject <SUBJECT_ID>
```

## 4. Train ETEA-CGRA

Run this after the matching EEG-TopoEncoder baseline checkpoints are available.

Cross-session:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_cross_session.py \
  --config configs/SEED_IV.yaml \
  --method ETEA_CGRA \
  --subject <SUBJECT_ID> \
  --train-session <TRAIN_SESSION> \
  --test-session <TEST_SESSION> \
  --manifest <MANIFEST_JSON>
```

Cross-subject:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_cross_subject.py \
  --config configs/SEED_IV.yaml \
  --method ETEA_CGRA \
  --session <SESSION_ID> \
  --test-subject <SUBJECT_ID>
```

For SEED, replace `configs/SEED_IV.yaml` with `configs/SEED.yaml`.
