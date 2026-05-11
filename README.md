# Meta-CXR-GCP

Migration của `META_CXR_kaggle.ipynb` (paper IEEE Access 2025) sang **GCP Compute Engine 2× T4 spot**, dùng **GCS bucket `gs://mimic-cxr-jpg-data/`** làm data source và checkpoint store.

## TL;DR

```bash
cp .env.example .env  # điền WANDB_API_KEY và xác nhận project/bucket
./scripts/01_setup_gcp.sh        # enable APIs, service account, IAM
./scripts/02_build_docker.sh     # build & push image lên Artifact Registry
./scripts/04_train.sh            # train smoke: tạo VM → mount GCS → papermill → upload → stop VM
```

Resume: `./scripts/04_resume_training.sh` (auto-pick `checkpoint_best.pth` trên GCS).

## Architecture

- **Code** self-contained trong repo (model/, mhcac/, biovil_t/, vision_encoders/, utils/, pretraining/)
- **Data** trên GCS, mount bằng gcsfuse vào `/mnt/gcs-data/`
- **Checkpoints** lưu `gs://mimic-cxr-jpg-data/checkpoints/<run_id>/`; `checkpoint_best.pth` không bao giờ xoá
- **Auth** ADC via metadata server trên VM (không cần JSON key)
- **Logging** W&B; executed notebook upload `gs://.../logs/`
- **Notebook execution** papermill, parametrize qua `configs/training.yaml`

## Project structure

```
.
├── notebooks/META_CXR_gcp.ipynb     # adapted từ META_CXR_kaggle.ipynb (chỉ sửa 7 cells)
├── configs/{gcp,training,data,checkpoint}.yaml
├── scripts/01..08_*.sh, vm_startup.sh, utils/
├── src/{config_loader,gcs_checkpoint,papermill_runner}.py
├── tests/
├── pretraining/, model/, mhcac/, biovil_t/, vision_encoders/, utils/   (copy từ repo gốc)
├── Dockerfile, requirements.txt
└── SETUP_GUIDE.md, COST_ESTIMATE.md, CLAUDE.md
```

## Docs

- `SETUP_GUIDE.md` — end-to-end VN (upgrade Free Trial, request T4 quota, run smoke test, resume)
- `COST_ESTIMATE.md` — 3 scenarios (smoke / full train / worst-case)
- `CLAUDE.md` — behavioral guidelines

## Constraints

- Free Trial $280, hết hạn 26/06/2026 — KHÔNG dùng cho production
- Region `us-central1` only
- 2× T4 spot, fallback zone `-a` → `-b` → `-c` → `-f`
- VM auto-stop sau khi notebook chạy xong (kể cả error)
