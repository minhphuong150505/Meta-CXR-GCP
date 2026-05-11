# Plan: Migrate META-CXR từ Kaggle → GCP Compute Engine + Cloud Storage

## Context

Anh có notebook `META_CXR_kaggle.ipynb` đã chạy được trên Kaggle 2× T4 (paper IEEE Access 2025, vision-language CXR report generation). Kaggle giới hạn 12h/session nên việc training nhiều epoch bị gián đoạn. Mục tiêu: migrate sang **GCP Compute Engine với 2× T4 spot** trong giới hạn **$280 Free Trial** (hết hạn 26/06/2026), giữ **kiến trúc/hyperparameters/training loop nguyên bản**, chỉ đổi data source (Kaggle dataset → GCS bucket `gs://mimic-cxr-jpg-data/`) và checkpoint destination (Kaggle Dataset → GCS).

**Đã xác nhận với anh:**
- Budget: **$280** Free Trial only (GenAI credit không apply cho GPU)
- Region: `us-central1`; phải upgrade Free Trial → Paid trước khi add GPU; tự request T4 quota
- GCS: bucket `gs://mimic-cxr-jpg-data/` (project `mimic-cxr-jpg-491409`) — đã verify chứa:
  - CSVs flat: `mimic-cxr-2.0.0-{split,chexpert,metadata,negbio}.csv`, `mimic_cxr_cleaned.csv`, `mimic-cxr-2.1.0-test-set-labeled.csv`
  - `p10/` — JPG images (`p10/<patient>/<study>/<img>.jpg`)
  - `report_p10/` — radiology `.txt` reports (`report_p10/<patient>/<study>.txt`, 6396 patients / 13,800+ files, **đã upload 2026-05-11**)
  - Dùng **gcsfuse** mount toàn bucket vào `/mnt/gcs-data/`.
- Notebook execution: **papermill** (parametrize qua YAML, log executed notebook)
- Checkpoint: ưu tiên **giữ best checkpoint vĩnh viễn trên GCS** (không bao giờ xoá) để resume tiết kiệm thời gian khi train nhiều phiên; ngoài ra giữ `last.pth` cho resume liên tục + rolling 5 latest `checkpoint_<N>.pth` cho safety chống spot eviction
- Auth: **ADC** trên VM (qua metadata server, không copy JSON key), service account JSON chỉ dùng dev local
- Logging: **W&B**, API key qua `.env`
- VM auto-stop sau training; spot fallback sang zone khác nếu hết capacity
- Repo target: `https://github.com/minhphuong150505/Meta-CXR-GCP` (đã tạo trống), local dir `/home/phuong/Documents/KLTN/Meta-CXR-GCP/`

## Phương châm
- **KHÔNG sửa repo Kaggle `META_CXR_again/`** — chỉ đọc tham khảo
- **KHÔNG sửa kiến trúc/training loop/loss/eval** — chỉ adapt 3 nhóm cell: imports/config, data paths, checkpoint
- **KHÔNG hardcode** project ID / bucket / paths / secrets — tất cả qua YAML + `.env`
- **KHÔNG commit secrets** — `.gitignore` chặn `.env`, JSON key, checkpoints, data

---

## Project Structure (sẽ tạo)

```
/home/phuong/Documents/KLTN/Meta-CXR-GCP/
├── CLAUDE.md, README.md, SETUP_GUIDE.md, COST_ESTIMATE.md
├── .env.example, .gitignore
├── Dockerfile, requirements.txt
├── notebooks/
│   ├── META_CXR_gcp.ipynb         # cloned từ META_CXR_kaggle.ipynb, đã adapt
│   └── META_CXR_eval.ipynb        # cloned từ META_CXR_eval.ipynb
├── configs/
│   ├── gcp.yaml                   # project_id, zone, machine_type, bucket, spot
│   ├── training.yaml              # batch_size, lr, epochs (mirror mimic_cxr_2gpu.yaml)
│   ├── data.yaml                  # gcs paths, mount point
│   └── checkpoint.yaml            # local_dir, gcs_prefix, keep_last_n, resume_policy
├── scripts/
│   ├── 01_setup_gcp.sh            # enable APIs, tạo service account, IAM
│   ├── 02_build_docker.sh         # build & push image lên Artifact Registry
│   ├── 03_create_vm.sh            # gcloud tạo VM 2× T4 spot (fallback zone)
│   ├── 04_train.sh                # MAIN: start VM → mount GCS → papermill → upload → stop
│   ├── 04_resume_training.sh      # resume từ checkpoint mới nhất GCS
│   ├── 05_dev_jupyter.sh          # SSH port-forward 8888 → JupyterLab trong Docker
│   ├── 06_stop_vm.sh / 07_start_vm.sh / 08_delete_all.sh
│   ├── vm_startup.sh              # gcsfuse mount + docker pull on boot
│   └── utils/
│       ├── check_budget.sh
│       └── check_gcs_checkpoints.sh
├── src/
│   ├── config_loader.py           # OmegaConf + dotenv
│   ├── gcs_checkpoint.py          # save local + upload GCS + verify + rolling delete
│   └── papermill_runner.py        # wrapper inject params từ YAML
└── tests/
    ├── test_config.py
    └── test_gcs_checkpoint.py
```

**Bổ sung — model code copy vào repo mới (anh đã chốt Option A):**
```
Meta-CXR-GCP/
├── pretraining/       # copy từ META_CXR_again/META-CXR/pretraining/
├── model/             # copy từ META_CXR_again/META-CXR/model/
├── mhcac/             # copy từ META_CXR_again/META-CXR/mhcac/
├── biovil_t/          # copy từ META_CXR_again/META-CXR/biovil_t/
├── vision_encoders/   # copy từ META_CXR_again/META-CXR/vision_encoders/
├── utils/             # copy từ META_CXR_again/META-CXR/utils/
├── threshold.json     # copy
└── ...
```
→ Repo self-contained, không phụ thuộc internet/remote. Bỏ cell 6 (clone) trong notebook vì code đã có sẵn trong Docker image build từ repo này.

---

## Adapt notebook (`notebooks/META_CXR_gcp.ipynb`)

Clone từ `META_CXR_kaggle.ipynb` rồi sửa **chỉ 7 cell**:

| Cell idx | Hiện tại (Kaggle) | Sửa thành (GCP) |
|---|---|---|
| 2 (deps) | `pip install` các package | Giữ nguyên (Docker image đã có nhưng giữ cell cho dev mode an toàn) |
| 4 (wandb) | `UserSecretsClient().get_secret("WANDB_API_KEY")` | `from dotenv import load_dotenv; load_dotenv()` rồi `os.environ["WANDB_API_KEY"]` đã set |
| 6 (clone repo) | clone `Meta-CXR-Kaggle` vào `/kaggle/working/META-CXR` | Trỏ `REPO_DIR = "/workspace/Meta-CXR-GCP"`, bỏ clone hoặc `git pull` |
| 8 (verify datasets) | Search `/kaggle/input/<slug>` | Verify mount `/mnt/gcs-data/`, check CSVs (`mimic-cxr-2.0.0-split.csv`, `mimic_cxr_cleaned.csv`…), confirm `report_p10/` folder tồn tại |
| 10 (write env_config) | Hardcode `/kaggle/input/...`, `/kaggle/working/output` | Đọc paths từ `configs/data.yaml`: `reports_root=/mnt/gcs-data/report_p10`, `mimic_cxr_jpg_root=/mnt/gcs-data`, output_dir từ `configs/training.yaml` |
| 12 (training launch) | `cwd=/kaggle/working/META-CXR`, resume từ Kaggle dataset mount | `cwd=/workspace/Meta-CXR-GCP`, resume từ `/mnt/gcs-data/checkpoints/<run_id>/` (hoặc local sau download); thêm **papermill parameters cell** ở đầu với `RUN_ID`, `RESUME_FROM` |
| 16 (push to Kaggle) | Kaggle CLI push dataset | **Thay hoàn toàn** bằng `GCSCheckpointCallback`: upload `gs://mimic-cxr-jpg-data/checkpoints/<run_id>/` với `gsutil cp + stat verify + rolling delete` |

Thêm 1 cell **parameters** (tag `parameters` cho papermill) ở đầu để override `RUN_ID`, `RESUME_FROM`, `BATCH_SIZE` từ CLI.

**Tất cả các cell khác (model, loss, training loop, eval) KHÔNG sửa.**

Cũng cần sửa `pretraining/configs/mimic_cxr_2gpu.yaml` dòng `output_dir: "/kaggle/working/output"` — inject qua papermill param hoặc env var, không hardcode.

---

## Checkpoint strategy (đáp 6.5)

`runner_base.py` của repo gốc đã save sẵn 3 loại file vào `output_dir`:
- `checkpoint_best.pth` (overwrite mỗi khi val cải thiện) → **GIỮ VĨNH VIỄN trên GCS**
- `checkpoint_last.pth` (overwrite mỗi epoch) → upload đè
- `checkpoint_<N>.pth` (mỗi `save_freq=3` epoch) → rolling delete giữ 5 latest

`src/gcs_checkpoint.py` wrap thêm chức năng upload:
1. Sau mỗi `_save_checkpoint` → `gsutil cp <local> gs://<bucket>/checkpoints/<run_id>/`
2. `gsutil stat` verify upload thành công TRƯỚC khi xoá local cũ
3. Rolling delete chỉ áp dụng cho `checkpoint_<N>.pth`, không bao giờ đụng `_best.pth` / `_last.pth`
4. SIGTERM handler (spot eviction): force-flush checkpoint hiện tại lên GCS trong 30s

Resume policy mặc định:
- Tìm `checkpoint_best.pth` trên GCS → nếu có, dùng làm starting point cho phiên mới (tiết kiệm thời gian như anh muốn)
- Override: `--resume-from checkpoint_last.pth` (continue training) hoặc `--resume-from epoch_N` (specific)

---

## GCS access pattern

**gcsfuse mount toàn bucket → `/mnt/gcs-data/`**:
- Mount lệnh: `gcsfuse --implicit-dirs mimic-cxr-jpg-data /mnt/gcs-data`
- Notebook đọc paths:
  - Images: `/mnt/gcs-data/p10/<patient>/<study>/<img>.jpg`
  - CSVs: `/mnt/gcs-data/mimic-cxr-2.0.0-split.csv`, `/mnt/gcs-data/mimic_cxr_cleaned.csv`, v.v.
  - **Reports: `/mnt/gcs-data/report_p10/<patient>/<study>.txt`** (thay `/kaggle/input/mimic-cxr-reported/files/**/s*.txt`)
- ADC qua metadata server (không cần JSON key trên VM)
- Mount tự động qua `scripts/vm_startup.sh`

Trade-off: latency I/O cao hơn local SSD. **Mitigation:** PyTorch DataLoader đã `num_workers=2`; nếu I/O bound có thể bump lên `num_workers=4` mà không đổi training loop. Nếu vẫn quá chậm, fallback (Phase 2.4): rsync subset về local SSD (split CSV để chỉ lấy `train` split).

---

## Dockerfile

Base: `nvcr.io/nvidia/pytorch:23.10-py3` hoặc `gcr.io/deeplearning-platform-release/pytorch-gpu` (đã có CUDA 12.1 + PyTorch 2.x). Cài thêm:
- `requirements.txt` (clone từ `META_CXR_again/META-CXR/requirements.txt` + thêm `papermill`, `python-dotenv`, `google-cloud-storage`)
- `gcsfuse` (apt từ Google repo)
- `gsutil` (có sẵn trong base image)
- OpenJDK 8 cho CheXpert labeler
- JupyterLab cho dev mode

---

## Cost estimate (rough, sẽ refine ở Phase 3)

Spot 2× T4 trên `n1-standard-8` ≈ **$0.38/hr**:
- Smoke test (1 epoch nhỏ, batch=2): ~2h → ~$0.80
- Full training (10 epoch × ~3h/epoch = 30h): ~$11.5
- Buffer cho spot eviction + retry: ×1.5 → ~$17
- Storage GCS (~700GB × $0.020/GB/month × 2 tháng): ~$28
- Network egress (nội region us-central1 → free): ~$0
- **Tổng dự kiến: ~$45–60 / $280** → còn dư nhiều cho retry, debug

---

## .sh scripts (key behaviors)

- **`set -euo pipefail`** strict mode
- Đọc config từ `.env` + parse YAML (qua `yq` hoặc Python helper)
- Log timestamp ra stdout
- Idempotent
- `04_train.sh`: start VM → wait SSH ready → SSH chạy Docker với `--gpus all`, mount `/mnt/gcs-data`, run `papermill notebooks/META_CXR_gcp.ipynb output/${RUN_ID}.ipynb -f configs/training.yaml` → upload executed notebook lên `gs://<bucket>/logs/` → `06_stop_vm.sh`
- `05_dev_jupyter.sh`: SSH với `-L 8888:localhost:8888`, JupyterLab trong Docker, hướng dẫn anh Ctrl+C + chạy stop khi xong
- Spot zone fallback: try `us-central1-a` → `-b` → `-c` → `-f`

---

## Critical files for implementation (đường dẫn tham chiếu)

**Sẽ tạo mới:**
- `/home/phuong/Documents/KLTN/Meta-CXR-GCP/` (toàn bộ structure trên)

**Tham chiếu read-only từ `META_CXR_again/`:**
- `META-CXR/META_CXR_kaggle.ipynb` (clone cells 0–16, sửa 7 cells)
- `META-CXR/pretraining/configs/mimic_cxr_2gpu.yaml` (mirror sang `configs/training.yaml`, đổi `output_dir`)
- `META-CXR/configs/env_config.yaml.example` (template generate ở runtime)
- `META-CXR/configs/kaggle_datasets.yaml` (replace bằng `configs/data.yaml`)
- `META-CXR/requirements.txt` (base cho Dockerfile)
- `META-CXR/local_config.py` (đọc env_config — không sửa, vẫn dùng được vì notebook sẽ generate `env_config.yaml` đúng paths GCP)
- `META-CXR/model/lavis/runners/runner_base.py` (đọc để hiểu `_save_checkpoint` signature)
- `META-CXR/CHECKPOINT_WORKFLOW.md` (hiểu pattern hiện tại)

---

## 5 phases execution (theo MIGRATION_PROMPT.md)

1. **Phase 1 — Architecture Proposal** (no code): chốt schemas YAML/.env, naming, cost projection chi tiết → đợi anh approve
2. **Phase 2 — Implementation** (module by module, mỗi module verify trước khi sang module kế):
   - 2.1 Folder + `.gitignore` + README skeleton
   - 2.2 Config layer (YAML + loader)
   - 2.3 Dockerfile + dependencies + gcsfuse
   - 2.4 Notebook adapted (diff phải minimal)
   - 2.5 `GCSCheckpointCallback` + tests
   - 2.6 All `.sh` scripts (dry-run từng cái)
   - 2.7 SIGTERM handler
3. **Phase 3 — Documentation**: `SETUP_GUIDE.md` (VN, end-to-end gồm upgrade Free Trial + request quota), `COST_ESTIMATE.md` (3 scenarios), `CLAUDE.md` repo mới
4. **Phase 4 — Git operations**: grep secrets, confirm 2 lần, push lên GitHub remote

---

## Verification

### Per-module (Phase 2):
- **Config:** `python -c "from src.config_loader import load; print(load())"` không lỗi
- **Docker:** `docker build -t meta-cxr-gcp . && docker run --rm meta-cxr-gcp python -c "import torch; print(torch.cuda.is_available())"` (locally CPU OK, GPU test trên VM)
- **Notebook diff:** `diff META_CXR_again/META-CXR/META_CXR_kaggle.ipynb notebooks/META_CXR_gcp.ipynb` — chỉ thấy thay đổi ở 7 cells đã liệt kê
- **GCS checkpoint:** `pytest tests/test_gcs_checkpoint.py` — mock save fake `.pth` 1KB, verify file lên `gs://<bucket>/checkpoints/<test_run>/`, verify rolling delete khi >5 file, verify best.pth không bị xoá
- **Scripts dry-run:** mỗi `.sh` chạy với `--dry-run` flag in ra `gcloud` command sẽ execute

### End-to-end (sau Phase 2):
1. `./scripts/01_setup_gcp.sh` — service account + IAM + APIs
2. `./scripts/02_build_docker.sh` — image push lên Artifact Registry
3. `./scripts/03_create_vm.sh` — VM up với 2× T4 spot
4. `./scripts/05_dev_jupyter.sh` — mở browser localhost:8888 → notebook chạy 1 cell test data load thành công
5. `./scripts/04_train.sh` smoke (override `max_epoch=1`, `batch_size=2`) — verify checkpoint lên `gs://mimic-cxr-jpg-data/checkpoints/<run_id>/` + VM auto-stop
6. Run lần 2 — verify auto-resume từ `checkpoint_best.pth` (skip epoch đã train)

---

## Quyết định cuối đã chốt với user

- **Code source: Option A** — copy toàn bộ model code vào repo mới (self-contained)
- **GCP billing: đã active** — không phải đợi upgrade. SETUP_GUIDE chỉ cần hướng dẫn request T4 quota nếu chưa có.
