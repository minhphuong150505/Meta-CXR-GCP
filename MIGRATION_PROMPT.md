# PROMPT: Migrate Meta-CXR-Kaggle → Google Cloud Compute Engine + Cloud Storage

> **Version:** 3.0 — Notebook-centric workflow, 2× T4 spot, GCS mount, .sh automation
> **Cách dùng:** Copy toàn bộ file này, paste vào Claude Code ở thư mục đã clone sẵn `Meta-CXR-Kaggle`. Claude Code sẽ tuân theo workflow Plan → Confirm → Execute, hỏi anh các thông tin cần thiết, và không tự ý làm gì trước khi anh duyệt.

---

## 0. Vai trò và nguyên tắc làm việc

Bạn là **Senior MLOps Engineer** đang được giao migrate một project Deep Learning (Vision-Language Model cho Chest X-Ray) từ Kaggle Notebook sang Google Cloud Platform (Compute Engine VM + Cloud Storage), với **budget hữu hạn từ trial credits**. Toàn bộ công việc phải tuân theo các nguyên tắc trong file `CLAUDE.md`:

1. **Think Before Coding** — Nêu rõ assumption, không đoán mò. Khi có nhiều cách hiểu, **phải hỏi**.
2. **Simplicity First** — Code tối thiểu giải quyết bài toán. Không thêm flexibility không được yêu cầu.
3. **Surgical Changes** — Code Kaggle hiện tại là **read-only reference**. Tuyệt đối **không sửa code cũ**. Mọi code mới ở thư mục riêng.
4. **Goal-Driven Execution** — Định nghĩa success criteria trước khi code. Verification sau mỗi bước.

**Bổ sung nguyên tắc cho project này:**

- ❌ **Không tự ý làm bất cứ gì** chưa được duyệt. Mỗi phase phải trình bày plan và **chờ tôi confirm** mới execute.
- ❌ **Không fix bug ngắn hạn** (workaround, hack, magic numbers). Mọi giải pháp phải maintainable lâu dài.
- ❌ **Không hardcode** GCP project ID, bucket name, paths, credentials, hyperparameters, machine types, regions. Tất cả qua YAML config hoặc `.env`.
- ❌ **Không commit/push** secrets: service account JSON, API tokens, `.env`, dataset, checkpoints, wandb keys.
- ❌ **Không refactor notebook training logic.** Notebook gốc `META_CXR_kaggle.ipynb` đã chạy được trên 2× T4 Kaggle — chỉ thay data source và checkpoint destination, **giữ nguyên kiến trúc training, hyperparameters, loss functions**.
- ✅ Khi không chắc, **dừng lại và hỏi**. Tốt hơn là hỏi 3 lần còn hơn refactor sau.

---

## 1. Budget reality

Tôi có 2 GCP credits:

### Credit 1: Free Trial (GCP standard $300)
- Remaining: ~7,003,839 VND ≈ **$280 USD**
- Period: April 26, 2026 → **June 26, 2026**
- Scope: Compute Engine + Cloud Storage OK
- **Restriction:** Free Trial mode KHÔNG được add GPU. Phải **upgrade Paid Account trước** mới dùng GPU được. Credit còn lại vẫn dùng sau upgrade.

### Credit 2: Trial credit for GenAI App Builder
- Remaining: ~26,345,000 VND ≈ **$1,000 USD**
- Period: May 2, 2026 → May 3, 2027
- Scope: "Certain usage" — **CHƯA CONFIRM** có apply cho Compute Engine không. Tên gợi ý chỉ apply cho Vertex AI Search / Agent Builder.
- **Hành động:** Tôi sẽ check T&C trong billing console và update.

### Budget assumption mặc định: **$280** (chỉ Free Trial)

### Hardware đã chốt: 2× NVIDIA T4 spot

- Machine type: `n1-standard-8` (8 vCPU, 30 GB RAM) + 2× T4 attached
- On-demand: ~$1.14/hr → ~246 hr với $280
- **Spot: ~$0.38/hr → ~737 hr với $280** ← default
- VRAM: 2× 16 GB = 32 GB total (DDP per-GPU)
- Đã được verify chạy được trên Kaggle với cùng setup → memory đã OK

### Timeline

- Free Trial expire: **June 26, 2026**
- Hard deadline training xong: **June 19, 2026** (buffer 1 tuần)
- Pattern: Stop VM khi không train, Start khi cần → tiết kiệm ~80% so với chạy 24/7

---

## 2. Bối cảnh project

### 2.1 Repo nguồn (hiện tại — không sửa)

- **URL:** `https://github.com/minhphuong150505/Meta-CXR-Kaggle`
- **Mục đích:** Reproduce paper IEEE Access 2025 *"Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model"* (META-CXR) trên Kaggle với 2× T4 GPU.
- **Tình trạng:** Notebook `META_CXR_kaggle.ipynb` chạy được trên Kaggle với 2× T4. Đã có Dockerfile, `configs/env_config.yaml.example`, OmegaConf, wandb.
- **Kiến trúc:**
  - Combined Vision Encoder (frozen): ResNet50 (BioViL-T) + ViT (PubMedCLIP) + Swin (MedClipViT)
  - META-Former (Q-Former modified): ITC + ITG + ITM losses
  - MHCAC: 8 Common Expert Tokens + 14 Specialized + Classification Heads (3-class)
  - LLM: Vicuna 7B + LoRA (rank=8, alpha=16)
- **Training:** Phase 1 (20K steps) + Phase 2 (5K steps)
- **Dataset chính:** MIMIC-CXR (jpg + reports + CheXpert + split CSV)

### 2.2 Repo đích (mới — sẽ tạo)

- **URL:** `https://github.com/minhphuong150505/Meta-CXR-GCP`
- **Mục đích:** Cùng notebook, cùng model, nhưng:
  - Data từ GCS bucket (thay Kaggle datasets)
  - Checkpoint save thẳng vào GCS (thay Kaggle output)
  - VM tự thuê trên Compute Engine 2× T4 spot
  - Toàn bộ workflow tự động hoá qua `.sh` scripts

---

## 3. Workflow tổng quan (rất quan trọng, đọc kỹ)

Project có **2 entry points**:

### A. Notebook mode (development / debug / visualize)

```bash
./scripts/dev_jupyter.sh
```

Workflow:
1. Start VM (nếu chưa chạy)
2. SSH với port forwarding `8888:localhost:8888`
3. Mount GCS bucket vào `/mnt/gcs-data/` qua gcsfuse
4. Start JupyterLab trong Docker container
5. Anh mở browser local `http://localhost:8888` → code y như Kaggle
6. Khi xong: anh tự `Ctrl+C` để stop Jupyter, run `./scripts/stop_vm.sh`

### B. Headless training mode (production)

```bash
./scripts/train_full.sh
```

Workflow:
1. Start VM (auto retry zone khác nếu spot không có capacity)
2. SSH vào, pull latest code từ git
3. Mount GCS bucket
4. Run notebook headless bằng `papermill`:
   - Inject parameters từ YAML config
   - Execute từng cell
   - Save executed notebook làm artifact log
5. Mỗi epoch: checkpoint local + upload GCS
6. Khi xong: auto upload final checkpoint + auto-stop VM

**Insight quan trọng:** Notebook là **single source of truth**. Cả 2 modes chạy chung 1 file `.ipynb`. Không có code Python riêng cho training — toàn bộ logic trong notebook.

---

## 4. Mục tiêu tổng quan (Definition of Done)

1. Thư mục mới tách biệt code Kaggle. Không sửa file Kaggle nào.
2. Edit **1 file YAML config + 1 file `.env`** là chạy được, không touch code.
3. Notebook training đọc data từ `/mnt/gcs-data/` (gcsfuse mount của GCS bucket).
4. Mỗi epoch: checkpoint save local `./checkpoints/` → upload `gs://<bucket>/checkpoints/<run_id>/` → rolling delete local.
5. `.sh` scripts cover toàn bộ lifecycle: setup → upload data → create VM → train (notebook hoặc headless) → stop → cleanup.
6. Dockerfile build được trên VM với Jupyter + papermill + gcsfuse + CUDA + PyTorch.
7. `SETUP_GUIDE.md` tiếng Việt từ A→Z, bao gồm upgrade Free Trial, request GPU quota, mount GCS, run training.
8. `COST_ESTIMATE.md` tính theo $280 với 3 scenarios (24/7 vs stop-start vs snapshot).
9. Repo `Meta-CXR-GCP` push được lên GitHub, không chứa secrets, `.gitignore` đầy đủ.
10. `CLAUDE.md` trong repo mới có rules + project structure cho session sau.

---

## 5. Workflow bắt buộc (5 phases)

### PHASE 0 — Discovery & Verification (đọc, không code)

**Bạn sẽ:**

1. Đọc repo `Meta-CXR-Kaggle`: `META_CXR_kaggle.ipynb`, `local_config.py`, `configs/env_config.yaml.example`, `Dockerfile`, `requirements.txt`, `CHECKPOINT_WORKFLOW.md`, `SETUP_GUIDE.md`.
2. Identify trong notebook:
   - Tất cả vị trí đọc path `/kaggle/input/` → cần map sang `/mnt/gcs-data/`
   - Vị trí save checkpoint → cần wrap callback upload GCS
   - Vị trí dùng Kaggle secrets (vd: wandb API key) → cần move sang `.env`
   - Hyperparameters / batch size / image size hiện tại (để verify fit 2× T4 16GB)
3. **Output Phase 0:**
   - Bản đồ điểm cần thay đổi (cell number trong notebook, line trong các .py)
   - Hyperparameter snapshot hiện tại
   - Câu hỏi clarifying (mục 6)
4. **Dừng. Chờ tôi trả lời.**

### PHASE 1 — Architecture Proposal (không code)

Trình bày design doc:

1. Cấu trúc folder repo mới.
2. Schema YAML config (chia nhóm: gcp, training, data, checkpoint).
3. Schema `.env` và `.env.example`.
4. Chiến lược notebook adaptation: liệt kê chính xác những cell nào sẽ thêm/sửa và lý do.
5. Chiến lược checkpoint: local path → GCS path → naming → rolling deletion → resume detection.
6. Chiến lược GCS mount (gcsfuse vs gcsfs vs rsync local).
7. Chiến lược Dockerfile (base image + Jupyter + papermill + gcsfuse).
8. Danh sách `.sh` scripts với mục đích từng cái.
9. Git strategy: remote, branch, `.gitignore`.
10. **Cost projection** dựa trên 2× T4 spot + stop/start pattern: ước lượng training time và compare với $280.
11. **Dừng. Chờ tôi approve.**

### PHASE 2 — Implementation (từng module một)

| # | Module | Verification |
|---|---|---|
| 2.1 | Folder structure, `.gitignore`, `README.md` skeleton | Tôi xem cấu trúc |
| 2.2 | Config layer (YAML + `.env.example` + loader Python) | Load thử bằng script test |
| 2.3 | Dockerfile + Jupyter + gcsfuse + dependencies | Build local, run `--gpus all`, mount test bucket |
| 2.4 | Notebook adapted (clone `META_CXR_kaggle.ipynb` → `META_CXR_gcp.ipynb`, change data paths + checkpoint logic) | Diff với notebook gốc — chỉ có thay đổi cần thiết |
| 2.5 | Checkpoint callback (local save + GCS upload + rolling delete) | Mock test: save checkpoint giả, verify file lên GCS |
| 2.6 | `.sh` scripts (xem mục 8 cho danh sách đầy đủ) | Dry-run từng script |
| 2.7 | Spot eviction handler: SIGTERM → emergency checkpoint upload GCS | Simulate SIGTERM, verify checkpoint uploaded |

### PHASE 3 — Documentation

1. **`SETUP_GUIDE.md`** tiếng Việt, đầy đủ steps:
   - Prerequisites (gcloud CLI, billing, account).
   - **Upgrade Free Trial → Paid account** (có hình mô tả hoặc lệnh).
   - **Verify credit scope** (cách check T&C GenAI credit).
   - Enable APIs: Compute Engine, Cloud Storage.
   - Tạo service account, download key, **lưu ngoài repo**.
   - Tạo GCS bucket cùng region với VM (us-central1 mặc định).
   - Upload MIMIC-CXR dataset lên GCS (script `02_upload_dataset.sh`).
   - **Request GPU quota** cho T4 (default = 0, mất 1-2 ngày).
   - Điền `.env`.
   - Build Docker image (script `03_build_docker.sh`).
   - Tạo VM (script `03_create_vm.sh`).
   - Chạy training (script `04_train.sh` hoặc `dev_jupyter.sh`).
   - **Monitor training:** xem log, check checkpoint GCS, check W&B.
   - **Resume sau spot eviction:** auto via script `04_train.sh --resume`.
   - **Stop/start VM** để tiết kiệm chi phí.
   - Cleanup khi xong: stop VM, optional delete VM giữ snapshot.

2. **`COST_ESTIMATE.md`** với 3 scenarios:
   - **Smoke test:** 1 epoch nhỏ, batch size = 2, ~2 hr, ~$0.80
   - **Recommended:** Full Phase 1 + 2 (25K steps), stop/start pattern, ~40-60 hr training, ~$15-25
   - **Worst case:** 24/7 run + spot eviction loss + retry, ~$50-80
   - Mỗi scenario: storage, compute, network, idle cost. So với $280 còn dư bao nhiêu.

3. **`CLAUDE.md`** trong repo mới (xem mục 9).

4. **`README.md`** ngắn: project là gì, link SETUP_GUIDE, link repo Kaggle, link paper.

### PHASE 4 — Git Operations

Trước khi push:

1. `git status`, `git diff --staged`, đưa tôi xem files sẽ commit.
2. Verify `.gitignore` loại trừ: `.env`, `*.json` (service account), `__pycache__/`, `checkpoints/`, `wandb/`, `*.pth`, `*.ckpt`, `outputs/`, `data/`, `.venv/`, `*.log`, `.ipynb_checkpoints/`.
3. Grep tất cả files commit, đảm bảo KHÔNG có: GCP project ID thực, bucket name thực, email service account, API keys.
4. Đợi tôi confirm **2 lần** (file list, grep result) trước `git push`.

```bash
cd <thư mục mới>
git init
git remote add origin https://github.com/minhphuong150505/Meta-CXR-GCP.git
git add .
git commit -m "<message rõ ràng>"
git push -u origin main
```

---

## 6. Câu hỏi clarifying BẮT BUỘC ở cuối Phase 0

### 6.1 Credit verification (ƯU TIÊN CAO NHẤT)
- Tôi đã upgrade Free Trial → Paid account chưa?
- GenAI App Builder credit có applicable cho Compute Engine không? (Tôi sẽ check T&C, đợi tôi confirm)
- Total budget cho bạn plan: **$280** (chỉ Free Trial) hay **$1,280** (cả 2)?

### 6.2 Region & GPU quota
- Region preference: `us-central1` (rẻ nhất, spot capacity tốt nhất) hay `asia-southeast1` (gần Việt Nam, latency thấp)?
- Tôi đã request quota T4 chưa? Nếu chưa, bạn sẽ ghi vào SETUP_GUIDE.

### 6.3 Data access pattern
- Dataset MIMIC-CXR đã upload GCS chưa? Nếu chưa, bạn sẽ làm script `02_upload_dataset.sh`. Tôi cần đưa bạn cấu trúc dataset hiện tại (path Kaggle).
- GCS access method:
  - **(A)** `gcsfuse` mount toàn bộ bucket vào `/mnt/gcs-data/` → notebook đọc path giống Kaggle, KHÔNG cần đổi data loader nhiều. Trade-off: latency cao hơn, có cache nhưng overhead I/O.
  - **(B)** `gsutil rsync` toàn bộ dataset về SSD VM 1 lần trước khi train → đọc local nhanh nhất. Trade-off: tốn disk space + thời gian sync ban đầu (MIMIC-CXR ~500GB, ~30-60 phút sync).
  - **(C)** Hybrid: CSV/index local, JPG stream từ GCS qua gcsfs.
- **Default đề xuất: (A) gcsfuse** vì giữ notebook không đổi nhiều. Confirm?

### 6.4 Notebook execution mode
- Notebook headless qua **papermill** (parameterize qua YAML) hay **jupyter nbconvert --execute** (đơn giản hơn nhưng không parameterize được)?
- **Default đề xuất: papermill** vì cho phép override hyperparameter qua CLI mà không sửa notebook.

### 6.5 Checkpoint strategy
- Format: `.pth` thuần PyTorch (như Kaggle hiện tại) hay `safetensors`?
- Save mỗi epoch (mặc định) hay mỗi N steps (an toàn hơn với spot)?
- Giữ tối đa N checkpoint gần nhất trên GCS? (rolling deletion để tiết kiệm storage)
- Resume mặc định từ `latest.pth` trên GCS, hay cho phép `--resume-from epoch_X.pth`?
- **Default đề xuất:** `.pth`, save mỗi epoch + mỗi 500 steps, giữ 5 latest, auto-resume từ `latest.pth`.

### 6.6 Authentication
- Service account JSON key file (download về local) hay Application Default Credentials qua VM metadata server?
- **Default đề xuất:** Service account JSON cho dev local, ADC cho VM (an toàn hơn vì không cần copy key vào VM).

### 6.7 Logging
- W&B (như Kaggle), TensorBoard (sync GCS), hay cả 2?
- W&B API key qua `.env` (không hardcode trong notebook).

### 6.8 VM lifecycle
- Auto-stop VM sau khi training xong (mặc định) hay giữ chạy để debug?
- Spot fallback: nếu zone không có capacity, tự thử zone khác (mặc định) hay fail luôn?

### 6.9 Project naming
- Tên thư mục: `gcp_training/` hay khác? Monorepo hay separate repo?
- **Default: separate repo** vì repo đích `Meta-CXR-GCP` riêng biệt.

---

## 7. Cấu trúc folder đề xuất (Phase 1 sẽ chốt)

```
Meta-CXR-GCP/                          # Repo mới, separate
├── CLAUDE.md                           # Rules + project structure
├── README.md                           # Overview ngắn
├── SETUP_GUIDE.md                      # Hướng dẫn end-to-end tiếng Việt
├── COST_ESTIMATE.md                    # Cost theo $280, 3 scenarios
├── .env.example                        # Template biến môi trường
├── .gitignore                          # Loại trừ secrets, checkpoints, data
├── Dockerfile                          # Image: CUDA + PyTorch + Jupyter + papermill + gcsfuse
├── requirements.txt                    # Python deps
├── notebooks/
│   ├── META_CXR_gcp.ipynb              # Training notebook (clone từ Kaggle, đã adapt)
│   └── META_CXR_eval.ipynb             # Evaluation notebook (clone từ Kaggle)
├── configs/
│   ├── gcp.yaml                        # project_id, zone, machine_type, bucket, spot
│   ├── training.yaml                   # batch_size, lr, epochs, steps, ckpt_every_n_steps
│   ├── data.yaml                       # GCS paths, mount point, split files
│   └── checkpoint.yaml                 # local_dir, gcs_prefix, keep_last_n, resume_policy
├── scripts/
│   ├── 01_setup_gcp.sh                 # Enable APIs, tạo service account, IAM
│   ├── 02_upload_dataset.sh            # Sync MIMIC-CXR local/Kaggle → GCS bucket
│   ├── 03_build_docker.sh              # Build image, push lên GCR/Artifact Registry
│   ├── 03_create_vm.sh                 # gcloud tạo VM 2× T4 spot, mount disk, startup script
│   ├── 04_train.sh                     # MAIN: start VM → mount GCS → run notebook headless qua papermill → upload final → stop VM
│   ├── 04_resume_training.sh           # Resume từ checkpoint mới nhất trên GCS
│   ├── 05_dev_jupyter.sh               # Dev mode: start VM → SSH port-forward 8888 → start JupyterLab
│   ├── 06_stop_vm.sh                   # Stop VM (giữ disk + checkpoint)
│   ├── 07_start_vm.sh                  # Start lại VM đã stop
│   ├── 08_snapshot_disk.sh             # Snapshot disk khi nghỉ dài
│   ├── 09_delete_all.sh                # Cleanup: VM + disk (giữ GCS bucket)
│   └── utils/
│       ├── check_budget.sh             # Query Cloud Billing API
│       ├── check_gcs_checkpoints.sh    # List checkpoints trên GCS
│       └── tail_logs.sh                # Tail log từ training run đang chạy
├── src/
│   ├── config_loader.py                # OmegaConf load YAML + .env
│   ├── gcs_checkpoint.py               # CheckpointCallback: local save + GCS upload + rolling delete
│   └── papermill_runner.py             # Wrapper execute notebook với params từ YAML
└── tests/
    ├── test_config.py                  # Sanity check config loading
    └── test_gcs_checkpoint.py          # Test save/upload/restore checkpoint giả
```

---

## 8. Chi tiết `.sh` scripts (Phase 2.6)

Quy tắc chung:
- Bash strict mode: `set -euo pipefail`
- Đọc config từ `.env` và `configs/gcp.yaml` (parse YAML bằng `yq` hoặc Python helper)
- Log mọi action ra stdout với timestamp
- Idempotent: chạy lại không gây hỏng

### `04_train.sh` (script quan trọng nhất)

Pseudocode:

```bash
#!/bin/bash
set -euo pipefail
source .env
RUN_ID=$(date +%Y%m%d_%H%M%S)

# 1. Start VM (fallback zone nếu spot không có capacity)
./scripts/07_start_vm.sh || ./scripts/03_create_vm.sh

# 2. Wait VM ready (gcloud compute ssh test)
wait_until_ssh_ready

# 3. SSH vào VM, run training
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
  set -e
  cd /workspace/Meta-CXR-GCP
  git pull
  
  # Mount GCS nếu chưa
  if ! mountpoint -q /mnt/gcs-data; then
    gcsfuse --implicit-dirs $GCS_BUCKET /mnt/gcs-data
  fi
  
  # Run notebook headless với papermill
  docker run --gpus all --rm \
    -v /workspace/Meta-CXR-GCP:/workspace \
    -v /mnt/gcs-data:/mnt/gcs-data \
    -e RUN_ID=$RUN_ID \
    meta-cxr-gcp:latest \
    papermill notebooks/META_CXR_gcp.ipynb \
              notebooks/runs/${RUN_ID}_output.ipynb \
              -f configs/training.yaml
  
  # Upload executed notebook lên GCS làm log
  gsutil cp notebooks/runs/${RUN_ID}_output.ipynb \
            gs://$GCS_BUCKET/logs/${RUN_ID}_notebook.ipynb
"

# 4. Auto-stop VM sau training xong
./scripts/06_stop_vm.sh
```

### `05_dev_jupyter.sh` (dev mode)

```bash
#!/bin/bash
set -euo pipefail
source .env

./scripts/07_start_vm.sh || ./scripts/03_create_vm.sh
wait_until_ssh_ready

# Port forward 8888 (Jupyter) và 6006 (TensorBoard)
gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --ssh-flag="-L 8888:localhost:8888" \
  --ssh-flag="-L 6006:localhost:6006" \
  --command="
    cd /workspace/Meta-CXR-GCP
    if ! mountpoint -q /mnt/gcs-data; then
      gcsfuse --implicit-dirs $GCS_BUCKET /mnt/gcs-data
    fi
    docker run --gpus all --rm \
      -v /workspace/Meta-CXR-GCP:/workspace \
      -v /mnt/gcs-data:/mnt/gcs-data \
      -p 8888:8888 -p 6006:6006 \
      meta-cxr-gcp:latest \
      jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
  "

echo "Mở browser: http://localhost:8888"
echo "Khi xong, Ctrl+C ở đây, rồi chạy ./scripts/06_stop_vm.sh"
```

### `03_create_vm.sh`

```bash
#!/bin/bash
set -euo pipefail
source .env

gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type=n1-standard-8 \
  --accelerator="type=nvidia-tesla-t4,count=2" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-standard \
  --metadata-from-file=startup-script=scripts/vm_startup.sh \
  --scopes=cloud-platform \
  --service-account="$SERVICE_ACCOUNT_EMAIL"
```

---

## 9. Notebook adaptation strategy (Phase 2.4)

**Quy tắc vàng:** Notebook gốc đã chạy được trên 2× T4 Kaggle → giữ nguyên kiến trúc model, training loop, hyperparameters. Chỉ thay đổi 3 nhóm cell:

### Group A: Imports & Config (đầu notebook)
- Thêm import: `omegaconf`, `papermill` parameters cell, `gcsfs` hoặc đường dẫn gcsfuse
- Load config từ `/workspace/configs/training.yaml` thay vì hardcode
- Parameters cell (papermill tag): `RUN_ID`, `RESUME_FROM`, `BATCH_SIZE` (override được)

### Group B: Data paths
- Thay `/kaggle/input/mimic-cxr-jpg/...` → `/mnt/gcs-data/mimic-cxr-jpg/...`
- Thay `/kaggle/input/chexpert/...` → `/mnt/gcs-data/chexpert/...`
- Path mapping qua `configs/data.yaml`, KHÔNG hardcode

### Group C: Checkpoint save (trong training loop)
- Wrap save logic bằng `GCSCheckpointCallback` từ `src/gcs_checkpoint.py`
- Logic:
  1. `torch.save(state, f"./checkpoints/epoch_{N}.pth")`
  2. `gsutil cp ./checkpoints/epoch_{N}.pth gs://$BUCKET/checkpoints/$RUN_ID/`
  3. Update `./checkpoints/latest.pth` symlink + cùng update trên GCS
  4. Rolling delete: nếu local có >5 file → xoá file cũ nhất
  5. Verify GCS upload thành công bằng `gsutil stat` trước khi xoá local
- Signal handler `SIGTERM`: trước khi VM bị kill (spot eviction), force save checkpoint + upload GCS, có 30s.

**Tất cả các cell khác (model, loss, training loop, evaluation) → KHÔNG sửa.**

---

## 10. Quy ước & ràng buộc kỹ thuật

### 10.1 Config loading
- `OmegaConf` (đã có trong repo Kaggle).
- Env var interpolation: `bucket: ${oc.env:GCS_BUCKET}`.
- `.env` load bằng `python-dotenv`.
- Papermill parameters: `RUN_ID`, `RESUME_FROM`, có thể override bất kỳ key nào trong YAML qua CLI.

### 10.2 Checkpoint
- Path local: `./checkpoints/epoch_<N>_step_<S>.pth`
- Path GCS: `gs://<bucket>/checkpoints/<run_id>/epoch_<N>_step_<S>.pth`
- Chứa: model state, optimizer, scheduler, epoch, step, RNG states, config snapshot, git commit hash.
- `latest.pth` symlink local + cùng update GCS metadata.
- Rolling deletion: giữ N latest (configurable).
- SIGTERM handler: emergency upload trong 30s.

### 10.3 Logging
- Python `logging` module (không `print`).
- Format: timestamp + level + module + message.
- Log file local `./logs/<run_id>.log` + tail upload GCS sau training.

### 10.4 Reproducibility
- Seed `torch`, `numpy`, `random`, `cuda`.
- Save config snapshot + git commit hash vào checkpoint.

### 10.5 Cost-conscious defaults
- VM spot mặc định.
- Stop VM ngay sau training xong (auto trong `04_train.sh`).
- GCS bucket cùng region VM (tránh egress).
- Boot disk Standard 200GB (đủ cho Docker image + code, dataset mount qua gcsfuse).
- Persistent disk có thể tách riêng nếu cần cache dataset local.

---

## 11. File CLAUDE.md trong repo mới

Tạo `Meta-CXR-GCP/CLAUDE.md`:

```markdown
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project-Specific Rules (Meta-CXR-GCP)

5. **Không hardcode** GCP project ID, bucket name, paths, credentials, hyperparameters. Tất cả qua `configs/*.yaml` hoặc `.env`.
6. **Không commit secrets:** `.env`, service account JSON, API keys, dataset, checkpoints, wandb keys. Grep trước mỗi commit.
7. **Không sửa code Kaggle gốc.** Repo này tách biệt với `Meta-CXR-Kaggle`.
8. **Không sửa kiến trúc training trong notebook.** Notebook gốc đã chạy được trên 2× T4 → chỉ adapt 3 nhóm cell: imports/config, data paths, checkpoint save. Mọi cell khác (model, loss, training loop, eval) giữ nguyên.
9. **Không fix bug ngắn hạn.** Mọi giải pháp phải maintainable lâu dài.
10. **Budget-aware:** mặc định 2× T4 spot. Mọi script tạo VM phải có `--provisioning-model=SPOT`. Mọi training script phải auto-stop VM khi xong.
11. **Checkpoint integrity:** không xoá local checkpoint khi GCS upload chưa verify thành công (`gsutil stat`).
12. **Hỏi trước khi:** đổi cấu trúc folder, đổi tên file/module, thêm dependency, đổi format checkpoint, đổi schema config, đổi machine type, region, đổi data access method (gcsfuse ↔ gcsfs ↔ rsync).

## Project Structure

[Fill ở Phase 3 sau khi structure chốt]

## Key Conventions

[Fill ở Phase 3]

## Do Not Touch

[Fill ở Phase 3 — list file/folder không được sửa]
```

---

## 12. Output format cho mỗi phase

```
## Phase X — [Tên]

### Plan
[Việc sẽ làm, ngắn gọn]

### Assumptions
[Giả định, đánh dấu "ASSUMPTION:"]

### Questions for you
[Câu hỏi clarifying nếu có]

### Cost impact (nếu có)
[Phase này ảnh hưởng budget thế nào]

### Verification criteria
[Làm sao biết phase này xong]

---
⏸️ **Waiting for your confirmation before proceeding.**
```

---

## 13. Bắt đầu

Bắt đầu **Phase 0 (Discovery & Verification)** ngay. Đừng code gì. Đọc repo, đọc `META_CXR_kaggle.ipynb` cell by cell để identify chỗ cần adapt (data paths + checkpoint logic), analyze hyperparameters hiện tại, và đặt câu hỏi clarifying mục 6.

Nếu có phần nào trong prompt không rõ, **hỏi tôi trước** thay vì diễn giải.
