# SETUP_GUIDE — Meta-CXR-GCP (VN)

Hướng dẫn end-to-end migrate training từ Kaggle sang GCP Compute Engine **2× T4 spot** trong giới hạn **Free Trial $280** (hết hạn 26/06/2026).

> **Giả định**: anh đã active billing trên project `mimic-cxr-jpg-491409`. Nếu chưa, vào Console → Billing → Link a billing account TRƯỚC. Free Trial không apply GPU; cần Upgrade to Paid (vẫn dùng credit còn lại).

---

## 0. Prerequisites trên máy local

```bash
# Cài gcloud CLI (Debian/Ubuntu)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init                          # chọn project mimic-cxr-jpg-491409
gcloud auth application-default login
docker --version                     # cần Docker 20.10+
```

## 1. Cấu hình `.env`

```bash
cd /home/phuong/Documents/KLTN/META_CXR_GCP
cp .env.example .env
$EDITOR .env                         # điền WANDB_API_KEY, WANDB_ENTITY
```

> KHÔNG commit `.env`. `.gitignore` đã chặn.

## 2. Request T4 quota (chỉ làm 1 lần)

Console → IAM & Admin → Quotas. Filter:
- Service: Compute Engine API
- Metric: `Preemptible NVIDIA T4 GPUs` (region `us-central1`)

Click **Edit**, request `2` (hoặc cao hơn). Thường được duyệt < 24h. Nếu vội: thử quota cho `NVIDIA T4 GPUs` (on-demand) cũng — sẽ đắt hơn nhưng available ngay.

## 3. Setup GCP (1 lần)

```bash
./scripts/01_setup_gcp.sh
```
Script làm:
- Enable APIs (compute, storage, artifactregistry, iam, logging, monitoring)
- Tạo service account `meta-cxr-training@...`
- Bind roles: `storage.objectAdmin`, `artifactregistry.reader`, `logging.logWriter`, `monitoring.metricWriter`
- Tạo Artifact Registry repo `meta-cxr` ở `us-central1`

Verify:
```bash
gcloud iam service-accounts list | grep meta-cxr-training
gcloud artifacts repositories list --location=us-central1
```

## 4. Build và push Docker image

```bash
./scripts/02_build_docker.sh
```
Lần đầu: build ~15–30 phút (base GCP DL image ~10 GB + deps ~3 GB). Push lên Artifact Registry ~5–10 phút tuỳ băng thông.

## 5. Tạo VM (2× T4 spot)

```bash
./scripts/03_create_vm.sh
```
Script tự fallback zone: `us-central1-a` → `-b` → `-c` → `-f` nếu hết spot capacity.

Verify VM up + GPU detect:
```bash
gcloud compute ssh meta-cxr-train --zone=$(cat .vm_zone) --command='nvidia-smi'
```

## 6. Smoke test (1 epoch, batch=2 — verify pipeline)

```bash
SMOKE=1 RUN_ID=smoke-test-1 ./scripts/04_train.sh
```
Mất ~30–60 phút. Verify:
- `gs://mimic-cxr-jpg-data/checkpoints/smoke-test-1/checkpoint_best.pth` xuất hiện
- `gs://mimic-cxr-jpg-data/logs/smoke-test-1.ipynb` xuất hiện (executed notebook)
- VM auto-stop sau khi xong (xem `gcloud compute instances list`)

## 7. Full training

```bash
RUN_ID=mimic_cxr_full_v1 ./scripts/04_train.sh
```
Ước tính: 10 epoch × ~3h ≈ 30h compute. Spot có thể bị evict; script không tự retry khi evict — anh phải resume bằng tay:

```bash
RUN_ID=mimic_cxr_full_v1 ./scripts/04_resume_training.sh
```
Mặc định: pick `checkpoint_best.pth` từ GCS. Để tiếp tục từ epoch cuối:
```bash
RUN_ID=mimic_cxr_full_v1 ./scripts/04_resume_training.sh checkpoint_last.pth
```

## 8. Dev mode (JupyterLab)

Khi cần debug interactively:
```bash
./scripts/05_dev_jupyter.sh
# Mở browser: http://localhost:8888/lab?token=meta-cxr-dev
```
Ctrl+C để cắt tunnel, rồi `./scripts/06_stop_vm.sh` để dừng VM (tránh đốt credit).

## 9. Cost monitoring

```bash
./scripts/utils/check_budget.sh
./scripts/utils/check_gcs_checkpoints.sh
```
Free Trial USD số dư: Console → Billing → Credits.

## 10. Cleanup

```bash
./scripts/06_stop_vm.sh                # Pause (disk persists, không tốn GPU $)
./scripts/08_delete_all.sh             # Xoá VM (giữ GCS bucket + Artifact Registry)
```
Xoá hoàn toàn:
```bash
gsutil -m rm -r gs://mimic-cxr-jpg-data/checkpoints
gcloud artifacts repositories delete meta-cxr --location=us-central1
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `ZONE_RESOURCE_POOL_EXHAUSTED` | Hết spot capacity zone | Script tự fallback zone; nếu cả 4 zone fail, đợi 15–30 phút rồi retry |
| `Permission denied (publickey)` SSH | OS Login chưa enable trên user | `gcloud compute os-login ssh-keys add` hoặc dùng IAP tunnel |
| `gcsfuse: Permission denied` trên VM | Service account thiếu `storage.objectAdmin` | Re-run `01_setup_gcp.sh` |
| `papermill: command not found` | Dùng image cũ chưa rebuild | `./scripts/02_build_docker.sh` rebuild + push |
| CUDA OOM trong smoke test | `batch_size_train` vẫn > 2 | Confirm `SMOKE=1` env được pass; xem `output/<run_id>.ipynb` cell 6 logs |

## Layout in-VM (sau khi gcsfuse mount)

```
/mnt/gcs-data/                              # gs://mimic-cxr-jpg-data/
├── p10/<patient>/<study>/<img>.jpg
├── report_p10/<patient>/<study>.txt
├── mimic-cxr-2.0.0-{split,chexpert,metadata,negbio}.csv
├── mimic_cxr_cleaned.csv
└── checkpoints/<run_id>/                   # uploaded by gcs_checkpoint.py
    ├── checkpoint_best.pth                 # GIỮ VĨNH VIỄN
    ├── checkpoint_last.pth
    └── checkpoint_<N>.pth                  # rolling 5 mới nhất
```
