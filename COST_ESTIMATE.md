# Cost Estimate — Meta-CXR-GCP

Free Trial budget: **$280**, hết hạn **26/06/2026**.

Tham khảo giá (us-central1, 2026-05): nguồn https://cloud.google.com/compute/all-pricing
- `n1-standard-8` spot: ~$0.076/hr
- `nvidia-tesla-t4` spot: ~$0.105/hr × 2 = $0.210/hr
- pd-balanced 200 GB: $0.030/GB-month × 200 / 730h ≈ $0.0082/hr
- **Tổng VM 2× T4 spot trên n1-standard-8 ≈ $0.30/hr** (so với $0.95/hr on-demand → tiết kiệm ~68%)
- GCS Standard: $0.020/GB-month
- Network egress nội region: **$0** (cùng us-central1)
- Network egress ra Internet (download model weights một lần): ~$0.12/GB

Giá có thể đổi; check `./scripts/utils/check_budget.sh` thường xuyên.

---

## Scenario A — Smoke test (verify pipeline)

| Component | Quantity | Cost |
|---|---|---|
| VM compute (1 epoch nhỏ, batch=2) | ~2h | $0.60 |
| Docker image pull (Internet, first boot) | ~3 GB | $0.36 |
| GCS read (CSVs + few thousand images) | ~5 GB | $0.10 |
| Checkpoint upload (best+last) | ~600 MB | < $0.01 |
| **Total** | | **~$1.10** |

## Scenario B — Full training (planned)

Giả định: 10 epoch × 3h/epoch ≈ 30h compute, 1 phiên không bị evict.

| Component | Quantity | Cost |
|---|---|---|
| VM compute | 30h × $0.30 | $9.00 |
| GCS storage (700 GB × 1.5 tháng) | 1050 GB-month | $21.00 |
| GCS read (10 epoch × 70 GB) | 700 GB | (free, intra-region) |
| Checkpoint storage (~5 GB rolling) | 5 GB-month × 2 tháng | $0.20 |
| Network egress | ~5 GB | $0.60 |
| **Total** | | **~$31** |

## Scenario C — Worst case (3× spot evictions + 1 debug iteration)

| Component | Quantity | Cost |
|---|---|---|
| Compute: 30h training + 10h re-do (eviction redo overlap với best.pth) | 40h × $0.30 | $12.00 |
| Compute: 5h debug/inference smoke | 5h × $0.30 | $1.50 |
| Storage 2 tháng | 700 GB-month × 2 | $28.00 |
| Buffer 30% | | $12.45 |
| **Total** | | **~$54** |

---

## So sánh: nếu dùng on-demand thay vì spot

Scenario B on-demand: 30h × $0.95 = $28.50 (vs $9 spot) → tiết kiệm $20.

## Đề nghị monitoring

- Mỗi sáng: `./scripts/utils/check_budget.sh` + xem Console → Billing → Credits
- Sau full train: `./scripts/utils/check_gcs_checkpoints.sh` để confirm rolling delete hoạt động (≤ 5 file numbered)
- Threshold cảnh báo: nếu credit còn < $150, switch sang resume-only strategy (training thêm chỉ khi cần thiết)

## Tổng kết

| Scenario | Cost | % of $280 budget |
|---|---|---|
| A: Smoke | $1 | < 1% |
| B: Full train + smoke | $32 | 11% |
| C: Worst (eviction + debug) | $54 | 19% |

→ Còn dư **$220+** cho retry, fine-tune, eval/inference, debug.
