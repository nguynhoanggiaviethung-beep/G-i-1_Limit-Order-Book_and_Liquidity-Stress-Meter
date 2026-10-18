# Crypto Volatility & Tail Risk

Backend nghiên cứu dùng Binance Spot BTCUSDT và ETHUSDT, nến 5 phút, realized variance và forecast trực tiếp 1/7/30 ngày. Code nghiên cứu hiện tách hai chế độ: `research` đọc kết quả walk-forward từ snapshot bất biến; `production` đọc dữ liệu Binance đang đồng bộ và tạo forecast mới mà không sửa snapshot hay kết quả lịch sử. Mốc nghiên cứu được cấu hình tại `2026-10-01T00:00:00Z` trong [research.yaml](config/research.yaml).

Phần Limit Order Book/Liquidity Stress Meter được giữ làm tài liệu cũ tại [limit_order_book_legacy.md](docs/limit_order_book_legacy.md); pipeline hiện tại tập trung vào volatility và tail risk.

## Cài đặt và đồng bộ Binance

Chạy trong PowerShell, tại thư mục gốc repository:

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt -r requirements-dev.txt
& $PY backend\scripts\sync_data.py
```

Python 3.11 được dùng làm môi trường chuẩn. Các package lõi đã khóa phiên bản trong `requirements.txt`; test/development ở `requirements-dev.txt`. Cài thêm LightGBM và XGBoost để chạy đủ các mô hình cây:

```powershell
& $PY -m pip install -r requirements-ml.txt
```

PyTorch là tùy chọn riêng cho các thử nghiệm deep learning, hiện chưa thuộc pipeline phát hành:

```powershell
& $PY -m pip install -r requirements-dl.txt
```

Lần tải đầu lấy lịch sử từ `2019-01-01`; các lần sau đồng bộ tiếp phần còn thiếu và lưu nến đóng vào `data/processed/`. Dữ liệu thô và Parquet lớn nằm ngoài Git. Khi FastAPI đang chạy, bộ đồng bộ nền tiếp tục cập nhật mỗi 5 phút; xem trạng thái tại `http://127.0.0.1:8000/api/v1/sync-status`.

## Chạy nghiên cứu

Lệnh dưới đây dựng snapshot đúng mốc cấu hình, tạo feature, chạy walk-forward, conditional quantile VaR/ES, robustness, đánh giá kinh tế và xuất bảng/biểu đồ. Để đồng bộ Binance trước pipeline, thêm `--sync`. Snapshot có sẵn được giữ bất biến; chỉ dùng `--force-snapshot` khi chủ động muốn dựng lại nó từ dữ liệu local.

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY backend\scripts\run_pipeline.py
```

Một lượt đầy đủ có thể chạy lâu: có nhiều symbol/horizon, GARCH, quantile forest và các biến thể robustness. Mô hình LightGBM/XGBoost được bỏ qua có thông báo nếu chưa cài `requirements-ml.txt`. Lệnh nhanh dưới đây chỉ chạy baseline RV, không chạy phần risk, quantile hay robustness:

```powershell
& $PY backend\scripts\run_pipeline.py --symbol BTCUSDT --horizon 1 --model random_walk --skip-mcs --skip-risk --skip-quantile --skip-robustness
```

Kết quả được ghi dưới `results/forecasts/`, `results/metrics/`, `results/tables/` và `results/figures/`. Pipeline xuất các bảng T1–T11, SVG F1–F9, regime metrics, volatility targeting, risk budget và dynamic margin. Kiểm tra độ phủ dữ liệu trong `data/snapshots/<as-of>/manifest.json` và `results/metrics/snapshot_quality.json`; các khoảng thiếu được giữ nguyên, không nội suy.

Workspace hiện có ma trận forecast RV baseline cho BTCUSDT/ETHUSDT × 1/7/30 ngày, gồm 16 model cổ điển mỗi chuỗi và 1,005 origin OOS mỗi model. `risk_forecasts.parquet` có 707,520 dòng `full_oos`; kết quả 30 ngày vẫn chỉ mang tính mô tả. Bảng T1–T11 và hình F1–F9 được tạo từ các artifact này cùng kết quả robustness. Quantile VaR/ES trực tiếp chưa chạy; LightGBM/XGBoost chưa cài nên các model đó được bỏ qua có thông báo. Dùng pipeline đầy đủ sau khi cài package tùy chọn để bổ sung các kết quả còn thiếu.

## Chạy API và xem dự báo

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m uvicorn app.main:app --app-dir backend --reload
```

Mở tài liệu API tại [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Endpoint `POST /api/v1/forecasts` nhận JSON:

```json
{
  "symbol": "BTCUSDT",
  "horizon_days": 1,
  "model": "har",
  "mode": "research",
  "confidence_levels": [0.95, 0.975, 0.99]
}
```

Đổi `mode` thành `production` để fit lại mô hình trên các ngày hoàn chỉnh từ dữ liệu Binance local mới nhất. Production từ chối feed đã cũ hơn 15 phút; VaR/ES của response được ghi rõ là xấp xỉ phân phối chuẩn. Chế độ này không lưu hay sửa forecast research. Model production mặc định dùng hyperparameter trong code; tuning theo purged CV áp dụng cho walk-forward research.

Ví dụ PowerShell cho production:

```powershell
$body = @{
  symbol = "BTCUSDT"
  horizon_days = 1
  model = "har"
  mode = "production"
  confidence_levels = @(0.95, 0.975, 0.99)
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/forecasts" `
  -ContentType "application/json" -Body $body
```

`research` cần các forecast đã tạo trước bằng pipeline. `GET /api/v1/forecasts/{forecast_id}` tra forecast nghiên cứu đã lưu; response production là dự báo tức thời, không được lưu để tra lại.

## Thiết kế, giới hạn và trạng thái

- Quy tắc nguồn, mốc thời gian, feature, chia tập và thước đo nằm trong [research_design.md](docs/research_design.md), [methodology.md](docs/methodology.md) và [data_dictionary.md](docs/data_dictionary.md).
- Nến 5 phút được xem là hoàn chỉnh khi có đủ 288 nến và 288 lợi suất liên tiếp trong ngày UTC. Ngày lỗi/gap không trở thành target.
- Risk 7/30 ngày dùng giả định đường sigma ngày phẳng trong FHS; kết quả 30 ngày chỉ mang tính mô tả do mẫu kiểm định hiệu dụng nhỏ.
- VaR/ES gồm các phương pháp phân phối, FHS/EVT và mô hình quantile trực tiếp (`linear_qr`, `lightgbm_quantile`, `qrf`). Với quantile model, VaR là quantile có điều kiện; ES là trung bình đuôi loss trong tập huấn luyện tại thời điểm forecast, nên được ghi nhận là ước lượng đuôi thực nghiệm. Quantile và robustness dùng cấu hình riêng, không sửa `config/research.yaml`.
- Biến ngoại sinh phái sinh/vĩ mô không được bật vì chưa có nguồn lịch sử point-in-time đã kiểm chứng. Deep learning được hoãn; `requirements-dl.txt` chỉ là môi trường tùy chọn, chưa có kết quả DL trong pipeline. Không tự tạo hoặc nội suy dữ liệu ngoài.
- Bài kiểm tra chạy bằng `& $PY -m pytest -c backend\pytest.ini backend\tests --basetemp .pytest-tmp`. CI tự chạy test trên Python 3.11.
- `data/`, model files và kết quả Parquet lớn không commit. Chỉ commit code, cấu hình và báo cáo nhỏ đã rà soát.
