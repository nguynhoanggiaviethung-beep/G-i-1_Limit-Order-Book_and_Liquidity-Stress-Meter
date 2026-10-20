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

Tải nguồn ngoại sinh để khảo sát riêng (không thay đổi snapshot baseline). Binance futures không cần API key; FRED cần API key đặt trong biến môi trường PowerShell:

```powershell
$env:FRED_API_KEY = "<FRED API key của bạn>"
& $PY backend\scripts\fetch_external_data.py
& $PY backend\scripts\build_external_features.py
```

Fetcher lưu `data/external/events.parquet`; feature point-in-time tách riêng dưới `data/features/external/<as-of>/` kèm coverage report. OI/basis có thể bị loại do không đủ lịch sử.

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

Workspace có ma trận forecast RV cho BTCUSDT/ETHUSDT × 1/7/30 ngày. Mỗi chuỗi có 1,005 origin OOS cho 16 model cổ điển và ba model cây (`lightgbm_l2`, `lightgbm_qlike`, `xgboost`). Forecast quantile trực tiếp (`linear_qr`, `lightgbm_quantile`, `qrf`) được lưu riêng trong `results/forecasts/quantile_forecasts.parquet` và được đưa vào đánh giá VaR/ES. Các bảng T1–T11 và hình F1–F9 được tái tạo từ forecast, risk và robustness artifacts. Kết quả horizon 30 ngày vẫn chỉ mang tính mô tả do số quan sát kiểm định hiệu dụng nhỏ. LSTM, GRU và Transformer có code tùy chọn nhưng chưa có forecast metrics nếu chưa cài PyTorch và chạy walk-forward cho các model này.

## Chạy giao diện frontend

Frontend React/TypeScript nằm trong `frontend/` và được tối ưu để trình bày trên laptop/PC. Thanh đầu trang có hai chế độ: “Dự báo” dùng dữ liệu Binance mới nhất; “Kết quả nghiên cứu” mở điều hướng tới toàn bộ bảng T1–T11 và hình F1–F9 trên dữ liệu lịch sử đã lưu. Màn hình dự báo giải thích bốn chỉ số bằng tiếng Việt, cho nhập số tiền, hiện khoảng thời gian dự báo theo giờ Việt Nam, xem 60 quan sát walk-forward và đối chiếu với HAR. Trang đánh giá mô hình nêu QLIKE của mô hình tốt nhất so với HAR theo tài sản, kỳ hạn và số kỳ kiểm tra tương ứng.

VaR/ES chỉ được quy đổi sang số tiền khi backend trả về Filtered Historical Simulation (FHS). Production hiện trả VaR/ES xấp xỉ chuẩn, vì vậy giao diện ghi rõ phương pháp và ẩn số tiền lỗ cho tới khi có FHS. Kỳ hạn 30 ngày được gắn nhãn mô tả; các cột p-value bị ẩn.

Mở hai terminal tại thư mục repository. Terminal thứ nhất chạy backend như hướng dẫn ở phần tiếp theo. Terminal thứ hai cài và chạy frontend bằng Node.js 22 trở lên:

```powershell
cd frontend
npm ci
npm run dev
```

Mở [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite chuyển tiếp `/api` tới backend ở `127.0.0.1:8000`; luồng giá nến đang chạy được làm mới trên giao diện mỗi 5 giây. Khi muốn backend tự phục vụ giao diện đã build, chạy `npm run build` trong `frontend/`, sau đó mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Tài liệu API vẫn ở `/docs`.

Hướng dẫn mở dashboard, trình tự thuyết trình, cách diễn giải từng chỉ số và các giới hạn hiện tại nằm trong [bản Markdown](HUONG_DAN_SU_DUNG_DASHBOARD_TAIL_RISK.md) và [bản Word mới](HUONG_DAN_SU_DUNG_DASHBOARD_TAIL_RISK_MOI.docx). Gói mã nguồn frontend kèm hai bản hướng dẫn nằm trong `frontend_crypto_tail_risk_presentation.zip`.

Sinh lại kiểu TypeScript từ OpenAPI sau khi thay đổi API bằng `npm run api:types`. Kiểm tra giao diện bằng `npm test`; Playwright/Axe có thể chạy với `npm run test:e2e` sau khi cài browser của Playwright.

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
- Có adapter tùy chọn cho funding, open interest, basis và macro FRED; chạy lệnh bên dưới để lưu event và tạo feature riêng. Feature chỉ được dùng khi còn trong thời hạn dữ liệu và đạt ngưỡng coverage; file feature ngoại sinh không được trộn vào snapshot baseline. VIX lấy từ `VIXCLS`; `DTWEXBGS` là proxy USD diện rộng, không phải chỉ số ICE DXY. Binance chỉ cung cấp lịch sử basis/open interest gần đây nên các biến này thường bị coverage gate loại khỏi đánh giá nhiều năm.
- LSTM, GRU và Transformer đã có model code dạng tùy chọn. Cài PyTorch bằng `requirements-dl.txt` trước khi chọn các model này; chúng không được ghi nhận là đã có forecast result cho tới khi chạy xong full walk-forward.
- Bài kiểm tra chạy bằng `& $PY -m pytest -c backend\pytest.ini backend\tests --basetemp .pytest-tmp`. CI tự chạy test trên Python 3.11.
- `data/`, model files và kết quả Parquet lớn không commit. Chỉ commit code, cấu hình và báo cáo nhỏ đã rà soát.
