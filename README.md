# Limit Order Book và Liquidity Stress Meter

## Crypto volatility / tail-risk backend (Code Plan v3)

The current backend also implements the frozen crypto volatility and tail-risk study for Binance Spot **BTCUSDT** and **ETHUSDT**, using 5-minute candles and direct 1/7/30-day forecasts. Research is frozen at `2026-09-30T23:59:59Z`; the live 5-minute updater is a separate input stream and does not mutate a frozen snapshot. See [research_design.md](docs/research_design.md) and [data_dictionary.md](docs/data_dictionary.md) for definitions and limits.

Run from the repository root in PowerShell after installing `backend/requirements.txt` in `.venv`:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY backend\scripts\freeze_snapshot.py
& $PY backend\scripts\validate_data.py
& $PY backend\scripts\build_features.py
& $PY backend\scripts\run_walkforward.py
& $PY backend\scripts\make_report_tables.py
& $PY backend\scripts\run_risk_eval.py
& $PY -m uvicorn app.main:app --app-dir backend --reload
```

The API docs are at `http://127.0.0.1:8000/docs`. A research forecast request uses `symbol`, `horizon_days` (`1`, `7`, or `30`), and a saved `model` (for example `random_walk` or `har`). See `/api/v1/forecasts`, `/api/v1/evaluation/{symbol}`, and `/api/v1/regimes/{symbol}`. Forecasts and metrics must be generated locally from the snapshot before these read endpoints return results.

## Trạng thái và phạm vi dữ liệu

Phạm vi dùng để phát triển và kiểm định theo định hướng nghiên cứu là **2025-06-28 đến hết 2025-12-31 UTC**, hai cặp **BTCUSDT và ETHUSDT**, dữ liệu Binance Spot L2 (order book) và giao dịch (trades). Đây là khoảng thời gian yêu cầu; coverage thực tế phụ thuộc những giờ mà nhà cung cấp có file. Không coi dữ liệu đã đầy đủ chỉ vì lệnh tải kết thúc thành công.

Dữ liệu thô và dữ liệu đã xử lý không được lưu trong Git. GitHub chia sẻ mã nguồn, schema, cấu hình và các báo cáo QA/manifest nhỏ. Mỗi máy chạy Backend cần có dữ liệu cục bộ ở đúng cấu trúc bên dưới. Frontend chỉ gọi API Backend theo hợp đồng; không tự tải, đọc hay dựng lại Parquet.

> Bộ BTCUSDT ba ngày (2025-06-28–2025-06-30, 132 file) là **smoke test cũ**, không phải bộ nghiên cứu sáu tháng và không đủ cho kết luận nghiên cứu. Manifest checksum hiện có thể vẫn chỉ mô tả bộ mẫu đó cho đến khi data owner hoàn tất và công bố manifest sáu tháng đã kiểm định.

## Quy trình chuẩn cho data owner / Backend

Các lệnh sau chạy trong Windows PowerShell, từ thư mục gốc repository.

### 1. Lấy mã nguồn và dựng môi trường

```powershell
git pull origin main
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

Nếu `.venv` đã tồn tại thì bỏ qua lệnh tạo môi trường. Dự án yêu cầu Python 3.11; nếu máy không có `py -3.11`, cài Python 3.11 hoặc dùng đúng executable Python 3.11 để tạo môi trường.

### 2. Tải toàn bộ khoảng nghiên cứu

```powershell
& $PY scripts/download_cryptohftdata.py `
  --start 2025-06-28 `
  --end 2025-12-31 `
  --symbols BTCUSDT ETHUSDT `
  --data-types orderbook trades
```

Downloader có thể chạy nhiều giờ, giới hạn tốc độ và tiếp tục được sau khi gián đoạn. Cần dung lượng trống đáng kể; không commit `data_raw/`. Chờ tiến trình kết thúc, rồi xem `data_raw/cryptohftdata/download_manifest.jsonl`: phân biệt rõ `downloaded`, `missing` và `error`. File bị nhà cung cấp thiếu không được tự tạo hoặc nội suy.

### 3. Chuẩn hóa và kiểm tra nguồn

```powershell
& $PY scripts/normalize_cryptohftdata.py
```

Nguồn lưu tại `data_raw/cryptohftdata/binance_spot/YYYY-MM-DD/HH/`. Pipeline chuẩn hóa ghi vào `data_processed/cryptohftdata/` và tạo QA/checksum tại `outputs/manifests/cryptohftdata_qa.json`. Rà QA theo ngày, symbol, loại dữ liệu, số file, lỗi schema và SHA-256; đối chiếu với `download_manifest.jsonl`. Việc chuẩn hóa **cập nhật file QA manifest đã theo dõi trong Git**: data owner phải review coverage và checksum trước khi commit manifest mới. Không thay manifest tham chiếu bằng một lần tải chưa kiểm định.

Khi data owner đã commit manifest đầy đủ, các thành viên khác chạy:

```powershell
& $PY scripts/verify_cryptohftdata_snapshot.py
```

Lệnh xác minh phải thành công với manifest của bộ dữ liệu đã được nhóm duyệt. Manifest cũ 132 file chỉ xác minh bộ smoke test; nó không chứng minh coverage sáu tháng. Không dùng `--allow-extra` để tuyên bố dữ liệu đầy đủ: tùy chọn đó chỉ cho phép file ngoài tập hash tham chiếu.

### 4. Replay riêng từng symbol

Không replay BTCUSDT và ETHUSDT trong cùng một book. Chạy riêng để trạng thái order book, sequence ID, output và QA không trộn lẫn:

```powershell
& $PY scripts/audit_cryptohftdata_replay.py `
  --symbol BTCUSDT `
  --output data_processed/cryptohftdata/replay `
  --qa outputs/manifests/cryptohftdata_replay_BTCUSDT_qa.json

& $PY scripts/audit_cryptohftdata_replay.py `
  --symbol ETHUSDT `
  --output data_processed/cryptohftdata/replay `
  --qa outputs/manifests/cryptohftdata_replay_ETHUSDT_qa.json
```

Replay chỉ xuất các giây có top-of-book hợp lệ và không lấp khoảng trống. Kiểm tra `observed_dates_utc`, `invalid_intervals`, `sequence_gaps`, `snapshot_recoveries` và thời điểm bắt đầu/kết thúc trong từng QA. Kết quả là top-of-book 1 giây, chưa phải bộ feature/label cuối cùng và cũng chưa đủ để tuyên bố dữ liệu nghiên cứu hoàn chỉnh. Cần xử lý full-depth features, nhãn, chia tập theo thời gian và kiểm soát leakage theo code plan trước khi huấn luyện/đánh giá mô hình.

### 5. Bàn giao cho Frontend và các thành viên khác

- Data owner/Backend gửi lên Git **code, schema, cấu hình, QA manifest đã duyệt**, không gửi Parquet/CSV dữ liệu thô.
- Sau `git pull`, máy chạy Backend tải dữ liệu theo đúng phạm vi đã chốt, xác minh checksum và chạy pipeline như trên. `git pull` không tải dữ liệu thị trường.
- Backend đọc file cục bộ dưới `data_processed/` và phục vụ API. Frontend dùng các endpoint/schema trong `schemas/`; Frontend không cần quyền truy cập `data_raw/`.
- Backend không trả dự báo giả hoặc dữ liệu cũ khi chưa có model/output hợp lệ; giữ trạng thái thiếu dữ liệu và lỗi QA trong API response.

## Cấu trúc dữ liệu cục bộ

```text
data_raw/cryptohftdata/binance_spot/YYYY-MM-DD/HH/{BTCUSDT,ETHUSDT}_{orderbook,trades}.parquet
data_raw/cryptohftdata/download_manifest.jsonl
data_processed/cryptohftdata/...
data_processed/cryptohftdata/replay/BTCUSDT_book_top_1s.parquet
data_processed/cryptohftdata/replay/ETHUSDT_book_top_1s.parquet
outputs/manifests/cryptohftdata_qa.json
outputs/manifests/cryptohftdata_replay_BTCUSDT_qa.json
outputs/manifests/cryptohftdata_replay_ETHUSDT_qa.json
```

Các thư mục raw/processed bị `.gitignore` loại khỏi Git. Không dùng `git add -f` để đưa dữ liệu lớn vào commit.

## Smoke test nhanh (không thay thế dữ liệu nghiên cứu)

Nếu chỉ muốn thử downloader và pipeline trước, tải đúng ba ngày BTCUSDT:

```powershell
& $PY scripts/download_cryptohftdata.py `
  --start 2025-06-28 `
  --end 2025-06-30 `
  --symbols BTCUSDT `
  --data-types orderbook trades
& $PY scripts/verify_cryptohftdata_snapshot.py
& $PY scripts/normalize_cryptohftdata.py
& $PY scripts/audit_cryptohftdata_replay.py --symbol BTCUSDT
```

Baseline smoke test tham chiếu có 132 file nguồn (66 orderbook, 66 trades), khoảng 322,3 MB và 203.459 giây top-of-book hợp lệ trong replay đã ghi nhận. Khoảng này có sequence gap chưa phục hồi ở cuối dữ liệu; con số chỉ mô tả snapshot baseline, không phải cam kết lần tải mới luôn cho đúng cùng số dòng.

## Pilot Tardis riêng biệt

Pilot Tardis ngày 2025-01-01 là nguồn đối chiếu riêng; không trộn vào phạm vi nghiên cứu CryptoHFTData. `outputs/manifests/source_log.json` ghi hash các file pilot đã có trong snapshot dự án. Downloader CryptoHFTData không tải pilot Tardis.

## Quy tắc dữ liệu và hợp đồng Backend

- Cấu hình: `config/default.yaml`.
- Từ điển và quy tắc xử lý: `docs/data_dictionary.csv`, `docs/data_contract.md`.
- Schema replay: `schemas/replay_snapshot.v1.example.json`.
- Response khi chưa có model: `schemas/predict_unavailable.v1.example.json`.
- Pipeline: `scripts/download_cryptohftdata.py`, `scripts/verify_cryptohftdata_snapshot.py`, `scripts/normalize_cryptohftdata.py`, `scripts/audit_cryptohftdata_replay.py`.
- Dùng timestamp ISO-8601 UTC; giá/khối lượng dạng decimal; giá trị không có trả `null` kèm trạng thái.
- CryptoHFTData dùng `received_time` nanosecond, snapshot/update Binance và update ID. Khi sequence gap xuất hiện, book không hợp lệ cho đến snapshot hoàn chỉnh mới. Giao dịch diễn giải qua `is_buyer_maker`: `true` là bên bán chủ động, `false` là bên mua chủ động.
- File mẫu trong `data_raw/cryptohftdata_sample/` chỉ kiểm tra định dạng, không đại diện mẫu nghiên cứu và không dùng huấn luyện.

## Backend nghiên cứu Crypto Volatility & Tail Risk

Backend nghiên cứu được đặt trong `backend/` theo code plan: Binance Spot public API → đồng bộ Parquet → validation → features → models/evaluation → FastAPI. Nến 5 phút cho BTCUSDT và ETHUSDT được cấu hình từ `config/settings.yaml`, với ngày bắt đầu mặc định 2019-01-01. Endpoint market data công khai không cần API key.

Trong PowerShell, từ thư mục repository:

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install -r backend\requirements.txt
& $PY backend\scripts\sync_data.py
& $PY backend\scripts\validate_data.py
& $PY backend\scripts\build_features.py
& $PY backend\scripts\train_models.py
& $PY -m uvicorn app.main:app --app-dir backend --reload
```

Trong terminal PowerShell khác, lấy dự báo GARCH 1 ngày:

```powershell
$body = @{ symbol = "BTCUSDT"; horizon_minutes = 1440; confidence_levels = @(0.95, 0.99); model = "garch" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/forecasts -ContentType "application/json" -Body $body
```

Lần chạy đầu tải lịch sử từ ngày cấu hình; các lần chạy sau nối dữ liệu từ nến cuối và hợp nhất theo khóa symbol, interval, open time. Dữ liệu gốc và chuẩn hóa nằm trong `data/raw/` và `data/processed/`; feature RV theo ngày nằm trong `data/features/`. `validate_data.py` ghi coverage và các khoảng thiếu vào `results/metrics/data_quality.json`. `train_models.py` dùng feature RV tại cùng mốc giờ UTC mỗi ngày để dự báo RV trọn ngày kế tiếp; script đánh giá rolling-mean, HAR-RV và Random Forest trên 20% cuối theo thời gian, fit GARCH(1,1) cho RV, VaR và ES. Forecast mới nhất nằm trong `results/forecasts/latest_forecasts.json`; chỉ số ngoài mẫu nằm trong `results/metrics/volatility_backtest.json`. Các tệp dữ liệu cục bộ bị Git bỏ qua. FastAPI cung cấp `/health` và dữ liệu đã lưu qua `/api/v1/market-data/{symbol}`. POST `/api/v1/forecasts` trả dự báo đã tính, GET `/api/v1/evaluation/{symbol}` trả chỉ số ngoài mẫu; horizon hiện hỗ trợ là 1 ngày (1440 phút).

Khi FastAPI đang chạy, tự động đồng bộ Binance được bật mặc định: server đồng bộ ngay khi khởi động, sau đó canh mỗi 5 phút và chờ thêm 10 giây để nến vừa đóng sẵn sàng. Quá trình cập nhật cả dữ liệu Parquet lẫn feature RV theo ngày. Xem tình trạng lần đồng bộ gần nhất tại `http://127.0.0.1:8000/api/v1/sync-status`; xem lịch sử nến đã đóng tại `http://127.0.0.1:8000/api/v1/market-data/BTCUSDT?limit=10`.

Giá và nến đang hình thành được nhận riêng qua Binance Spot WebSocket, với cập nhật thường xuyên trong lúc cây nến 5 phút còn mở. Xem BTC tại `http://127.0.0.1:8000/api/v1/live-market-data/BTCUSDT` hoặc cả hai cặp tại `http://127.0.0.1:8000/api/v1/live-market-data`. Trường `candle.price` là giá gần nhất; `candle.is_closed` cho biết nến đã hoàn tất chưa; `last_message_at_utc` giúp kiểm tra luồng còn nhận dữ liệu. Nếu kết nối bị ngắt, backend tự thử kết nối lại. Giữ cửa sổ chạy Uvicorn mở để luồng live và lịch đồng bộ tiếp tục hoạt động. Việc đồng bộ không tự huấn luyện lại các mô hình; chạy `train_models.py` khi muốn làm mới dự báo và đánh giá.




