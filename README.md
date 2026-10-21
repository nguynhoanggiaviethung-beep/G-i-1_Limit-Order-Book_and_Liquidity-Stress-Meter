# Crypto Limit Order Book and Liquidity Risk

Backend nghiên cứu Binance Spot theo hướng **LOB-only** cho BTCUSDT và ETHUSDT. Dữ liệu đầu vào của feature/model chỉ đến từ REST depth snapshot và WebSocket diff-depth updates. Không dùng OHLCV, nến 5 phút, trade hay aggTrade làm đầu vào.

Mid-price được tính từ best bid/ask trong local book đã đồng bộ. Vì vậy các mục tiêu RV, VaR và ES là proxy dựa trên mid-price LOB; chúng không đại diện giá khớp lệnh độc lập.

## Giới hạn lịch sử

Binance Spot REST depth trả snapshot book hiện tại, không cung cấp lịch sử LOB nhiều năm. Collector bắt đầu tạo lịch sử từ lúc được chạy. Các file OHLCV và kết quả tail-risk cũ còn trên máy được giữ nguyên nhưng backend LOB không đọc chúng. Không nội suy khoảng trống sequence hay thời gian thiếu mẫu.

## Thành phần chính

- `backend/app/ingestion/`: REST snapshot, WebSocket diff-depth và collector độc lập theo symbol.
- `backend/app/orderbook/`: căn chỉnh snapshot/update ID, cập nhật bid/ask, phát hiện gap và kiểm tra book.
- `backend/app/storage/`: JSONL raw append-only, Parquet snapshot/feature và manifest SHA-256.
- `backend/app/features/`: spread, top-N depth, OBI, microprice, lag/rolling features và target từ mid-price.
- `backend/app/datasets/`: dataset theo thời gian và kiểm tra target availability/leakage.
- `backend/app/models/` và `backend/app/evaluation/`: baseline persistence/Ridge, VaR/ES lịch sử và đánh giá walk-forward.
- `backend/app/api/routes/lob.py`: API đọc trạng thái collector, local book và feature mới nhất.
- `frontend/src/LOBApp.tsx`: dashboard desktop theo dõi LOB trực tiếp.
- `config/symbols.yaml`, `config/settings.yaml`: cặp giao dịch, sampling, storage và horizon.

## Cài đặt

Chạy trong PowerShell ở thư mục gốc project:

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

Không cần Binance API key để đọc market data công khai. REST snapshot thử market-data-only host trước, sau đó luân phiên qua các official Spot REST hosts đã cấu hình nếu endpoint hiện tại không phản hồi. Lệnh sau kiểm tra REST snapshot và nhận ít nhất một diff-depth event cho từng symbol:

```powershell
& $PY backend\scripts\test_connection.py
```

## Chạy collector và API

Khởi động API bằng Uvicorn. Khi server chạy, collector tự kết nối và lưu dữ liệu cho từng symbol; lỗi một symbol không dừng symbol còn lại.

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Mở `http://127.0.0.1:8000/docs` để xem API. Các endpoint đọc dữ liệu:

- `GET /health` — trạng thái backend và collector.
- `GET /api/v1/collector/status` — kết nối, event rate, độ trễ, update ID, reconnect/resync, book-valid time và dung lượng trống.
- `GET /api/v1/lob/latest?symbol=BTCUSDT` — top 20 bid/ask gần nhất khi book còn hợp lệ và mới.
- `GET /api/v1/features/latest?symbol=BTCUSDT` — feature LOB mới nhất cùng quality flags.
- `GET /api/v1/forecasts/latest?symbol=BTCUSDT` — gồm nhóm theo dõi 1 phút/5 phút/1 giờ và dự báo biến động dài hạn 1/7/30 ngày; so sánh benchmark truyền thống với mô hình LOB và chấm QLIKE khi nhãn tương lai đã hoàn tất.

WebSocket diff-depth được nhận liên tục; cấu hình sampling hiện lưu một mẫu theo clock mỗi giây. Dashboard đọc API mỗi 1 giây khi feed ổn định, và giãn nhịp kiểm tra khi đang kết nối lại. Nếu chưa có book hợp lệ, endpoint book/features trả 503 thay vì đưa dữ liệu stale ra như dữ liệu mới; forecast endpoint vẫn trả số mẫu và lỗi kết nối để giải thích trạng thái.

Có thể chạy collector riêng không kèm API bằng:

```powershell
& $PY backend\scripts\collect_lob.py
```

Chạy **một trong hai** cách thu thập trên một thư mục dữ liệu tại cùng thời điểm, tránh ghi hai collector song song.

## Chạy giao diện

Mở terminal thứ hai. Nếu terminal hiện tại đã ở trong `frontend`, bỏ qua `cd frontend`.

```powershell
cd frontend
npm ci
npm run dev
```

Mở `http://127.0.0.1:5173`. Giữ cả API/collector và Vite đang chạy. Nếu đã đứng ở thư mục `frontend`, không chạy `cd frontend` lần nữa.

## Kiểm tra dữ liệu và tạo kết quả nghiên cứu

Raw event được append theo `data/raw/lob/<symbol>/date=YYYY-MM-DD/`. Feature Parquet và sampled top-20 snapshots được partition theo symbol/ngày. Manifest ở `data/state/lob/<symbol>/manifest.jsonl` ghi path, time range, row count, checksum, collector version và thống kê book hợp lệ.

```powershell
& $PY backend\scripts\validate_lob.py
& $PY backend\scripts\build_lob_dataset.py
& $PY backend\scripts\run_lob_evaluation.py
```

Nhóm theo dõi intraday tạo target 60 giây, 300 giây và 3.600 giây từ mẫu một giây. RV tương lai là tổng bình phương log-return mid-price trong đúng horizon. HAR-RV và Ridge/Gradient Boosting được huấn luyện tiền trình sau ít nhất 60 nhãn hoàn tất, không chồng lấn.

Nhóm dài hạn gộp mid-price returns và đặc trưng spread/imbalance/depth/microprice thành ngày UTC. Một ngày chỉ được dùng khi có ít nhất 99,5% mẫu dự kiến và không có khoảng trống lớn hơn 5 giây. Dự báo 1/7/30 ngày là RV tích lũy tương lai, không phải giá mục tiêu. Rolling mean là benchmark; HAR-RV dùng RV ngày 1/7/30, Ridge và Gradient Boosting bổ sung trung bình đặc trưng LOB ngày. Dự báo dài hạn chốt theo ngày UTC hoàn chỉnh gần nhất. Các nhãn huấn luyện chỉ mature sau khi toàn bộ kỳ hạn tương lai hoàn tất; cần khoảng 31/43/89 ngày LOB liên tục để có 30 nhãn mature cho horizon 1/7/30 ngày, gồm lookback trước forecast đầu tiên. Các ngày tóm tắt được lưu tại `data/features/lob_daily/` để khôi phục warm-up sau khi khởi động lại. QLIKE 7/30 ngày dùng origin hằng ngày có cửa sổ chồng lấn nên chỉ mang tính mô tả.

VaR/ES lịch sử ngắn hạn chỉ hiển thị khi có đủ các cửa sổ hoàn tất (200 mẫu cho VaR 95%, 1.000 mẫu cho VaR 99%) để tránh diễn giải đuôi phân phối từ quá ít quan sát. Walk-forward chỉ dùng nhãn đã kết thúc trước origin; các khoảng invalid/gap không được tạo target.

Kết quả nằm ở `results/metrics/lob/`, `results/forecasts/lob/` và QA ở `results/metrics/lob_data_quality.json`. Nếu dữ liệu hoặc OOS tail sample chưa đủ, script ghi trạng thái thiếu dữ liệu thay vì tạo kết quả giả. Mốc nghiên cứu lịch sử sẽ bắt đầu từ khi collector chạy; không tuyên bố đây là backtest nhiều năm.

## Sampling, sequence và độ phủ book

Collector mở WebSocket trước, buffer diff-depth events, lấy REST snapshot, bỏ event cũ và chỉ phát hành book sau event đầu tiên bao phủ `lastUpdateId`. Update bị stale được bỏ; nếu `U > lastUpdateId + 1`, book bị invalid và collector đồng bộ lại bằng snapshot. Quantity bằng 0 xóa mức giá.

Binance REST hỗ trợ snapshot tối đa 5.000 mức mỗi phía; cấu hình hiện lấy 100 mức để giảm độ trễ và REST weight, vẫn đủ cho top-20 levels dashboard hiển thị. Nếu snapshot trả về sau update stream, collector đổi REST host rồi thử đồng bộ lại, thay vì lặp lại endpoint cũ. Khi WebSocket lỗi, lần reconnect kế tiếp đổi qua host stream dự phòng. Các mức sâu hơn snapshot ban đầu không được biết cho đến khi Binance gửi thay đổi mức đó; nghiên cứu phải giới hạn diễn giải độ sâu theo những mức đã được quan sát. Mọi timestamp lưu UTC gồm event time và local receive time riêng biệt.

Các quy tắc sequence dựa trên [tài liệu WebSocket Streams chính thức của Binance](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md); REST snapshot theo [Spot REST API](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md).

## Không commit dữ liệu thị trường

Raw JSONL, Parquet, checkpoint, logs và kết quả lớn nằm ngoài Git qua `.gitignore`. Chỉ commit code, cấu hình và tài liệu đã rà soát. Có thể xem quy ước field tại [data_dictionary.md](docs/data_dictionary.md), định nghĩa mục tiêu tại [methodology.md](docs/methodology.md) và phạm vi nghiên cứu tại [research_design.md](docs/research_design.md).

Các script/module OHLCV và tài liệu Tail-Risk dashboard cũ còn trong workspace để lưu dấu công việc trước đây; các lệnh pipeline OHLCV đã bị tắt, và chúng không được API LOB hiện hành gọi. Entry frontend hiện tại là `frontend/src/LOBApp.tsx`; các trang dashboard cũ không được nạp.
