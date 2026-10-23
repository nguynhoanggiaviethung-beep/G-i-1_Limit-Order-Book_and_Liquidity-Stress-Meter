# Hướng dẫn chạy dự án LOB & Liquidity Stress Meter

> Tài liệu này dành cho Windows và PowerShell. Dự án gồm API Python, dashboard React và các script thu thập, replay, kiểm tra dữ liệu sổ lệnh. Dashboard chỉ đọc dữ liệu, không đặt lệnh.

## 1. Yêu cầu trước khi chạy

- Python 3.11 trở lên.
- Node.js 20 trở lên, kèm npm.
- Kết nối Internet nếu muốn dùng luồng Binance Spot live.
- Mở PowerShell tại thư mục dự án:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter'
```

Kiểm tra đang ở đúng thư mục:

```powershell
Get-Location
Test-Path .\backend\app\main.py
Test-Path .\frontend\package.json
```

Hai lệnh `Test-Path` cần trả về `True`.

## 2. Cài môi trường lần đầu

Chỉ cần làm bước này một lần, hoặc làm lại khi dependency thay đổi.

```powershell
py -3.11 -m venv .venv
$PY = '.\.venv\Scripts\python.exe'
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

Cài dependency cho giao diện:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter\frontend'
npm ci
```

Nếu máy đã có `.venv` hoặc thư mục `frontend\node_modules`, không cần tạo/cài lại; chuyển sang bước chạy.

## 3. Chạy API backend

Mở **PowerShell thứ nhất**, chuyển về thư mục gốc dự án rồi chạy:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter'
$PY = '.\.venv\Scripts\python.exe'
& $PY -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Giữ cửa sổ này mở. Khi API khởi động, backend bắt đầu nhận dữ liệu công khai từ Binance Spot cho BTCUSDT và ETHUSDT, đồng bộ sổ lệnh và ghi capture. Không cần API key. Backend có thể cần một lúc để đồng bộ snapshot ban đầu.

- Tài liệu API: <http://127.0.0.1:8000/docs>
- Kiểm tra API: <http://127.0.0.1:8000/health>

## 4. Chạy dashboard frontend

Mở **PowerShell thứ hai**:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter\frontend'
npm run dev
```

Nếu chưa cài package frontend, chạy `npm ci` trước `npm run dev`. Mở địa chỉ:

<http://127.0.0.1:5173>

Giữ cả hai cửa sổ backend và frontend mở. Dashboard sử dụng API ở cổng 8000 thông qua proxy Vite. Dừng từng chương trình bằng `Ctrl+C` trong cửa sổ tương ứng.

## 5. Sử dụng dashboard

### Theo dõi live

1. Dashboard mở ở chế độ **LIVE** và nhận cập nhật liên tục khi backend còn chạy.
2. Chọn BTC/USDT hoặc ETH/USDT và horizon 1s, 5s hoặc 10s.
3. Chờ trạng thái book chuyển sang đã đồng bộ. Sau khi mất kết nối hoặc đồng bộ lại sổ lệnh, một số feature cần tối đa khoảng 5 phút dữ liệu liên tục để khởi động lại.
4. Nếu chưa có xác suất, mở <http://127.0.0.1:8000/docs> hoặc endpoint `/api/v1/models` để kiểm tra model tương thích.

### Xem kết quả nghiên cứu

Chọn **nghiên cứu** trên thanh đầu trang. Phần đầu trang tóm tắt câu trả lời cho RQ1–RQ5; phía dưới là các hình validation, holdout cổ điển và DeepLOB cùng giải thích cách đọc, số liệu và ý nghĩa nghiên cứu. Chọn **dashboard** để quay lại màn hình live/replay.

Đây là kết quả pilot đã lưu, không cập nhật theo luồng live. DeepLOB chỉ có kết quả validation. Holdout cổ điển đã được chấm một lần; không chạy lại bước chấm holdout khi chuẩn bị báo cáo.

### Xem replay

Chuyển chế độ sang **replay**, chọn cặp và mốc thời gian nằm trong khoảng dữ liệu đã replay. Replay hiển thị book đã lưu; nó không tải lại L2 lịch sử từ Binance. Nếu ngày hoặc giờ nằm ngoài capture, dashboard sẽ báo không có book hợp lệ.

## 6. Tạo dữ liệu replay từ capture mới (tùy chọn)

API live tự lưu dữ liệu thô và event chuẩn hóa tại:

```text
data_raw\binance_spot\captures\<capture-id>
data_interim\binance_spot\captures\<capture-id>\events
```

Có thể tạo capture riêng trong một PowerShell khác. Ví dụ thu BTCUSDT trong 10 phút:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter'
$PY = '.\.venv\Scripts\python.exe'
& $PY backend\scripts\collect_binance_lob.py --symbol BTCUSDT --duration-seconds 600
```

Khi lệnh kết thúc, lấy thư mục event từ dòng `Normalized replay events` hoặc `interim_event_root` mà collector in ra. Đường dẫn cần dùng là thư mục con kết thúc bằng `events\symbol=BTCUSDT`, ví dụ:

```text
D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter\data_interim\binance_spot\captures\20261022T081338862305Z\events\symbol=BTCUSDT
```

Tên capture ở ví dụ chỉ minh họa; hãy thay bằng thư mục thực tế collector vừa tạo. Sau đó replay, kiểm tra chất lượng và tạo features:

```powershell
$SYMBOL = 'BTCUSDT'
$EVENTS = Read-Host 'Dán đường dẫn đầy đủ đến thư mục events\symbol=BTCUSDT'
& $PY backend\scripts\replay_cryptohftdata.py --input $EVENTS --symbol $SYMBOL
& $PY backend\scripts\validate_lob_data.py --symbol $SYMBOL
& $PY backend\scripts\build_lob_classification_dataset.py --symbol $SYMBOL
```

Với ETH, đổi `$SYMBOL` thành `ETHUSDT` và dán đường dẫn đến thư mục `events\symbol=ETHUSDT`. Sau khi hoàn tất, chọn một mốc nằm trong khoảng dữ liệu và tải lại dashboard.

Capture 10 phút chỉ để xác nhận luồng và replay; không đủ làm dữ liệu nghiên cứu hoặc huấn luyện đánh giá nghiêm túc. Binance REST aggregate trades không chứa lịch sử độ sâu L2 nên không thể dùng để dựng lại order book quá khứ.

## 7. Các thư mục dữ liệu chính

```text
data_raw/       event gốc/capture bất biến
data_interim/   event đã chuẩn hóa, trước replay
data_processed/ grid book một giây và features
outputs/        báo cáo chất lượng, metrics, biểu đồ và manifest
```

Dữ liệu archive phục vụ nghiên cứu pilot tách biệt với dữ liệu replay/live của dashboard. Dashboard tiêu chuẩn đọc các grid trong `data_processed/lob/`; không tự đọc archive nghiên cứu để thay cho capture/replay.

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Cannot find path ...\frontend` | Chạy `Get-Location`, chuyển về thư mục gốc dự án, rồi vào `frontend`. Kiểm tra `Test-Path .\frontend\package.json` trả về `True`. |
| `npm error ENOENT ... package.json` | npm đang chạy sai thư mục. Chạy `Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter\frontend'` rồi thử lại. |
| Không mở được cổng 5173 | Kiểm tra PowerShell frontend còn chạy và Vite không báo lỗi. Dùng URL được in trong terminal nếu cổng bị đổi. |
| Dashboard không kết nối API | Kiểm tra cửa sổ backend còn chạy; mở `/health` và `/docs` ở cổng 8000. |
| `đang đồng bộ` hoặc `đang kết nối lại` | Chờ snapshot Binance khớp với chuỗi cập nhật. Collector sẽ tự thử kết nối và đồng bộ lại. |
| Chưa có snapshot hoặc `No valid book at or before the requested time` | Chọn thời điểm trong khoảng replay có dữ liệu. Capture live không thể tạo lại book L2 cho thời gian trước lúc capture. |
| Có book nhưng chưa có xác suất | Chờ feature rolling warm-up; kiểm tra `/api/v1/models` để xác nhận có artifact đúng cặp và horizon. |

## 9. Lệnh kiểm tra frontend

Từ thư mục gốc dự án, có thể kiểm tra type và tạo bản build frontend:

```powershell
npm --prefix frontend run build
```

Lệnh này không khởi động dashboard; để dùng giao diện trong lúc phát triển, chạy `npm run dev` tại thư mục `frontend`.

## Tài liệu liên quan

- [Hướng dẫn chạy frontend](huong_dan_chay_frontend.md)
- [Thiết kế nghiên cứu và các mốc dữ liệu](research_design.md)
- [Từ điển dữ liệu](data_dictionary.md)
- [README và sơ đồ dự án](../README.md)
