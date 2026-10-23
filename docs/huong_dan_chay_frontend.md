# Hướng dẫn chạy frontend Liquidity Stress Meter

Tài liệu này hướng dẫn mở dashboard chạy trên Windows. Dashboard mặc định hiển thị **Binance Spot live liên tục**, có chế độ replay để xem dữ liệu đã capture, chỉ đọc và không đặt lệnh.

## 1. Mở đúng thư mục dự án

Mở PowerShell tại thư mục dự án. Đường dẫn hiện tại của dự án:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter'
```

Kiểm tra PowerShell đang ở đúng chỗ và thư mục frontend có `package.json`:

```powershell
Get-Location
Test-Path .\frontend\package.json
```

Dòng thứ hai cần trả về `True`. Nếu kết quả là `False`, hãy chuyển về thư mục dự án trước khi chạy lệnh npm. Lỗi `ENOENT ... package.json` thường có nghĩa là npm đang chạy ở sai thư mục.

## 2. Khởi động API backend

Mở **PowerShell thứ nhất** tại thư mục gốc dự án, rồi chạy:

```powershell
$PY = '.\.venv\Scripts\python.exe'
& $PY -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Giữ cửa sổ này mở. Khi API khởi động, backend tự mở các luồng public market-data cho BTCUSDT và ETHUSDT, đồng bộ sổ lệnh, rồi ghi capture liên tục. Không cần API key và không đặt lệnh. Khi dừng backend bằng `Ctrl+C`, capture hiện tại được đóng và lưu manifest. Tài liệu API ở:

<http://127.0.0.1:8000/docs>

Nếu báo không tìm thấy `.venv\Scripts\python.exe`, hãy tạo môi trường và cài dependency từ thư mục gốc:

```powershell
py -m venv .venv
$PY = '.\.venv\Scripts\python.exe'
& $PY -m pip install -r requirements.txt
```

Sau đó chạy lại lệnh uvicorn ở trên.

## 3. Khởi động frontend

Mở **PowerShell thứ hai** tại thư mục gốc dự án. Chuyển vào đúng thư mục frontend:

```powershell
Set-Location -LiteralPath 'D:\Gói 1\G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter\frontend'
```

Cài các package lần đầu hoặc sau khi `package-lock.json` thay đổi, rồi chạy Vite:

```powershell
npm ci
npm run dev
```

Khi terminal hiện địa chỉ local, mở dashboard tại:

<http://127.0.0.1:5173>

Lần sau, nếu package đã cài rồi, chỉ cần chạy `npm run dev`. Giữ cả hai cửa sổ PowerShell (backend và frontend) mở trong lúc sử dụng. Dừng từng server bằng `Ctrl+C`.

## 4. Dùng dashboard live liên tục

1. Mở <http://127.0.0.1:5173>. Dashboard tự vào chế độ **LIVE** và nhận cập nhật liên tục từ backend.
2. Chọn `BTC/USDT` hoặc `ETH/USDT`, rồi horizon `1s`, `5s` hoặc `10s`.
3. Chờ backend đồng bộ book. Nếu kết nối bị gián đoạn hoặc sequence bị lỗi, dashboard báo trạng thái và collector tự kết nối/đồng bộ lại.
4. Các xác suất chỉ khả dụng khi có model tương thích và đủ dữ liệu rolling liên tục (tối đa 5 phút). Sau khi mất kết nối hoặc dựng lại book, thời gian warm-up bắt đầu lại.
5. Chọn chế độ **replay** nếu muốn xem dữ liệu đã lưu. Khoảng ngày replay chỉ áp dụng cho dữ liệu đã capture/replay, không thể tạo lại L2 lịch sử bằng cách bật live.

Dashboard sử dụng luồng SSE qua proxy Vite tại `127.0.0.1:8000`; trình duyệt tự kết nối lại nếu mất SSE. Không cần tạo file `.env` hay sửa `VITE_API_ROOT` cho cách chạy mặc định này.

## 5. Nếu dashboard chưa có dữ liệu

Chế độ live cần backend còn chạy và có kết nối Internet tới Binance Spot. Luồng live giữ dữ liệu trong bộ nhớ cho dashboard và ghi source events dưới `data_raw/binance_spot/captures/` cùng normalized events dưới `data_interim/binance_spot/captures/`. Nếu muốn dùng capture đó trong chế độ replay, tạo grid chuẩn hóa bằng các bước sau từ thư mục gốc dự án:

```powershell
$PY = '.\.venv\Scripts\python.exe'
$EVENTS = Read-Host 'Dán đường dẫn thư mục chuẩn hóa kết thúc bằng events\symbol=BTCUSDT'
& $PY backend\scripts\replay_cryptohftdata.py --input $EVENTS --symbol BTCUSDT
& $PY backend\scripts\validate_lob_data.py --symbol BTCUSDT
& $PY backend\scripts\build_lob_classification_dataset.py --symbol BTCUSDT
```

Lấy đường dẫn event từ dòng `interim_event_root` mà collector in ra, rồi chọn thư mục con tương ứng kết thúc bằng `events\symbol=BTCUSDT`. Sau khi replay xong, tải lại dashboard. Với ETHUSDT, thay symbol trong các lệnh bằng `ETHUSDT` và dùng thư mục event của ETHUSDT.

Capture ngắn đủ để xác nhận luồng và replay; nó không đủ cho nghiên cứu walk-forward. Xác suất live dùng artifact đã lưu và chỉ được hiển thị sau khi đủ cửa sổ feature; các kết quả live không được xem là bằng chứng hiệu năng nghiên cứu.

## 6. Xử lý các thông báo thường gặp

| Thông báo / hiện tượng | Cách xử lý |
| --- | --- |
| `Cannot find path ...\frontend` | Chạy `Get-Location`. Nếu đang ở thư mục gốc dự án, dùng `Set-Location .\frontend`; nếu không, chuyển về đúng đường dẫn dự án ở mục 1 rồi thử lại. |
| `npm error ENOENT ... package.json` | Đang chạy npm ở sai thư mục. Chạy `Set-Location ...\frontend`, xác nhận `Test-Path .\package.json` trả về `True`, rồi chạy lại npm. |
| Không mở được `127.0.0.1:5173` | Kiểm tra terminal frontend còn chạy và không có lỗi khởi động. |
| Dashboard hiện “đang kết nối Binance” | Kiểm tra terminal backend còn chạy và máy có Internet; collector tự thử các endpoint WebSocket dự phòng. |
| Dashboard hiện “đang đồng bộ sổ lệnh” hoặc “đang kết nối lại” | Chờ backend nhận snapshot REST khớp với chuỗi update. Nếu Binance mất kết nối, collector tự thử lại. |
| Dashboard chưa có xác suất | Chờ đủ tối đa 5 phút dữ liệu live liên tục; nếu tiếp tục unavailable, kiểm tra artifact cho cặp và horizon trong `/api/v1/models`. |
| Lỗi kết nối API hoặc không tải được dữ liệu | Kiểm tra terminal backend còn chạy; mở `http://127.0.0.1:8000/docs` để xác nhận API hoạt động. |
| `No valid book at or before the requested time` | Thời điểm đã chọn nằm trước dữ liệu đầu tiên hoặc ngoài khoảng replay. Xóa thời điểm để nạp snapshot mới nhất, hoặc chọn một thời điểm trong khoảng quan sát hiển thị. |
| `0 dòng`, `chưa có dữ liệu` hoặc không có snapshot | Capture mới chỉ tạo dữ liệu thô/interim. Hoàn tất bước replay và kiểm tra dữ liệu ở mục 5, rồi tải lại trang. |
| Model artifact chưa có / xác suất chưa khả dụng | Đây là trạng thái dự kiến khi chưa có model tương thích. Thu thập đủ dữ liệu theo thiết kế, sau đó mới chạy quy trình huấn luyện; dashboard không tự tạo xác suất demo. |

