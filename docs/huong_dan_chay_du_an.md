# Hướng dẫn cài đặt và chạy dự án Liquidity Stress Meter

Tài liệu này dành cho người **chưa từng cài đặt dự án lập trình**. Làm lần lượt từng bước, chép nguyên các lệnh và dán vào cửa sổ dòng lệnh là được.

Hướng dẫn có **hai cách**, chọn theo máy của bạn:

- **Cách 1 – Windows** (Windows 10 hoặc 11): xem [Phần A](#phần-a--cài-đặt-và-chạy-trên-windows).
- **Cách 2 – macOS** (MacBook, iMac, chip Intel hoặc Apple M1/M2/M3/M4): xem [Phần B](#phần-b--cài-đặt-và-chạy-trên-macos).

Sau khi chạy được, đọc [Phần C](#phần-c--sử-dụng-dashboard) để biết cách dùng dashboard, và [Phần D](#phần-d--lỗi-thường-gặp) nếu gặp lỗi.

> Dự án **chỉ đọc dữ liệu công khai** của sàn Binance. Nó không cần tài khoản, không cần API key và **không bao giờ đặt lệnh mua bán**. Đây là sản phẩm nghiên cứu, không phải khuyến nghị đầu tư.

---

## Trước khi bắt đầu: dự án gồm những gì?

Dự án có hai phần chạy cùng lúc, mỗi phần trong một cửa sổ dòng lệnh riêng:

| Phần | Vai trò | Viết bằng | Địa chỉ khi chạy |
|---|---|---|---|
| **Backend (API)** | Nhận dữ liệu sổ lệnh từ Binance và chạy mô hình dự báo | Python | http://127.0.0.1:8000 |
| **Frontend (dashboard)** | Giao diện web để xem kết quả | Node.js / React | http://127.0.0.1:5173 |

Bạn cần cài hai công cụ miễn phí: **Python** (chạy backend) và **Node.js** (chạy dashboard).

**Cần chuẩn bị:**

- Máy có kết nối Internet.
- Khoảng 6 GB ổ trống: thư viện khoảng 2 GB, model khoảng 1,7 GB, phần còn lại cho dữ liệu thu được.
- RAM từ 8 GB trở lên. Khi chạy, model 1 giây cần khoảng 2 GB RAM.
- **Thư mục model** `meter_6m` (khoảng 1,7 GB). Thư mục này **không có trên GitHub** vì quá nặng. Hãy xin người phụ trách dự án qua USB, Google Drive hoặc OneDrive. Nếu thiếu thư mục này, dashboard vẫn hiện sổ lệnh nhưng **không có xác suất dự báo**.

---

## Phần A – Cài đặt và chạy trên Windows

### A1. Cài Python

1. Vào https://www.python.org/downloads/ và tải **Python 3.12** (nút "Download Python 3.12.x"). Không dùng bản cũ hơn 3.11.
2. Mở file vừa tải về.
3. **Quan trọng:** ở màn hình đầu tiên, **đánh dấu ô "Add python.exe to PATH"** ở dưới cùng.
4. Bấm **Install Now** và chờ xong.

### A2. Cài Node.js

1. Vào https://nodejs.org/ và tải bản **LTS** (bản 20 trở lên).
2. Mở file vừa tải về, bấm **Next** liên tục rồi **Install**, giữ nguyên mọi lựa chọn mặc định.

### A3. Mở PowerShell và kiểm tra

PowerShell là cửa sổ dòng lệnh có sẵn trên Windows.

1. Bấm phím **Windows**, gõ `PowerShell`, rồi bấm **Enter**.
2. Gõ lần lượt hai lệnh sau, mỗi lệnh bấm **Enter**:

```powershell
py --version
node --version
```

Nếu thấy hiện ra số phiên bản, ví dụ `Python 3.12.7` và `v22.11.0`, là đã cài đúng. Nếu báo "không nhận ra lệnh", hãy đóng PowerShell, mở lại rồi thử lần nữa; nếu vẫn lỗi, cài lại bước A1 và nhớ đánh dấu ô "Add python.exe to PATH".

### A4. Tải mã nguồn dự án

**Cách đơn giản nhất (không cần Git):**

1. Vào https://github.com/nguynhoanggiaviethung-beep/G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter
2. Bấm nút xanh **Code**, chọn **Download ZIP**.
3. Giải nén file ZIP vào một thư mục dễ nhớ, ví dụ `D:\LOB`. Sau khi giải nén, bên trong `D:\LOB` phải thấy trực tiếp các thư mục `backend`, `frontend`, `config`… Nếu chúng nằm trong thêm một thư mục con nữa, hãy dùng đường dẫn của thư mục con đó ở các bước sau.

Nếu bạn đã quen dùng Git, có thể tải bằng lệnh `git clone` thay cho file ZIP.

### A5. Đặt thư mục model vào đúng chỗ

Chép thư mục `meter_6m` nhận được vào `backend\artifacts\`. Kết quả phải đúng như sau:

```text
D:\LOB\backend\artifacts\meter_6m\BTCUSDT\horizon_1.joblib
D:\LOB\backend\artifacts\meter_6m\BTCUSDT\horizon_5.joblib
D:\LOB\backend\artifacts\meter_6m\BTCUSDT\horizon_10.joblib
D:\LOB\backend\artifacts\meter_6m\ETHUSDT\horizon_1.joblib
... (tương tự cho ETHUSDT)
```

### A6. Cài thư viện (chỉ làm một lần)

Trong PowerShell, chạy lần lượt từng lệnh dưới đây. Ở lệnh đầu, thay `D:\LOB` bằng thư mục của bạn.

```powershell
Set-Location -LiteralPath 'D:\LOB'
```

Tạo môi trường Python riêng cho dự án:

```powershell
py -3.12 -m venv .venv
```

Cài thư viện Python. Bước này mất khoảng 3–10 phút, tùy tốc độ mạng:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Cài thư viện cho dashboard:

```powershell
Set-Location -LiteralPath 'D:\LOB\frontend'
npm ci
```

Khi chạy xong và không thấy dòng chữ `ERROR` màu đỏ là đã cài xong. Các cảnh báo màu vàng (`WARN`) có thể bỏ qua.

### A7. Chạy dự án (làm mỗi lần muốn dùng)

Bạn cần **hai cửa sổ PowerShell** mở cùng lúc.

**Cửa sổ 1 – chạy backend.** Mở một PowerShell mới rồi chạy:

```powershell
Set-Location -LiteralPath 'D:\LOB'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Đợi đến khi thấy dòng `Application startup complete`. Sau đó **để nguyên cửa sổ này**, không tắt.

**Cửa sổ 2 – chạy dashboard.** Mở thêm một PowerShell mới rồi chạy:

```powershell
Set-Location -LiteralPath 'D:\LOB\frontend'
npm run dev
```

Khi thấy dòng `Local: http://127.0.0.1:5173/`, mở trình duyệt (Chrome, Edge…) và vào **http://127.0.0.1:5173**.

**Tắt dự án:** bấm vào từng cửa sổ PowerShell rồi nhấn **Ctrl + C**.

---

## Phần B – Cài đặt và chạy trên macOS

### B1. Mở Terminal

Terminal là cửa sổ dòng lệnh có sẵn trên Mac. Nhấn **Command (⌘) + Space**, gõ `Terminal`, rồi nhấn **Return**.

### B2. Cài Homebrew

Homebrew là công cụ giúp cài phần mềm trên Mac bằng một lệnh. Nếu máy đã có Homebrew, bỏ qua bước này.

Dán lệnh sau vào Terminal rồi nhấn **Return**. Máy sẽ hỏi mật khẩu đăng nhập Mac; khi gõ mật khẩu, màn hình không hiện ký tự nào, đó là bình thường.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Khi cài xong, Terminal sẽ in ra mục **"Next steps"** gồm hai hoặc ba lệnh. Chép và chạy đúng các lệnh đó: chúng giúp Terminal nhận ra lệnh `brew`. Sau đó đóng Terminal và mở lại.

### B3. Cài Python, Node.js và thư viện OpenMP

```bash
brew install python@3.12 node libomp
```

`libomp` là thư viện mà mô hình XGBoost cần để chạy trên Mac. Nếu thiếu nó, backend sẽ báo lỗi `libomp.dylib`.

Kiểm tra lại:

```bash
python3.12 --version
node --version
```

Nếu thấy số phiên bản, ví dụ `Python 3.12.7` và `v22.11.0`, là đã cài đúng.

### B4. Tải mã nguồn dự án

1. Vào https://github.com/nguynhoanggiaviethung-beep/G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter
2. Bấm nút xanh **Code**, chọn **Download ZIP**.
3. Mở file ZIP trong thư mục **Downloads** để giải nén. Đổi tên thư mục vừa giải nén thành `LOB` rồi kéo nó vào thư mục cá nhân của bạn (thư mục có biểu tượng ngôi nhà). Đường dẫn khi đó là `~/LOB`. Bên trong `~/LOB` phải thấy trực tiếp các thư mục `backend`, `frontend`, `config`…

### B5. Đặt thư mục model vào đúng chỗ

Chép thư mục `meter_6m` nhận được vào `backend/artifacts/`. Kết quả phải đúng như sau:

```text
~/LOB/backend/artifacts/meter_6m/BTCUSDT/horizon_1.joblib
~/LOB/backend/artifacts/meter_6m/BTCUSDT/horizon_5.joblib
~/LOB/backend/artifacts/meter_6m/BTCUSDT/horizon_10.joblib
~/LOB/backend/artifacts/meter_6m/ETHUSDT/horizon_1.joblib
... (tương tự cho ETHUSDT)
```

### B6. Cài thư viện (chỉ làm một lần)

Chạy lần lượt từng lệnh trong Terminal:

```bash
cd ~/LOB
```

Tạo môi trường Python riêng cho dự án:

```bash
python3.12 -m venv .venv
```

Cài thư viện Python. Bước này mất khoảng 3–10 phút:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Cài thư viện cho dashboard:

```bash
cd ~/LOB/frontend
npm ci
```

Khi chạy xong và không thấy dòng chữ `ERROR` là đã cài xong. Các cảnh báo (`WARN`) có thể bỏ qua.

### B7. Chạy dự án (làm mỗi lần muốn dùng)

Bạn cần **hai cửa sổ Terminal**. Đang ở Terminal, nhấn **⌘ + T** để mở thêm một tab mới.

**Tab 1 – chạy backend:**

```bash
cd ~/LOB
.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Đợi đến khi thấy dòng `Application startup complete`. Sau đó **để nguyên tab này**.

**Tab 2 – chạy dashboard:**

```bash
cd ~/LOB/frontend
npm run dev
```

Khi thấy dòng `Local: http://127.0.0.1:5173/`, mở Safari hoặc Chrome và vào **http://127.0.0.1:5173**.

**Tắt dự án:** bấm vào từng tab rồi nhấn **Control + C**. Lưu ý là phím **Control**, không phải Command.

---

## Phần C – Sử dụng dashboard

Phần này giống nhau trên Windows và macOS.

### C1. Màn hình chính (LIVE)

Dashboard mở ở chế độ **LIVE**, tức theo dõi thị trường đang diễn ra:

1. Chọn cặp **BTC/USDT** hoặc **ETH/USDT**.
2. Chọn **horizon**, tức dự báo cho bao nhiêu giây tới: **1s**, **5s** hoặc **10s**. Mặc định là 5s.
3. Đợi vài giây để trạng thái chuyển sang **"đang nhận dữ liệu"** và mục **chất lượng book** báo **hợp lệ**.

Ý nghĩa các ô trên màn hình:

| Ô | Ý nghĩa |
|---|---|
| **mid-price** | Giá giữa của giá mua cao nhất và giá bán thấp nhất |
| **spread** | Khoảng cách mua–bán, càng nhỏ càng dễ giao dịch |
| **depth imbalance** | Bên mua hay bên bán đang đông hơn, từ −1 (bán áp đảo) đến +1 (mua áp đảo) |
| **order flow (OFI)** | Dòng lệnh mua/bán vừa đổ vào trong 5 giây qua |
| **hướng mid-price** | Xác suất giá **tăng (up)**, **đứng yên (flat)** hoặc **giảm (down)** sau số giây đã chọn |
| **xác suất stress** | Xác suất thị trường sắp khó giao dịch. Nếu vượt **ngưỡng** ghi bên cạnh, dashboard báo **vượt ngưỡng** |

**Lưu ý về 5 phút đầu:** một số chỉ số cần 5 phút dữ liệu liên tục mới tính đủ. Trong 5 phút đầu sau khi bật backend, hoặc sau khi mất kết nối, xác suất có thể kém chính xác hơn. Hãy đợi khoảng 5 phút trước khi đọc kết quả.

### C2. Chế độ replay (xem lại)

Bấm **replay** để xem lại sổ lệnh và dự báo ở một thời điểm trong quá khứ. Chỉ xem lại được những khoảng thời gian mà máy bạn **đã thu dữ liệu**, tức đã từng bật backend. Sàn không cho tải lại sổ lệnh của quá khứ, nên không xem được thời điểm trước lần đầu bạn chạy dự án.

### C3. Trang nghiên cứu

Bấm **nghiên cứu** ở đầu trang để xem kết quả nghiên cứu 6 tháng. Trang gồm phần tóm tắt, giải thích thuật ngữ, câu trả lời 5 câu hỏi nghiên cứu và 7 hình kèm cách đọc. Trang này không cần backend và không thay đổi theo thời gian thực.

---

## Phần D – Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Windows báo `py` hoặc `python` không được nhận ra | Cài lại Python (bước A1), **đánh dấu "Add python.exe to PATH"**, rồi mở PowerShell mới |
| Windows báo "running scripts is disabled" khi chạy `npm` | Chạy lệnh `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` trong PowerShell, chọn **Y**, rồi thử lại |
| macOS báo `command not found: brew` | Chạy các lệnh trong mục "Next steps" mà bước B2 đã in ra, rồi mở Terminal mới |
| macOS báo lỗi có chữ `libomp.dylib` | Chạy `brew install libomp`, rồi bật lại backend |
| `npm ci` báo lỗi `ENOENT ... package.json` | Bạn đang ở sai thư mục. Hãy chuyển vào thư mục `frontend` trước, như ở bước A6 hoặc B6 |
| Trình duyệt báo "không thể truy cập trang" | Kiểm tra cửa sổ dashboard (`npm run dev`) còn chạy và không báo lỗi |
| Dashboard báo không kết nối được API | Kiểm tra cửa sổ backend còn chạy. Mở http://127.0.0.1:8000/health; nếu thấy `"status":"ok"` là backend đang chạy |
| Có sổ lệnh nhưng ô xác suất ghi **"— chưa có xác suất"** và mục **model artifact** báo **"chưa có"** | Nếu mới bật backend dưới 1 phút thì đợi thêm. Nếu vẫn vậy, thư mục model đang thiếu hoặc đặt sai chỗ: kiểm tra lại bước A5 hoặc B5, rồi tắt và bật lại backend |
| Báo lỗi `address already in use` hoặc cổng 8000/5173 đã bị dùng | Dự án đang chạy ở một cửa sổ khác. Hãy tìm và tắt cửa sổ đó bằng Ctrl + C, hoặc khởi động lại máy |
| Trạng thái **"đang đồng bộ"** hoặc **"đang kết nối lại"** kéo dài | Kiểm tra mạng. Backend tự kết nối lại; đợi thêm 1–2 phút |
| Lần đầu chọn **1s** thấy chậm vài giây | Bình thường. Model 1 giây khá nặng (khoảng 0,9 GB) nên lần đầu cần thời gian nạp |

Có thể kiểm tra trực tiếp tình trạng model tại http://127.0.0.1:8000/api/v1/models khi backend đang chạy. Mỗi dòng ghi `"status":"ready"` là một model đã sẵn sàng.

---

## Phần E – Dành cho người muốn tìm hiểu thêm (không bắt buộc)

Phần này không cần thiết để dùng dashboard.

**Dữ liệu được lưu ở đâu?** Khi backend chạy, nó tự ghi dữ liệu nhận được vào các thư mục `data_raw/` và `data_interim/` bên trong thư mục dự án. Mỗi giờ chạy tốn khoảng 20 MB. Muốn giải phóng ổ đĩa, có thể xóa các thư mục này khi backend đã tắt.

**Đưa dữ liệu vừa thu vào chế độ replay.** Ví dụ với BTC trên Windows; trên macOS thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python` và dấu `\` bằng `/`:

```powershell
Set-Location -LiteralPath 'D:\LOB'
$EVENTS = Read-Host 'Dán đường dẫn thư mục data_interim\binance_spot\captures\<mã capture>\events\symbol=BTCUSDT'
.\.venv\Scripts\python.exe backend\scripts\replay_cryptohftdata.py --input $EVENTS --symbol BTCUSDT
.\.venv\Scripts\python.exe backend\scripts\validate_lob_data.py --symbol BTCUSDT
.\.venv\Scripts\python.exe backend\scripts\build_lob_classification_dataset.py --symbol BTCUSDT
```

**Kiểm tra mã nguồn có chạy đúng không:**

```powershell
Set-Location -LiteralPath 'D:\LOB\backend'
..\.venv\Scripts\python.exe -m pytest -q
```

Trên macOS:

```bash
cd ~/LOB/backend && ../.venv/bin/python -m pytest -q
```

**Chạy lại nghiên cứu.** Việc này cần bộ dữ liệu archive 6 tháng (hàng chục GB, không có trên GitHub) và mất nhiều giờ. Các lệnh nằm trong `README.md`, mục *Reproduce the six-month study*. **Không chạy lại bước chấm điểm test**: test chỉ được chấm một lần.

## Tài liệu liên quan

- [README và sơ đồ dự án](../README.md)
- [Báo cáo nghiên cứu](../paper/bao_cao_nghien_cuu.md)
- [Thiết kế nghiên cứu](research_design.md)
- [Từ điển dữ liệu](data_dictionary.md)
