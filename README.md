# Limit Order Book và Liquidity Stress Meter

## Chia sẻ và tái lập dữ liệu

GitHub lưu mã nguồn, cấu hình, schema, báo cáo QA và manifest checksum. Dữ liệu thị trường thô và dữ liệu đã xử lý được giữ ngoài Git để repository nhẹ và tải nhanh. Mỗi thành viên tải các file nguồn vào thư mục cục bộ, sau đó chạy cùng pipeline chuẩn hóa và replay.

File `outputs/manifests/cryptohftdata_qa.json` là manifest checksum tham chiếu cho **đúng bộ dữ liệu CryptoHFTData hiện đang có trên máy chủ dự án**. Manifest ghi đường dẫn tương đối, kích thước và SHA-256 của từng file nguồn. Sau khi tải dữ liệu, chạy `scripts/verify_cryptohftdata_snapshot.py`. Chỉ coi dữ liệu đã khớp khi lệnh xác minh thành công.

## Tái tạo đúng bộ dữ liệu hiện có

### Phạm vi bộ dữ liệu

Bộ dữ liệu baseline hiện được ghim trong manifest gồm:

- Nguồn: CryptoHFTData, Binance Spot L2 order book và trades
- Cặp giao dịch: BTCUSDT
- Ngày UTC: 2025-06-28, 2025-06-29 và 2025-06-30
- Số file dự kiến: 132 file, gồm 66 file orderbook và 66 file trades; tổng dung lượng thô khoảng 322,3 MB
- Kết quả replay dự kiến: 203.459 dòng top-of-book hợp lệ được quan sát theo từng giây

Đây là bộ dữ liệu hiện có để Backend và Frontend tích hợp. Bộ này **chưa đủ phạm vi nghiên cứu** trong `config/default.yaml`: chưa có ETHUSDT và chỉ bao phủ ba ngày. Mục tiêu sáu tháng, hai cặp vẫn là công việc thu thập dữ liệu tiếp theo. Giữ nguyên khoảng thời gian thiếu; không nội suy dữ liệu.

### Các bước trên Windows PowerShell

Với repository đã clone, cập nhật mã nguồn từ thư mục gốc:

```powershell
git pull origin main
```

Với thành viên clone mới:

```powershell
git clone https://github.com/nguynhoanggiaviethung-beep/G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter.git
Set-Location G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter
```

Từ thư mục gốc repository, tạo môi trường nếu máy chưa có. Dự án chọn Python 3.11 làm phiên bản mục tiêu:

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

Nếu đã có `.venv`, bỏ qua lệnh tạo môi trường. Nếu PowerShell không tìm thấy lệnh `py`, hãy cài/chọn Python 3.11 hoặc dùng trực tiếp đường dẫn tới Python 3.11 để tạo `.venv`.

Tải đúng phạm vi baseline hiện tại, không tải nhầm toàn bộ mục tiêu sáu tháng:

```powershell
& $PY scripts/download_cryptohftdata.py `
  --start 2025-06-28 `
  --end 2025-06-30 `
  --symbols BTCUSDT `
  --data-types orderbook trades
```

Đối chiếu từng file tải về với SHA-256 đã commit:

```powershell
& $PY scripts/verify_cryptohftdata_snapshot.py
```

Chỉ tiếp tục khi lệnh báo đủ **132/132 file khớp**. Sau đó chuẩn hóa Parquet và dựng lại replay:

```powershell
& $PY scripts/normalize_cryptohftdata.py
& $PY scripts/audit_cryptohftdata_replay.py
```

### Vị trí file sau khi tải và xử lý

Downloader lưu file thô tại:

```text
data_raw/cryptohftdata/binance_spot/YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet
data_raw/cryptohftdata/binance_spot/YYYY-MM-DD/HH/BTCUSDT_trades.parquet
data_raw/cryptohftdata/download_manifest.jsonl
```

Chuẩn hóa tạo dữ liệu trong `data_processed/cryptohftdata/` và cập nhật báo cáo `outputs/manifests/cryptohftdata_qa.json`. Replay tạo:

```text
data_processed/cryptohftdata/replay/BTCUSDT_book_top_1s.parquet
outputs/manifests/cryptohftdata_replay_qa.json
```

Các thư mục dữ liệu này được `.gitignore` loại khỏi Git. Giữ file trên máy mỗi thành viên; không dùng `git add -f` để đưa dữ liệu vào commit. Các báo cáo QA nhỏ được lưu trong Git để nhóm đối chiếu số lượng dòng, sequence gap và coverage. Verifier kiểm tra SHA-256 của file nguồn để xác nhận dữ liệu tải về giống chính xác với baseline.

### Điều kiện xác nhận tái tạo thành công

Verifier phải báo đủ 132 file và kích thước cùng SHA-256 đều khớp. Sau chuẩn hóa, QA cần ghi nhận ba ngày, BTCUSDT, 132 file (66 orderbook và 66 trades) và không có lỗi chuẩn hóa. Sau replay, đối chiếu báo cáo QA và số dòng với baseline đã commit. Hash khớp xác nhận các file nguồn giống hệt nhau; điều đó **không** có nghĩa bộ mẫu ba ngày đã đủ cho toàn bộ nghiên cứu.

Downloader gửi yêu cầu cho đủ 24 giờ mỗi ngày. Manifest baseline chỉ chứa các file nguồn thực sự có trên nhà cung cấp; giờ không có file sẽ được ghi là `missing` trong `download_manifest.jsonl` cục bộ của từng thành viên. Số lượng và checksum trong manifest đã commit xác định baseline cần tái tạo. Downloader có thể chạy tiếp sau khi bị gián đoạn và giới hạn tối đa 55 yêu cầu mỗi phút. API key miễn phí là tùy chọn; nếu nhà cung cấp cấp key, lưu trong biến môi trường cục bộ `CRYPTOHFTDATA_API_KEY`. Không commit key và không gửi key qua chat.

Nếu xác minh checksum thất bại, hãy xem thông báo của verifier và `download_manifest.jsonl`. Không ghi đè manifest tham chiếu chỉ để làm cho lần tải lỗi chuyển thành đạt. Có thể nhà cung cấp đã thay đổi hoặc gỡ file; khi đó cả nhóm cần thống nhất bộ dữ liệu thay thế rồi cập nhật manifest checksum có kiểm soát.

## Mở rộng sang phạm vi nghiên cứu đầy đủ

Phạm vi nghiên cứu dự kiến là 2025-06-28 đến 2025-12-31, gồm BTCUSDT và ETHUSDT. Phạm vi này chưa có trong manifest QA baseline đã commit. Không xem kết quả tải toàn kỳ là baseline dùng chung cho đến khi coverage được kiểm định và manifest checksum mới được rà soát, commit. Khi nhóm thống nhất thu thập kỳ đầy đủ, downloader nhận ngày bắt đầu/kết thúc, symbol và loại dữ liệu như sau:

```powershell
& $PY scripts/download_cryptohftdata.py `
  --start 2025-06-28 `
  --end 2025-12-31 `
  --symbols BTCUSDT ETHUSDT `
  --data-types orderbook trades
```

Kiểm tra dung lượng đĩa trước khi chạy. Phạm vi sáu tháng lớn hơn nhiều và có thể thiếu giờ dữ liệu. Ghi coverage thực tế theo từng symbol và loại dữ liệu; không suy luận rằng dữ liệu liên tục chỉ từ ngày bắt đầu và kết thúc được yêu cầu.

## Pilot đối chiếu Tardis riêng biệt

Pilot Tardis ngày 2025-01-01 là bộ dữ liệu đối chiếu riêng. Không trộn pilot này vào baseline nghiên cứu CryptoHFTData. `outputs/manifests/source_log.json` lưu hash của các file pilot hiện có trên máy chủ dự án, nhưng downloader CryptoHFTData không tải các file Tardis đó. Thành viên cần tái tạo pilot phải lấy riêng đúng các file này và xác minh hash theo source log.

## Chi tiết xử lý và giới hạn

Các dòng CryptoHFTData dùng `received_time` đơn vị nanosecond, `event_type` (`snapshot`/`update`), Binance update ID và khối lượng level tuyệt đối. Adapter cần:

1. Đọc từng file Parquet bằng PyArrow.
2. Khởi tạo book từ snapshot đầy đủ, áp dụng update theo sequence và đánh dấu book không hợp lệ sau gap cho đến khi có snapshot mới.
3. Chỉ chuyển trạng thái qua ranh giới file khi sequence liên tục đã được xác nhận. Một số giờ không có snapshot bên trong file.
4. Diễn giải phía chủ động của giao dịch theo `is_buyer_maker`: `true` là bên bán chủ động; `false` là bên mua chủ động.
5. Ghi giờ thiếu, sequence không hợp lệ, tuổi snapshot và hash file vào QA; không điền các khoảng thiếu bằng quan sát tự tạo.

Các file mẫu trong `data_raw/cryptohftdata_sample/` chỉ dùng để kiểm tra định dạng, không đại diện cho mẫu nghiên cứu. Không dùng chúng để huấn luyện mô hình. Muốn đủ điều kiện nghiên cứu vẫn cần coverage đã thống nhất, replay có xử lý sequence, chia train/validation/test theo thời gian và chỉ fit nhãn/ngưỡng trên tập train.

## Hợp đồng Backend

- Cấu hình: `config/default.yaml`
- Từ điển dữ liệu và quy tắc xử lý: `docs/data_dictionary.csv`, `docs/data_contract.md`
- Schema replay snapshot: `schemas/replay_snapshot.v1.example.json`
- Response khi chưa có model: `schemas/predict_unavailable.v1.example.json`
- Pipeline dữ liệu: `scripts/download_cryptohftdata.py`, `scripts/verify_cryptohftdata_snapshot.py`, `scripts/normalize_cryptohftdata.py`, `scripts/audit_cryptohftdata_replay.py`
- Script pilot cũ: `scripts/prepare_backend_pilot.py`, `scripts/compare_quotes.py`, `scripts/compare_quotes_eth.py`

Timestamp API dùng ISO-8601 UTC; giá và khối lượng dùng chuỗi decimal; giá trị chưa có trả về `null` kèm trạng thái. Không trả dự báo cũ hoặc xác suất tự đặt khi chưa có model artifact.
