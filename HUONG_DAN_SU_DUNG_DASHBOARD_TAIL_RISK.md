# Hướng dẫn sử dụng Dashboard Crypto Tail Risk

Dashboard có hai chế độ: **Dự báo** để xem dữ liệu Binance và ước lượng hiện tại; **Kết quả nghiên cứu** để trình bày các bảng, biểu đồ và kiểm định đã chạy trên dữ liệu lịch sử. Hai chế độ dùng hai nguồn dữ liệu khác nhau, vì vậy không nên gọi kết quả backtest là dự báo trực tiếp.

## 1. Mở dashboard trên máy tính

Dashboard hiện chạy trên máy đang mở VS Code. Cần có kết nối Internet để backend nhận dữ liệu Binance.

### Chạy API

Mở terminal tại thư mục gốc dự án và chạy:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

Giữ terminal này mở. Kiểm tra API tại <http://127.0.0.1:8000/docs>; trạng thái đồng bộ dữ liệu tại <http://127.0.0.1:8000/api/v1/sync-status>.

### Chạy giao diện

Mở terminal thứ hai tại thư mục gốc dự án. Chỉ chạy `npm ci` lần đầu cài frontend hoặc khi danh sách thư viện thay đổi:

```powershell
cd frontend
npm ci
npm run dev
```

Mở <http://127.0.0.1:5173/>. Giữ cả hai terminal mở trong lúc trình bày. Dừng bằng `Ctrl+C`. Địa chỉ `127.0.0.1` chỉ dùng được trên máy đang chạy ứng dụng; muốn người xem kết nối từ máy khác cần triển khai cả backend lẫn frontend lên máy chủ có địa chỉ công khai.

## 2. Chọn chế độ trình bày

Ở thanh trên cùng, chọn một trong hai mục:

- **Dự báo:** giá Binance cập nhật trực tiếp; nến hoàn chỉnh và dự báo được làm mới theo chu kỳ 5 phút. Màn hình cho chọn Bitcoin hoặc Ethereum, kỳ hạn 1/7/30 ngày, mức tin cậy và số tiền để dễ hình dung quy mô dao động.
- **Kết quả nghiên cứu:** các bảng và biểu đồ được tính từ dữ liệu lịch sử đã lưu. Thanh điều hướng bên trái mở đủ các phần Tổng quan, Dự báo, Đánh giá mô hình, Rủi ro và backtest, Trạng thái và chiến lược, Phương pháp và dữ liệu.

Khi mở **Kết quả nghiên cứu**, bắt đầu từ trang Tổng quan và đi theo bốn liên kết trên màn hình:

1. **So sánh mô hình — T3–T6.** T3 là bảng chính. HAR là mô hình tham chiếu truyền thống; QLIKE thấp hơn nghĩa là sai số dự báo biến động thấp hơn trên tập kiểm tra. Giao diện nêu mô hình tốt nhất so với HAR khi cùng số kỳ đánh giá. T4 kiểm tra từng mô hình với mốc random walk và hiệu chỉnh khi kiểm tra nhiều mô hình; T5 cho biết nhóm mô hình có chất lượng tương đương tốt nhất; T6 kiểm tra độ hiệu chỉnh của dự báo.
2. **Kiểm tra rủi ro — T7–T8.** T7 so sánh số lần lỗ vượt VaR thực tế với số lần dự kiến theo mức tin cậy. T8 kiểm tra ước lượng Expected Shortfall ở phần đuôi xấu.
3. **Trạng thái và chiến lược — T9–T10.** T9 trình bày kết quả theo trạng thái thị trường; T10 so sánh các chiến lược kinh tế được đánh giá.
4. **Dữ liệu và phương pháp — T1–T2, T11 và F1–F9.** Xem mô tả dữ liệu, thống kê lợi suất/RV, kiểm tra độ bền và toàn bộ hình nghiên cứu.

Các bảng hiển thị theo tài sản và kỳ hạn đang chọn. Có thể tải dữ liệu bảng bằng nút tải CSV. Kỳ hạn 30 ngày chỉ mang tính mô tả vì các cửa sổ dự báo chồng lấn làm giảm số quan sát độc lập; các kết luận kiểm định tương ứng được ẩn.

## 3. Đọc bốn chỉ số ở chế độ Dự báo

Các lời giải thích cạnh từng con số tự cập nhật theo tài sản, kỳ hạn, mức tin cậy và số tiền đang nhập.

### Dao động dự báo · sigma

Sigma cho biết quy mô dao động tham chiếu theo cả chiều tăng và giảm. Ví dụ `±2%` nghĩa là mức dao động ước lượng khoảng 2% trong kỳ đã chọn. Số tiền bên cạnh chỉ giúp hình dung quy mô tương ứng; đây không phải dự báo lợi nhuận hay mức lỗ tối đa.

### Ngưỡng lỗ có thể bị vượt · VaR

VaR là một ngưỡng lỗ tham chiếu. Mức tin cậy 95% tương ứng khoảng 5 kỳ trên 100 kỳ tương tự có thể lỗ nhiều hơn; mức 99% tương ứng khoảng 1 kỳ trên 100 trong dài hạn. Đây không phải lịch cố định và cũng không giới hạn khoản lỗ lớn nhất.

### Lỗ trung bình trong nhóm kỳ xấu · ES

Expected Shortfall ước lượng mức lỗ trung bình trong nhóm kỳ tệ nhất ở mức tin cậy đang chọn. Ví dụ mức 95% nhìn vào khoảng 5% kỳ có kết quả xấu nhất. Đây là ước lượng, không phải khoản lỗ chắc chắn.

### Điểm biến động gốc · RV

RV là phương sai dự báo, có đơn vị phần trăm bình phương. Mô hình dùng số này để tính các chỉ số biến động; không đọc nó như phần trăm lợi nhuận hoặc số tiền có thể lỗ.

## 4. Đọc biểu đồ VaR

Cột thể hiện lợi suất thực tế: phần trên mức 0 là ngày tăng, phần dưới là ngày giảm. **Đường xanh đứt** là ngưỡng lỗ VaR; đường nằm dưới mức 0 để có thể so trực tiếp với lợi suất âm. Dấu **×** đánh dấu ngày lỗ thực tế vượt ngưỡng. Chuỗi này là kết quả walk-forward trong quá khứ, không phải kết quả dự báo của hôm nay.

## 5. Giới hạn cần nêu khi thuyết trình

- VaR/ES ở chế độ Dự báo hiện dùng xấp xỉ phân phối chuẩn. Do giả định này có thể đánh giá thấp biến động cực đoan, dashboard chưa quy đổi VaR/ES production sang tiền; số tiền được quy đổi cho sigma.
- Kết quả FHS trong T7/T8 thuộc bài kiểm tra nghiên cứu, không phải phương pháp VaR/ES production đang dùng.
- Dự báo 30 ngày chỉ mang tính mô tả vì các cửa sổ chồng lấn làm giảm số mẫu độc lập.
- Giá trực tiếp, dữ liệu và dự báo phụ thuộc vào Internet, Binance và backend trên máy đang chạy dự án.
- Dashboard trình bày ước lượng thống kê, không phải lời khuyên mua bán hay cam kết về giá.

## 6. Khi dashboard không hiện dữ liệu

- Kiểm tra terminal API còn chạy và mở được <http://127.0.0.1:8000/docs>.
- Kiểm tra terminal frontend còn chạy và địa chỉ đang mở là <http://127.0.0.1:5173/>.
- Nếu Binance mất kết nối, kiểm tra Internet rồi xem trạng thái tại <http://127.0.0.1:8000/api/v1/sync-status>.
- Nếu bảng nghiên cứu trống, kiểm tra dữ liệu nghiên cứu đã được tạo theo pipeline trong README. Dữ liệu Binance hiện tại và kết quả backtest lịch sử là hai nguồn riêng.
