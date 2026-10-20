# Gói frontend thuyết trình

Thư mục `frontend/` trong gói là bản dashboard đã chỉnh để dễ trình bày trên laptop/PC. Thanh đầu trang chuyển trực tiếp giữa hai chế độ “Dự báo” và “Kết quả nghiên cứu”; chế độ nghiên cứu có lối vào tới các bảng T1–T11 và hình F1–F9, kèm phần giải thích so sánh QLIKE với HAR và ngưỡng VaR màu xanh.

Gói có mã nguồn, bản build `dist/`, hướng dẫn Word `HUONG_DAN_SU_DUNG_DASHBOARD_TAIL_RISK_MOI.docx` và bản Markdown. Frontend gọi API backend của dự án hiện tại. Sau khi giải nén, đặt thư mục `frontend/` trong thư mục dự án gốc; chạy backend từ thư mục gốc theo hướng dẫn trong file Word rồi chạy `npm run dev` bên trong `frontend/`.

Địa chỉ `127.0.0.1` dành cho máy đang chạy ứng dụng. Muốn chia sẻ dashboard qua Internet cho người xem trên máy khác, cần triển khai cả backend lẫn frontend lên máy chủ có địa chỉ công khai.
