# Tài liệu LOB cũ đã ngừng áp dụng

File này từng chứa quy trình từ các phiên bản dự án trước, bao gồm CryptoHFTData, OHLCV và các lệnh nghiên cứu không còn phù hợp với backend hiện tại. Không dùng các lệnh hoặc đường dẫn cũ trong nội dung lịch sử để chạy pipeline.

Thiết kế hiện hành là Binance Spot REST depth snapshot + WebSocket diff-depth, local book có sequence validation và feature/target LOB-only. Hướng dẫn cập nhật nằm trong [README.md](../README.md), [research_design.md](research_design.md), [methodology.md](methodology.md) và [data_dictionary.md](data_dictionary.md).

Các dữ liệu/kết quả OHLCV cũ trên máy được giữ nguyên để không mất công việc trước đây, nhưng collector, API và dashboard LOB hiện tại không đọc chúng.
