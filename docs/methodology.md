# Phương pháp xử lý và đánh giá LOB

## Thu nhận và local book

WebSocket diff-depth được mở trước REST snapshot để không bỏ qua các update trong lúc khởi tạo. Mọi event được append raw JSONL cùng symbol, stream, connection ID, collector version, event time và receive time trước khi xử lý.

REST snapshot ưu tiên Binance Market Data Only domain; khi request thất bại, collector thử tiếp các official Spot REST hosts đã cấu hình. Request có timeout ngắn, retry có giới hạn và chẩn đoán hiển thị endpoint cuối đã lỗi. Snapshot mặc định 1.000 levels để giảm REST weight; đây vẫn đủ cho top-20 levels đang trình bày.

Collector bỏ event có `u <= snapshot.lastUpdateId`. Snapshot chỉ được chấp nhận khi sự kiện tiếp theo có dải update ID chứa snapshot ID. Với book đã khởi tạo, event có `u <= lastUpdateId` là stale và bị bỏ; event có `U > lastUpdateId + 1` là sequence gap. Gap làm book invalid và kích hoạt reconnect/resnapshot; không tiếp tục phát feature từ trạng thái chưa đồng bộ. Các level được thay quantity theo event; quantity bằng 0 xóa level. Giá/quantity lưu bằng Decimal để tránh làm sai key do binary float.

## Storage và provenance

- Raw diff-depth và REST snapshot: append-only JSONL theo symbol/ngày, chia file khi đạt giới hạn dung lượng.
- Sampled top-20 local book: Parquet theo symbol/ngày.
- Feature: Parquet theo symbol/ngày.
- Manifest JSONL: loại artifact, path, row count, time range, SHA-256, connection/collector version và số hàng book-valid.
- Checkpoint không thay thế raw stream; khởi động mới vẫn đồng bộ bằng snapshot mới.

Feature/snapshot Parquet ghi từng phần để giảm số file. Snapshot đã xử lý giữ top 20 mỗi phía; raw events và manifest là nguồn để kiểm tra/replay sâu hơn.

## Feature

- `mid_price = (best_bid + best_ask) / 2`.
- `absolute_spread = best_ask - best_bid`; `relative_spread_bps = absolute_spread / mid_price * 10,000`.
- `bid_depth_N` và `ask_depth_N`: tổng khối lượng trong top N mức giá, với N=5, 10, 20. Notional depth cộng `price * quantity`.
- `OBI_N = (bid_depth_N - ask_depth_N) / (bid_depth_N + ask_depth_N)`. Nếu mẫu số bằng 0, OBI bằng 0.
- `microprice = (best_ask * best_bid_qty + best_bid * best_ask_qty) / (best_bid_qty + best_ask_qty)`. Nếu tổng quantity bằng 0, dùng mid-price và ghi trạng thái chất lượng.
- Microprice offset lưu theo bps so với mid. Depth concentration lưu tỷ lệ quantity ở top 5 trên tổng quantity top N. Depth profile slope hồi quy tỷ trọng quantity từng mức theo khoảng cách giá chuẩn hóa theo bps.
- Lag/rolling mean, standard deviation, median và quantile 95 chỉ dựa trên các mẫu trước origin; mid-price return và trailing RV dùng dữ liệu đã quan sát tới sample hiện tại.

## Target và leakage

Target được căn theo thời gian lấy mẫu cố định. Với horizon H, các return từ `t+1` đến `t+H` dùng để tính forward RV và cumulative log-return. Hàng target chỉ hợp lệ nếu tất cả state trong đường đi hợp lệ, timestamp cách nhau đúng sampling interval và không có gap.

`target_end_utc` là timestamp cuối horizon. Trong walk-forward, hàng chỉ được train nếu `target_end_utc <= origin_time_utc`. Feature matrix loại symbol/time/quality metadata và toàn bộ target/future/label columns. Chronological split giữ thứ tự; imputer/scaler chỉ fit trên train của từng refit.

## Mô hình và score

- Benchmark RV persistence dùng RV của cửa sổ quá khứ cùng độ dài horizon (nếu chưa có cột đúng horizon trong một fixture ngắn, dùng trailing 60 giây ngoại suy theo `H/60`).
- HAR-RV hồi quy log-RV tương lai theo RV quá khứ 1 phút, 5 phút và 1 giờ; đây là benchmark tuyến tính truyền thống.
- Ridge và histogram gradient boosting hồi quy log-RV, dùng imputer và preprocessing được fit trên training rows đã mature tại origin.
- Linear quantile regression dự báo quantile return điều kiện từ LOB features; ES ước lượng bằng tail mean lịch sử với loss vượt VaR.
- VaR/ES baseline: quantile loss distribution của target returns quá khứ; VaR là positive loss quantile, ES là trung bình tail loss tại/vượt VaR.
- Volatility: MAE, RMSE, QLIKE (`log(forecast) + actual/forecast`); forecast phải dương.
- VaR: pinball loss, breach rate và Kupiec unconditional coverage.
- VaR/ES đồng thời: FZ0 score; score nhỏ hơn tốt hơn khi cùng mẫu/horizon/confidence.
- Risk horizons dùng các origin không chồng lấn để hạn chế phụ thuộc cơ học giữa target windows.

Mọi kết quả được gắn với symbol, horizon, clock interval, tập feature, OOS rows và nguồn LOB. Mid-price proxy, giới hạn snapshot depth, gap, sample validity và độ dài lịch sử phải nêu khi báo cáo.

## Dự báo trực tuyến nhiều horizon

Dashboard giữ nhóm theo dõi RV ngắn hạn 1 phút, 5 phút và 1 giờ từ clock samples một giây. Nhóm này được so sánh với dự báo dài hạn thay vì đại diện cho dự báo xa.

Nhóm dài hạn gộp các mẫu LOB hợp lệ thành ngày UTC. Một ngày yêu cầu coverage tối thiểu 99,5% và không có khoảng trống clock lớn hơn 5 giây; ngày không đạt điều kiện bị loại và không được nối target qua gap. RV ngày là tổng bình phương log-return mid-price từ book. Đặc trưng LOB theo ngày gồm spread, microprice offset, OBI và depth. Forecast origin đặt ở cuối ngày đã hoàn tất; target là tổng RV trong 1, 7 hoặc 30 ngày UTC kế tiếp.

Rolling mean theo RV ngày là benchmark truyền thống. HAR-RV hồi quy log RV tương lai theo RV ngày gần nhất và trung bình RV 7/30 ngày; Ridge và histogram gradient boosting bổ sung đặc trưng book theo ngày. Nhãn chỉ được thêm vào tập huấn luyện sau khi toàn bộ target horizon hoàn tất; supervised models đợi tối thiểu 30 nhãn mature (xấp xỉ 31/43/89 ngày dữ liệu liên tục cho horizon 1/7/30 ngày, gồm lookback trước forecast origin đầu tiên). Tóm tắt ngày nhỏ gọn được lưu riêng và replay khi backend khởi động lại. QLIKE dùng forecast được phát hành tại mỗi daily origin; các cửa sổ 7/30 ngày chồng lấn nên điểm số có phụ thuộc thời gian và chỉ mang tính mô tả.

Reconnect/stale gap xóa cửa sổ ngắn hạn, accumulator ngày hiện tại và hàng đợi target chưa mature; dữ liệu ngày mature trước gap giữ trong tệp tóm tắt nhưng chuỗi liên tục bắt đầu lại. QLIKE không chấm forecast in-sample. VaR/ES lịch sử ngắn hạn cần tối thiểu 200/1.000 cửa sổ cho mức 95%/99%.
