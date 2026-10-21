# Thiết kế nghiên cứu LOB-only

## Mục tiêu

Nghiên cứu xem trạng thái và biến động của Limit Order Book (LOB) có giúp dự báo biến động và rủi ro đuôi của mid-price proxy hay không. Feature, target, lựa chọn mô hình và đánh giá đều dùng dữ liệu LOB đã được tái tạo hợp lệ.

## Phạm vi nguồn

- Binance Spot REST `GET /api/v3/depth` tạo snapshot khởi đầu.
- Binance Spot WebSocket `<symbol>@depth@100ms` cung cấp diff-depth updates.
- Symbol ban đầu là BTCUSDT và ETHUSDT; danh sách lấy từ `config/symbols.yaml`.
- Không dùng OHLCV, kline, trade, aggTrade, funding, open interest hay dữ liệu ngoài.
- Lịch sử bắt đầu từ thời gian chạy collector. REST depth không cung cấp lịch sử sổ lệnh.

## Đồng bộ book và chất lượng

Mỗi symbol có kết nối, snapshot, book map, sequence và reconnect riêng. Collector buffer WebSocket events rồi lấy snapshot. Event đầu tiên được áp dụng phải bao phủ update ID của snapshot; event cũ bị loại. Nếu update range bắt đầu sau `lastUpdateId + 1`, book mất hợp lệ, không phát feature, và collector khởi động lại chu trình bằng một snapshot mới.

Snapshot REST được giới hạn tối đa 5.000 mức mỗi phía. Các mức ngoài snapshot không được biết cho tới khi có update. Giữ event time, transaction time khi nguồn cung cấp, local receive time, connection ID và phiên bản collector. Feed cũ, book rỗng/crossed, lỗi sequence và disconnect đều phải hiện trong status/quality flags.

## Sampling và target horizon

Mặc định lấy mẫu theo clock mỗi giây, không giả định stream nhất thiết phát event ở mỗi mốc. Một mẫu chỉ được lưu khi book đồng bộ và event chưa stale. `config/settings.yaml` quy định chu kỳ sampling, thời gian stale tối đa, flush và horizons.

Mid-price tại mẫu `t`:

`mid_t = (best_bid_t + best_ask_t) / 2`

Log-return giữa các mẫu liên tiếp:

`r_t = ln(mid_t / mid_(t-1))`

RV mục tiêu tại origin `t` trên `H` bước lấy mẫu phía trước:

`RV_(t,H) = sum(r_(t+i)^2), i=1..H`

Return mục tiêu là `ln(mid_(t+H) / mid_t)`. H mặc định 60, 300 và 3.600 giây. Nếu thiếu mẫu trên grid hoặc book invalid ở horizon tương lai, target được để invalid; không nội suy hay điền giá.

Dashboard còn có dự báo dài hạn 1, 7 và 30 ngày. Nhóm này gộp RV mid-price và đặc trưng spread/OBI/depth/microprice từ LOB theo ngày UTC. Một ngày chỉ hợp lệ khi coverage ít nhất 99,5% và khoảng trống mẫu không quá 5 giây. Nhãn là tổng RV của các ngày tương lai và chỉ được chấm sau khi đủ toàn bộ horizon; không dùng OHLCV hay giao dịch.

VaR/ES đo tổn thất `-return` của mid-price proxy. Chúng không được diễn giải là risk theo execution price.

## Feature

Nhóm feature gồm top-of-book và spread; tổng quantity/notional depth ở 5/10/20 mức; OBI quantity/notional; microprice; depth concentration/profile slope; lag, thay đổi và rolling mean/std/quantiles; mid-price return và trailing RV. Công thức chi tiết nằm trong [methodology.md](methodology.md).

Feature được ghi với sample timestamp, event/receive time, sequence ID, thời gian trễ, trạng thái book và quality flags. Target columns không được đưa vào feature matrix.

## Chia tập và walk-forward

Các mẫu sắp theo clock timestamp, không shuffle. Mỗi origin chỉ được train trên nhãn có `target_end_utc <= origin_time_utc`. Imputer, scaler và Ridge fit lại trên phần quá khứ tại mỗi lần refit; không fit trên OOS. Mặc định phần cuối theo thời gian là OOS; số training tối thiểu và tần suất refit lấy từ cấu hình.

Volatility báo MAE, RMSE và QLIKE. Baseline RV là persistence trên trailing RV; mô hình hồi quy gồm Ridge và histogram gradient boosting. VaR/ES gồm empirical historical baseline và linear quantile regression từ LOB features. ES dùng tail mean từ training losses tại/beyond VaR. Với risk horizons lớn hơn một sample, backtest giảm mẫu OOS về origin không chồng lấn trước khi tính coverage/score.

Kết quả chỉ được xuất nếu có đủ history/label. Nếu chưa đạt ngưỡng, báo `insufficient_*`; không thay bằng dữ liệu OHLCV hay sample tổng hợp.
