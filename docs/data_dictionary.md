# Từ điển dữ liệu LOB-only

Thời gian lưu bằng UTC. Binance giá/quantity bắt đầu ở dạng decimal string trong raw feed; book map dùng decimal chính xác. Các feature numeric là float khi xuất Parquet.

## Raw JSONL

Vị trí: `data/raw/lob/<symbol>/date=YYYY-MM-DD/events-NNNN.jsonl`.

| Field | Ý nghĩa |
|---|---|
| `source` | `binance_spot_ws_diff_depth` hoặc `binance_spot_rest_depth_snapshot`. |
| `symbol` | Symbol Binance Spot, ví dụ BTCUSDT. |
| `stream` | Tên diff-depth stream hoặc REST route snapshot. |
| `connection_id` | ID duy nhất cho mỗi lần kết nối WebSocket. |
| `collector_version` | Phiên bản logic ghi/đồng bộ collector. |
| `event_time_utc` | Event time `E` của depth update; không có với REST snapshot. |
| `transaction_time_utc` | Transaction time nếu event có trường đó; Spot diff-depth hiện có thể không cung cấp. |
| `received_at_utc` | Thời gian máy local nhận WebSocket event. |
| `snapshot_received_at_utc` | Thời gian nhận REST snapshot. |
| `payload` | Raw snapshot/update JSON, gồm sequence ID và các thay đổi bid/ask. |

## Sampled snapshot và feature

Vị trí: `data/processed/lob_snapshots/symbol=<symbol>/date=<day>/` và `data/features/lob/symbol=<symbol>/date=<day>/`.

| Field | Ý nghĩa |
|---|---|
| `symbol` | Tài sản của sample. |
| `sample_time_utc` | Clock time khi trạng thái book được lấy mẫu. |
| `event_time_utc` | Event time của update mới nhất đã áp dụng. |
| `received_at_utc` | Local receive time của update mới nhất đã áp dụng. |
| `last_update_id` | `u` cuối cùng đã áp dụng trong local book. |
| `book_valid` | True nếu snapshot/update sequence đồng bộ, book không crossed/empty và feed còn mới. |
| `quality_flags` | Danh sách nguyên nhân invalid/stale khi có. |
| `event_age_ms` | Chênh lệch sample/receive-event theo milliseconds. |
| `best_bid`, `best_bid_qty` | Giá và lượng bid tốt nhất. |
| `best_ask`, `best_ask_qty` | Giá và lượng ask tốt nhất. |
| `mid_price` | Trung bình best bid/ask, dùng làm proxy. |
| `absolute_spread` | Ask trừ bid. |
| `relative_spread_bps` | Spread chia mid-price, nhân 10.000. |
| `bid_depth_qty_N`, `ask_depth_qty_N` | Tổng quantity top N levels; N=5,10,20. |
| `bid_depth_notional_N`, `ask_depth_notional_N` | Tổng `price * quantity` top N levels. |
| `obi_qty_N`, `obi_notional_N` | Imbalance signed trong [-1,1] theo quantity hoặc notional. |
| `microprice` | Giá bid/ask trọng số theo quantity phía đối diện. |
| `microprice_minus_mid_bps` | Microprice offset so với mid-price theo bps. |
| `depth_profile_slope_bid_20`, `depth_profile_slope_ask_20` | Slope của tỷ trọng quantity theo khoảng cách giá top-20, khoảng cách chuẩn hóa theo bps. |
| `*_lag_1`, `*_change` | Feature lag hoặc sai khác với mẫu trước. |
| `*_mean_N`, `*_std_N`, `*_q50_N`, `*_q95_N` | Rolling mean/std/median/95th percentile trên tối đa N mẫu trước; N=60,300,3600. |
| `mid_return` | Log-return giữa mid hiện tại và mẫu trước. |
| `past_rv_N` | Tổng bình phương return đã quan sát trong tối đa N mẫu gần nhất. |

Snapshot Parquet giữ `bid_levels_top_20_json` và `ask_levels_top_20_json`; raw event stream vẫn được giữ riêng để audit/replay.

## Tóm tắt LOB theo ngày cho dự báo dài hạn

Vị trí: `data/features/lob_daily/symbol=<symbol>/daily_summary.jsonl`. Mỗi dòng chỉ được ghi sau khi một ngày UTC hoàn tất và đạt ít nhất 99,5% coverage, không có khoảng trống mẫu lớn hơn 5 giây.

| Field | Ý nghĩa |
|---|---|
| `date_utc` | Ngày UTC được tổng hợp. |
| `sample_count`, `coverage` | Số mẫu LOB hợp lệ và tỷ lệ so với số mẫu dự kiến trong ngày. |
| `daily_rv` | Tổng bình phương `mid_return` trong ngày, dùng làm target volatility. |
| `daily_return` | Tổng log-return mid-price trong ngày. |
| `lob_features` | Trung bình ngày của spread, microprice offset, OBI và depth top-N. |

Các dòng này giúp nạp lại lịch sử dự báo sau restart; chúng không chứa OHLCV hoặc giao dịch.

## Dataset targets

Vị trí: `data/features/lob_datasets/symbol=<symbol>/horizon_samples=<H>.parquet`.

| Field | Ý nghĩa |
|---|---|
| `target_return` | `ln(mid_(t+H)/mid_t)` từ mid-price LOB. |
| `target_rv` | Tổng bình phương mid-price returns trong H mẫu tương lai. |
| `target_abs_loss` | `max(-target_return, 0)` theo đơn vị return. |
| `target_end_utc` | Timestamp cuối target horizon. |
| `target_valid` | Target chỉ true khi book liên tục, hợp lệ và đúng clock grid hết horizon. |

Không đưa `target_*`, symbol, timestamp hoặc quality metadata vào feature matrix. `target_end_utc` bắt buộc không sớm hơn forecast origin; train row chỉ dùng khi target đã mature tại origin.

## Collector status và manifest

`GET /api/v1/collector/status` trả trạng thái kết nối, book validity, update ID, event rate, latency, reconnect/resync count, sample count, valid-sample ratio, disk-free bytes và lỗi gần nhất. Manifest `data/state/lob/<symbol>/manifest.jsonl` bổ sung artifact path, date/time range, row count, SHA-256 và collector version.
