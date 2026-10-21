import { useEffect, useState } from "react";
import { Activity, BookOpen, CircleAlert, Database, RefreshCw, ShieldCheck } from "lucide-react";

type SymbolCode = "BTCUSDT" | "ETHUSDT";
type CollectorSymbolStatus = {
  symbol: string;
  status: string;
  book_valid: boolean;
  quality_flags: string[];
  last_event_time_utc: string | null;
  last_received_at_utc: string | null;
  last_sample_time_utc: string | null;
  last_update_id: number | null;
  event_rate_per_second: number;
  latency_ms: number | null;
  reconnect_count: number;
  resync_count: number;
  sample_count: number;
  expected_sample_count: number;
  valid_sample_count: number;
  book_valid_seconds: number;
  book_valid_ratio: number;
  book_valid_time_ratio: number;
  disk_free_bytes: number | null;
  last_error: string | null;
};
type CollectorStatus = {
  status: string;
  provider: string;
  sampling_interval_seconds: number;
  stale_after_seconds: number;
  symbol_status: Record<string, CollectorSymbolStatus>;
};
type FeatureRow = {
  symbol: string;
  sample_time_utc: string;
  event_time_utc: string | null;
  received_at_utc: string;
  event_age_ms: number;
  last_update_id: number;
  book_valid: boolean;
  quality_flags: string[];
  best_bid: number;
  best_bid_qty: number;
  best_ask: number;
  best_ask_qty: number;
  mid_price: number;
  absolute_spread: number;
  relative_spread_bps: number;
  microprice: number;
  microprice_minus_mid_bps: number;
  obi_qty_5: number;
  obi_qty_10: number;
  obi_qty_20: number;
  bid_depth_qty_5: number;
  ask_depth_qty_5: number;
  bid_depth_qty_10: number;
  ask_depth_qty_10: number;
  bid_depth_qty_20: number;
  ask_depth_qty_20: number;
  past_rv_60: number | null;
  past_rv_300: number | null;
  past_rv_3600: number | null;
  [key: string]: string | number | boolean | string[] | null;
};
type BookSnapshot = {
  symbol: string;
  sample_time_utc: string;
  event_time_utc: string | null;
  received_at_utc: string;
  last_update_id: number;
  book_valid: boolean;
  bids: [string, string][];
  asks: [string, string][];
  features: FeatureRow;
};
type ForecastModel = {
  name: string;
  family: "traditional" | "lob_model";
  model_key: string;
  status: string;
  forecast_rv: number | null;
  forecast_volatility_percent: number | null;
  training_examples: number;
  minimum_train_examples: number;
  oos_windows: number;
  oos_qlike: number | null;
};
type ForecastHorizon = {
  horizon_seconds: number;
  label: string;
  status: string;
  target: string;
  forecast_volatility_unit: string;
  valid_samples_in_window: number;
  required_samples: number;
  training_examples: number;
  minimum_train_examples: number;
  models: ForecastModel[];
  tail_risk: Record<string, { var_loss_percent: number; es_loss_percent: number; observations: number; method: string }>;
  out_of_sample: Record<string, { oos_windows: number; oos_qlike: number | null }>;
  fit_failures: Record<string, string>;
};
type LongForecastHorizon = {
  horizon_days: number;
  label: string;
  status: string;
  target: string;
  complete_days: number;
  required_days: number;
  training_examples: number;
  minimum_train_examples: number;
  overlapping_daily_origins: boolean;
  models: ForecastModel[];
  out_of_sample: Record<string, { oos_windows: number; oos_qlike: number | null }>;
  fit_failures: Record<string, string>;
};
type LongForecastResponse = {
  symbol: string;
  status: string;
  source: string;
  data_as_of_utc: string | null;
  complete_consecutive_days: number;
  horizons: LongForecastHorizon[];
  note: string;
};
type ForecastResponse = {
  symbol: string;
  status: string;
  feed_status: string;
  book_valid: boolean;
  last_error: string | null;
  quality_flags: string[];
  source: string;
  as_of_utc: string | null;
  sampling_interval_seconds: number;
  horizons: ForecastHorizon[];
  long_horizons: LongForecastResponse;
};

const API_ROOT = import.meta.env.VITE_API_ROOT ?? "";
const HORIZON_PURPOSE: Record<number, string> = {
  60: "Rủi ro rất ngắn hạn: nhạy với biến động mới nhất của mid-price và trạng thái book.",
  300: "Rủi ro intraday ngắn hạn: giảm phụ thuộc vào một dao động đơn lẻ trong book.",
  3600: "Rủi ro intraday theo giờ: horizon dài hơn, cần chuỗi dữ liệu liên tục lâu hơn.",
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : body.detail?.message;
    throw new Error(detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function number(value: unknown, digits = 2) {
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(parsed)
    : "—";
}

function percent(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "—";
}

function localTime(value?: string | null) {
  if (!value) return "Chưa có mẫu";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : new Intl.DateTimeFormat("vi-VN", { dateStyle: "medium", timeStyle: "medium" }).format(date);
}

function Metric({ label, value, note, tone = "" }: { label: string; value: string; note: string; tone?: string }) {
  return <article className={`lob-metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>;
}

function BookSide({ title, levels, side }: { title: string; levels: [string, string][]; side: "bid" | "ask" }) {
  let cumulative = 0;
  const rows = levels.slice(0, 20).map(([price, quantity], index) => {
    cumulative += Number(quantity) || 0;
    return { price, quantity, cumulative, index };
  });
  return <section className={`lob-book-side ${side}`}>
    <header><div><strong>{title}</strong><span>Top {rows.length} mức giá</span></div><small>Giá&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Khối lượng</small></header>
    <div className="lob-book-head"><span>#</span><span>Giá (USDT)</span><span>Khối lượng</span><span>Cộng dồn</span></div>
    {rows.length ? rows.map((row) => <div className="lob-book-row" key={`${side}-${row.price}`}>
      <span>{String(row.index + 1).padStart(2, "0")}</span><strong>{number(row.price, 2)}</strong><span>{number(row.quantity, 5)}</span><span>{number(row.cumulative, 5)}</span>
    </div>) : <div className="lob-book-empty">Chưa nhận được sổ lệnh hợp lệ.</div>}
  </section>;
}

export function LOBApp() {
  const [symbol, setSymbol] = useState<SymbolCode>("BTCUSDT");
  const [status, setStatus] = useState<CollectorStatus | null>(null);
  const [book, setBook] = useState<BookSnapshot | null>(null);
  const [feature, setFeature] = useState<FeatureRow | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      const results = await Promise.allSettled([
        getJson<CollectorStatus>("/api/v1/collector/status"),
        getJson<BookSnapshot>(`/api/v1/lob/latest?symbol=${symbol}`),
        getJson<FeatureRow>(`/api/v1/features/latest?symbol=${symbol}`),
        getJson<ForecastResponse>(`/api/v1/forecasts/latest?symbol=${symbol}`),
      ]);
      if (!mounted) return;
      const [statusResult, bookResult, featureResult, forecastResult] = results;
      if (statusResult.status === "fulfilled") setStatus(statusResult.value);
      if (bookResult.status === "fulfilled") setBook(bookResult.value);
      if (featureResult.status === "fulfilled") setFeature(featureResult.value);
      if (forecastResult.status === "fulfilled") setForecast(forecastResult.value);
      const feedError = statusResult.status === "rejected" ? statusResult.reason : null;
      setError(feedError instanceof Error ? feedError.message : feedError ? String(feedError) : null);
      setLastRefresh(new Date().toISOString());
      const currentStatus = statusResult.status === "fulfilled" ? statusResult.value.symbol_status[symbol]?.status : null;
      timer = setTimeout(refresh, currentStatus === "connected" ? 1000 : 2500);
    };
    void refresh();
    return () => { mounted = false; clearTimeout(timer); };
  }, [symbol]);

  const feed = status?.symbol_status[symbol];
  const data = feature ?? book?.features ?? null;
  const connected = feed?.status === "connected";
  const valid = Boolean(feed?.book_valid && data?.book_valid);
  const diagnostic = feed?.last_error ?? forecast?.last_error ?? error;

  return <main className="lob-app">
    <header className="lob-header">
      <div className="lob-brand"><div className="lob-brand-mark"><BookOpen size={21} /></div><div><strong>LOB / CRYPTO RISK</strong><span>Limit Order Book Research</span></div></div>
      <div className="lob-header-right">
        <label htmlFor="lob-symbol">Tài sản</label>
        <select id="lob-symbol" value={symbol} onChange={(event) => setSymbol(event.target.value as SymbolCode)}>
          <option value="BTCUSDT">BTC / USDT</option><option value="ETHUSDT">ETH / USDT</option>
        </select>
        <span className={`lob-connection ${connected ? "is-connected" : ""}`}><i />{connected ? "ĐANG NHẬN LOB" : feed?.status === "synchronizing" ? "ĐANG ĐỒNG BỘ" : "ĐANG KẾT NỐI"}</span>
      </div>
    </header>

    <section className="lob-intro">
      <div>
        <div className="lob-eyebrow"><Activity size={14} /> BINANCE SPOT · BTCUSDT / ETHUSDT</div>
        <h1>Sổ lệnh và rủi ro thanh khoản</h1>
        <p>Dashboard trình bày trạng thái sổ lệnh đã đồng bộ và các đặc trưng dùng cho nghiên cứu. Toàn bộ tín hiệu được lấy từ các mức giá bid/ask trên LOB.</p>
      </div>
      <div className="lob-scope"><ShieldCheck size={17} /><span><strong>LOB-only</strong><small>Không dùng nến OHLCV hay luồng giao dịch.</small></span></div>
    </section>

    <section className="lob-status-grid" aria-label="Chất lượng luồng dữ liệu">
      <Metric label="KẾT NỐI" value={connected ? "Đang chạy" : feed?.status === "synchronizing" ? "Đồng bộ book" : "Đang thử lại"} note={status?.provider ?? "Binance Spot"} tone={connected ? "good" : "pending"} />
      <Metric label="BOOK HỢP LỆ" value={valid ? "Có" : "Chưa có"} note={feed?.quality_flags?.join(", ") || (valid ? "Sequence liên tục · feed mới" : "Chờ snapshot và chuỗi update") } tone={valid ? "good" : "pending"} />
      <Metric label="EVENT RATE" value={number(feed?.event_rate_per_second, 2)} note="Sự kiện diff-depth mỗi giây · trung bình 60 giây" />
      <Metric label="ĐỘ TRỄ NHẬN" value={feed?.latency_ms == null ? "—" : `${number(feed.latency_ms, 0)} ms`} note="Chênh event time tới local receive time" />
      <Metric label="UPDATE ID CUỐI" value={feed?.last_update_id == null ? "—" : new Intl.NumberFormat("en-US").format(feed.last_update_id)} note="Mã sequence cuối đã áp dụng" />
      <Metric label="BOOK-VALID TIME" value={percent(feed?.book_valid_time_ratio)} note={`${number(feed?.book_valid_seconds, 0)} giây mẫu hợp lệ kể từ khi backend khởi động`} />
      <Metric label="RECONNECT / RESYNC" value={`${feed?.reconnect_count ?? 0} / ${feed?.resync_count ?? 0}`} note="Kết nối lại / đồng bộ lại snapshot" />
      <Metric label="MẪU ĐÃ LƯU" value={new Intl.NumberFormat("vi-VN").format(feed?.sample_count ?? 0)} note={`Lấy mẫu theo clock mỗi ${status?.sampling_interval_seconds ?? 1} giây`} />
    </section>

    {diagnostic && <div className="lob-alert"><CircleAlert size={18} /><div><strong>{valid ? "Collector đang có lỗi kết nối" : "Chưa đồng bộ được sổ lệnh"}</strong><span>{diagnostic}. Bộ đếm reconnect/resync cho biết số lần lỗi; restart API sau khi cập nhật code để áp dụng thời gian chờ kết nối mới.</span></div><RefreshCw size={15} />{feed?.last_error && <code>{feed.last_error}</code>}</div>}

    <section className="lob-market-grid">
      <div className="lob-panel lob-book-panel">
        <header className="lob-panel-title"><div><span className="lob-section-index">01 / LIVE BOOK</span><h2>Sổ lệnh hiện tại</h2><p>20 mức giá tốt nhất mỗi phía từ local book đã đồng bộ.</p></div><span className={`lob-valid-tag ${valid ? "valid" : "invalid"}`}>{valid ? "HỢP LỆ" : "CHƯA HỢP LỆ"}</span></header>
        <div className="lob-book-summary"><div><span>MID-PRICE PROXY</span><strong>{data ? number(data.mid_price, 2) : "—"}</strong></div><div><span>SPREAD</span><strong>{data ? `${number(data.relative_spread_bps, 3)} bps` : "—"}</strong></div><div><span>UPDATE ID</span><strong>{book?.last_update_id ? new Intl.NumberFormat("en-US").format(book.last_update_id) : "—"}</strong></div><div><span>MẪU GẦN NHẤT</span><strong className="lob-time">{localTime(data?.sample_time_utc)}</strong></div></div>
        <div className="lob-book-sides"><BookSide title="BID · LỆNH MUA" levels={book?.bids ?? []} side="bid" /><BookSide title="ASK · LỆNH BÁN" levels={book?.asks ?? []} side="ask" /></div>
        <footer className="lob-panel-note">Mid-price = (best bid + best ask) / 2. Đây là proxy suy ra từ sổ lệnh, không phải giá khớp lệnh.</footer>
      </div>

      <div className="lob-panel lob-feature-panel">
        <header className="lob-panel-title"><div><span className="lob-section-index">02 / BOOK FEATURES</span><h2>Tín hiệu từ độ sâu</h2><p>Đặc trưng đầu vào được tính từ snapshot tại cùng thời điểm.</p></div></header>
        <div className="lob-feature-highlight"><div><span>MICROPRICE</span><strong>{data ? number(data.microprice, 2) : "—"}</strong></div><div><span>SO VỚI MID</span><strong className={Number(data?.microprice_minus_mid_bps) >= 0 ? "positive" : "negative"}>{data ? `${number(data.microprice_minus_mid_bps, 3)} bps` : "—"}</strong></div><p>Microprice đặt trọng số cho giá bid/ask theo khối lượng đang chờ ở phía đối diện. Độ lệch so với mid-price cho biết book đang nghiêng về phía mua hay bán.</p></div>
        <div className="lob-obi-list">
          {[5, 10, 20].map((levels) => {
            const imbalance = Number(data?.[`obi_qty_${levels}`] ?? 0);
            return <div className="lob-obi-row" key={levels}><div><strong>OBI {levels}</strong><span>Top {levels} mức giá · −1 nghiêng bán, +1 nghiêng mua</span></div><strong className={imbalance > 0 ? "positive" : imbalance < 0 ? "negative" : ""}>{data ? number(imbalance, 4) : "—"}</strong><div className="lob-obi-bar"><i style={{ width: `${Math.min(100, Math.abs(imbalance) * 100)}%`, marginLeft: imbalance < 0 ? `${50 - Math.abs(imbalance) * 50}%` : "50%" }} /></div></div>;
          })}
        </div>
        <div className="lob-depth-table"><div><span>TOP-N DEPTH</span><span>BID</span><span>ASK</span></div>{[5, 10, 20].map((levels) => <div key={levels}><strong>{levels} mức</strong><span>{number(data?.[`bid_depth_qty_${levels}`], 5)}</span><span>{number(data?.[`ask_depth_qty_${levels}`], 5)}</span></div>)}</div>
        <p className="lob-panel-note">OBI = (tổng khối lượng bid − tổng khối lượng ask) / tổng hai phía. Depth tính bằng khối lượng tài sản cơ sở trong top-N levels.</p>
      </div>
    </section>

    <section className="lob-forecast-section lob-long-forecast-section">
      <header className="lob-forecast-heading">
        <div><span className="lob-section-index">03 / DỰ BÁO DÀI HẠN · LOB ONLY</span><h2>Biến động dự báo trong 1, 7 và 30 ngày</h2><p>Dự báo mức biến động tích lũy của mid-price suy ra từ sổ lệnh; không dự báo giá lên hay xuống.</p></div>
        <span className="lob-forecast-asof">Ngày dữ liệu đủ gần nhất: {localTime(forecast?.long_horizons?.data_as_of_utc)}</span>
      </header>
      <div className="lob-forecast-explainer"><strong>Cách đọc:</strong> dữ liệu mỗi giây được gộp thành RV và đặc trưng LOB theo ngày UTC. So sánh RV rolling mean (mốc truyền thống), HAR-RV ngày, Ridge và Gradient Boosting dùng thêm spread, OBI, depth, microprice. Cần đủ ngày LOB hoàn chỉnh liên tiếp; mô hình học cần thêm ít nhất 30 kết quả tương lai đã xảy ra. QLIKE thấp hơn nghĩa là dự báo RV khớp hơn; QLIKE 7/30 ngày có các cửa sổ chồng lấn nên chỉ để tham khảo.</div>
      <div className="lob-horizon-grid">
        {(forecast?.long_horizons?.horizons ?? [1, 7, 30].map((days) => ({
          horizon_days: days, label: `${days} ngày`, status: "warming_up", target: "RV tích lũy tương lai từ dữ liệu LOB", complete_days: 0, required_days: days,
          training_examples: 0, minimum_train_examples: 30, overlapping_daily_origins: true, models: [], out_of_sample: {}, fit_failures: {},
        }))).map((horizon) => <article className="lob-horizon-card lob-long-horizon-card" key={`long-${horizon.horizon_days}`}>
          <header><div><span>HORIZON DÀI</span><h3>{horizon.label}</h3></div><span className={`lob-valid-tag ${horizon.status === "ready" ? "valid" : "invalid"}`}>{horizon.status === "ready" ? "CÓ DỰ BÁO" : "ĐANG TÍCH LŨY"}</span></header>
          <p className="lob-horizon-target"><strong>{horizon.horizon_days === 1 ? "Rủi ro cho ngày kế tiếp." : horizon.horizon_days === 7 ? "Rủi ro cộng dồn trong một tuần." : "Rủi ro cộng dồn trong 30 ngày."}</strong><br />{horizon.target}.</p>
          <div className="lob-long-progress"><strong>{horizon.complete_days} / {horizon.required_days} ngày LOB hợp lệ liên tiếp</strong><span>{horizon.training_examples} / {horizon.minimum_train_examples} nhãn đã trưởng thành để huấn luyện</span></div>
          <div className="lob-model-table">
            <div className="lob-model-table-head"><span>MÔ HÌNH</span><span>ĐỘ LỆCH CHUẨN DỰ BÁO</span><span>QLIKE OOS ↓</span></div>
            {horizon.models.map((model) => <div className="lob-model-row" key={`${horizon.horizon_days}-${model.model_key}`}>
              <div><strong>{model.name}</strong><small>{model.family === "traditional" ? "Mốc so sánh truyền thống" : `LOB ML · ${model.training_examples}/${model.minimum_train_examples} nhãn huấn luyện`}</small></div>
              <strong>{model.forecast_volatility_percent == null ? "Đang làm nóng" : `${number(model.forecast_volatility_percent, 5)}%`}</strong>
              <span>{model.oos_qlike == null ? `— · ${model.oos_windows} mẫu` : `${number(model.oos_qlike, 5)} · ${model.oos_windows} mẫu`}</span>
            </div>)}
            {!horizon.models.length && <div className="lob-model-row lob-long-empty"><strong>Chưa có dự báo</strong><span>Chờ ngày LOB hoàn chỉnh đầu tiên.</span></div>}
          </div>
          <footer>{horizon.horizon_days === 1 ? "Mốc 1 ngày cần ít nhất 1 ngày hoàn chỉnh để có benchmark; mô hình học cần khoảng 31 ngày để có 30 nhãn." : horizon.horizon_days === 7 ? "Mốc 7 ngày cần 7 ngày lịch sử trước khi dự báo; khoảng 43 ngày để có 30 nhãn mature." : "Mốc 30 ngày cần 30 ngày lịch sử trước khi dự báo; khoảng 89 ngày để có 30 nhãn mature."} Dự báo dài hạn chốt theo ngày UTC, dữ liệu mới nhất được cập nhật ở lần hoàn tất ngày gần nhất: {forecast?.long_horizons?.data_as_of_utc ? localTime(forecast.long_horizons.data_as_of_utc) : "chưa có"}.</footer>
        </article>)}
      </div>
      {forecast?.long_horizons?.note && <div className="lob-forecast-explainer lob-long-note">{forecast.long_horizons.note}</div>}
    </section>

    <section className="lob-forecast-section">
      <header className="lob-forecast-heading">
        <div><span className="lob-section-index">04 / LIVE INTRADAY</span><h2>Theo dõi biến động ngắn hạn</h2><p>Ước lượng độ biến động trong vài phút tới từ trạng thái sổ lệnh đang nhận theo thời gian thực.</p></div>
        <span className="lob-forecast-asof">Cập nhật {localTime(forecast?.as_of_utc)}</span>
      </header>
      <div className="lob-forecast-explainer"><strong>Cách đọc:</strong> đây là lớp theo dõi book ngắn hạn. RV persistence giữ RV vừa quan sát; HAR-RV dùng cửa sổ 1 phút, 5 phút và 1 giờ. Ridge và Gradient Boosting thêm spread, imbalance, depth và microprice. Xem phần dự báo dài hạn bên trên cho mốc 1/7/30 ngày.</div>
      <div className="lob-horizon-grid">
        {(forecast?.horizons ?? []).map((horizon) => <article className="lob-horizon-card" key={horizon.horizon_seconds}>
          <header><div><span>HORIZON</span><h3>{horizon.label}</h3></div><span className={`lob-valid-tag ${horizon.status === "ready" ? "valid" : "invalid"}`}>{horizon.status === "ready" ? "CÓ DỰ BÁO" : "ĐANG TÍCH LŨY"}</span></header>
          <p className="lob-horizon-target"><strong>{HORIZON_PURPOSE[horizon.horizon_seconds] ?? "Rủi ro theo horizon từ LOB."}</strong><br />{horizon.target}. Cần {horizon.required_samples.toLocaleString("vi-VN")} mẫu liên tục để tạo benchmark.</p>
          <div className="lob-model-table">
            <div className="lob-model-table-head"><span>MÔ HÌNH</span><span>ĐỘ LỆCH CHUẨN DỰ BÁO</span><span>QLIKE OOS ↓</span></div>
            {horizon.models.map((model) => <div className="lob-model-row" key={model.model_key}>
              <div><strong>{model.name}</strong><small>{model.family === "traditional" ? "Benchmark truyền thống" : `LOB ML · ${model.training_examples}/${model.minimum_train_examples} nhãn huấn luyện`}</small></div>
              <strong>{model.forecast_volatility_percent == null ? "Đang làm nóng" : `${number(model.forecast_volatility_percent, 5)}%`}</strong>
              <span>{model.oos_qlike == null ? `— · ${model.oos_windows} mẫu` : `${number(model.oos_qlike, 5)} · ${model.oos_windows} mẫu`}</span>
            </div>)}
          </div>
          <div className="lob-tail-risk">
            <span>RỦI RO LỊCH SỬ TỪ CÁC CỬA SỔ KHÔNG CHỒNG LẤN</span>
            {["0.95", "0.99"].map((confidence) => {
              const tail = horizon.tail_risk[confidence];
              return <div key={confidence}><strong>VaR {confidence === "0.95" ? "95%" : "99%"}</strong><span>{tail ? `${number(tail.var_loss_percent, 5)}%` : "Chưa đủ mẫu"}</span><strong>ES</strong><span>{tail ? `${number(tail.es_loss_percent, 5)}%` : "—"}</span></div>;
            })}
          </div>
          <footer>{horizon.training_examples} nhãn horizon đã hoàn tất · OOS là dự báo được chấm sau khi tương lai xảy ra. {horizon.horizon_seconds === 3600 ? "Horizon dài cần thời gian tích lũy lâu hơn." : ""}</footer>
        </article>)}
      </div>
      {!forecast && <div className="lob-forecast-explainer">Đang chờ API dự báo LOB. Nếu backend đang chạy phiên bản cũ, khởi động lại backend để nạp endpoint mới.</div>}
    </section>

    <section className="lob-research-note">
      <div className="lob-research-icon"><Database size={19} /></div>
        <div><span className="lob-section-index">05 / RESEARCH READINESS</span><h2>Lịch sử LOB bắt đầu từ lúc collector chạy</h2><p>Binance Spot depth snapshot dùng để khởi tạo book hiện tại, không cung cấp lịch sử LOB nhiều năm qua REST. Hệ thống lưu raw diff-depth JSONL, Parquet snapshot/feature và manifest checksum từ thời điểm thu thập. Sau khi tích lũy đủ chuỗi mẫu hợp lệ, hãy chạy builder và walk-forward để có kết quả RV, VaR và ES ngoài mẫu.</p></div>
      <div className="lob-research-meta"><span>NGUỒN MỤC TIÊU</span><strong>Mid-price từ LOB</strong><small>Clock sampling · {status?.sampling_interval_seconds ?? 1}s</small><small>Last sample · {localTime(data?.sample_time_utc)}</small><small>Dashboard refresh · {localTime(lastRefresh)}</small></div>
    </section>

    <footer className="lob-footer"><span>Read-only research dashboard · Không gửi lệnh giao dịch</span><span>API <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">127.0.0.1:8000/docs</a></span></footer>
  </main>;
}
