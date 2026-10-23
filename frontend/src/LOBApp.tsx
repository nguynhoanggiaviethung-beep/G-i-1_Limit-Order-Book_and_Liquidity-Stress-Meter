import { useCallback, useEffect, useRef, useState } from "react";

type SymbolCode = "BTCUSDT" | "ETHUSDT";
type Mode = "live" | "replay";
type Level = [string, string];
type Snapshot = {
  symbol: SymbolCode;
  timestamp_utc: string;
  bids: Level[];
  asks: Level[];
  features: Record<string, unknown>;
  quality: { book_valid: boolean; seconds_since_gap?: number | null; sequence_gap?: boolean };
};
type DataStatus = {
  status: string;
  source: string;
  symbols: Record<string, { status: string; partition_count: number; row_count: number; first_date_utc: string | null; last_date_utc: string | null; reason: string | null }>;
};
type Meter = {
  status: string;
  reason?: string | null;
  timestamp_utc?: string | null;
  horizon_seconds: number;
  spread?: string | null;
  depth_imbalance?: Record<string, number | null>;
  ofi_5s?: number | null;
  p_up?: number | null;
  p_flat?: number | null;
  p_down?: number | null;
  p_stress?: number | null;
  stress_probability_threshold?: number | null;
  model_version?: string | null;
  quality?: { book_valid?: boolean; reason?: string; seconds_since_gap?: number | null };
};
type LiveUpdate = {
  schema: string;
  symbol: SymbolCode;
  status: string;
  reason?: string | null;
  updated_at_utc: string;
  snapshot: Snapshot | null;
  meter: Meter;
  horizon_seconds: number;
};

const API_ROOT = import.meta.env.VITE_API_ROOT ?? "";
async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : body.detail?.message;
    throw new Error(detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function num(value: unknown, digits = 4, signed = false) {
  const parsed = Number(value);
  if (value == null || !Number.isFinite(parsed)) return "—";
  const text = new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(parsed);
  return signed && parsed > 0 ? `+${text}` : text;
}

function pct(value: unknown) {
  const parsed = Number(value);
  return value != null && Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "—";
}

function formatTime(value?: string | null) {
  if (!value) return "chưa có dữ liệu";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "—"
    : new Intl.DateTimeFormat("vi-VN", { dateStyle: "medium", timeStyle: "medium", timeZone: "Asia/Ho_Chi_Minh" }).format(date);
}

function toLocalInput(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function noReplayMessage(symbol: SymbolCode, timestamp: string | undefined, status: DataStatus | null, reason?: string) {
  const symbolData = status?.symbols?.[symbol];
  const first = symbolData?.first_date_utc;
  const last = symbolData?.last_date_utc;
  const range = first && last
    ? `Dữ liệu ${symbol} hiện có từ ${first} đến ${last} (UTC).`
    : `Hiện chưa có dữ liệu replay cho ${symbol}.`;
  const selected = timestamp ? ` tại ${formatTime(new Date(timestamp).toISOString())}` : "";
  const day = timestamp ? new Date(timestamp).toISOString().slice(0, 10) : null;
  const outsideRange = Boolean(day && first && last && (day < first || day > last));
  const staleBook = reason?.match(/Latest valid book is ([\d.]+)s old \(limit ([\d.]+)s\)/);
  const explanation = outsideRange
    ? "Chọn thời điểm trong khoảng dữ liệu; capture live không thể khôi phục sổ lệnh lịch sử của ngày trước đó."
    : staleBook
      ? `Snapshot gần nhất cách mốc yêu cầu ${staleBook[1]} giây, vượt ngưỡng cho phép ${staleBook[2]} giây.`
      : "Ngày này có dữ liệu, nhưng không tìm thấy book hợp lệ gần mốc đã chọn; hãy chọn giờ khác trong khoảng quan sát.";
  return `Không có replay book cho ${symbol}${selected}. ${range} ${explanation}`;
}

function liveStatusLabel(status: string) {
  switch (status) {
    case "live": return "đang nhận dữ liệu";
    case "syncing": return "đang đồng bộ sổ lệnh";
    case "reconnecting": return "đang kết nối lại";
    case "stale": return "dữ liệu bị gián đoạn";
    case "disabled": return "đã tắt nguồn live";
    case "error": return "lỗi nguồn dữ liệu";
    case "stopped": return "đã dừng";
    default: return "đang kết nối Binance";
  }
}

function Stat({ label, value, unit, source }: { label: string; value: string; unit?: string; source: string }) {
  return (
    <article className="stat">
      <span className="lbl">{label}</span>
      <strong className="big">{value}{unit && <small>{unit}</small>}</strong>
      <span className="src">nguồn · {source}</span>
    </article>
  );
}

function Segmented<T extends string | number>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; text: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="field">
      <span className="lbl">{label}</span>
      <div className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.text}</button>
        ))}
      </div>
    </div>
  );
}

function BookSide({ side, levels, scale }: { side: "bid" | "ask"; levels: Level[]; scale: number }) {
  let running = 0;
  return (
    <section className="book">
      <h3>{side === "bid" ? "bid · mua" : "ask · bán"}</h3>
      <div className="row head"><span>#</span><span>giá (usdt)</span><span>qty</span><span>cộng dồn</span></div>
      {levels.slice(0, 10).map(([price, qty], i) => {
        running += Number(qty) || 0;
        return (
          <div className="row" key={`${side}-${price}`}>
            <i className={`bar ${side}`} style={{ width: `${scale ? Math.min(100, (running / scale) * 100) : 0}%` }} />
            <span className="mu">{String(i + 1).padStart(2, "0")}</span>
            <span>{num(price, 2)}</span>
            <span>{num(qty, 5)}</span>
            <span>{num(running, 5)}</span>
          </div>
        );
      })}
      {!levels.length && <p className="empty">Chưa có snapshot hợp lệ.</p>}
    </section>
  );
}

function ProbRow({ mark, label, value, tone }: { mark: string; label: string; value?: number | null; tone: "data" | "mute" | "ink" }) {
  const v = value != null && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
  return (
    <div className="prob">
      <div><span>{mark} {label}</span><strong>{v === null ? "—" : pct(v)}</strong></div>
      <div className="track"><i className={tone} style={{ width: `${(v ?? 0) * 100}%` }} /></div>
    </div>
  );
}

export function LOBApp() {
  const [mode, setMode] = useState<Mode>("live");
  const [symbol, setSymbol] = useState<SymbolCode>("BTCUSDT");
  const [horizon, setHorizon] = useState(5);
  const [requestedTime, setRequestedTime] = useState("");
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [meter, setMeter] = useState<Meter | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState("starting");
  const [liveReason, setLiveReason] = useState<string | null>("Đang khởi động nguồn dữ liệu Binance Spot.");
  const requestVersion = useRef(0);

  const refresh = useCallback(async () => {
    const version = ++requestVersion.current;
    const isCurrent = () => version === requestVersion.current;
    setBusy(true);
    setError(null);
    setSnapshot(null);
    setMeter(null);
    try {
      const nextStatus = await getJson<DataStatus>("/api/v1/data-status");
      if (!isCurrent()) return;
      setStatus(nextStatus);
      let next: Snapshot | null = null;
      if (requestedTime) {
        const ts = new Date(requestedTime).toISOString();
        const r = await getJson<{ snapshot: Snapshot | null; reason?: string }>(
          `/api/v1/replay/${symbol}/at?timestamp_utc=${encodeURIComponent(ts)}&depth=20`,
        );
        if (!isCurrent()) return;
        next = r.snapshot;
        if (!next) {
          setError(noReplayMessage(symbol, ts, nextStatus, r.reason));
          return;
        }
      } else {
        const r = await getJson<{ snapshots: Snapshot[] }>(`/api/v1/replay/${symbol}?limit=1&depth=20`);
        if (!isCurrent()) return;
        next = r.snapshots.at(-1) ?? null;
        if (!next) {
          setError(noReplayMessage(symbol, undefined, nextStatus));
          return;
        }
      }
      setSnapshot(next);
      if (!requestedTime) setRequestedTime(toLocalInput(next.timestamp_utc));
      const nextMeter = await getJson<Meter>(`/api/v1/meter/${symbol}?timestamp_utc=${encodeURIComponent(next.timestamp_utc)}&horizon_seconds=${horizon}`);
      if (!isCurrent()) return;
      setMeter(nextMeter);
    } catch (cause) {
      if (isCurrent()) setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (isCurrent()) setBusy(false);
    }
  }, [horizon, requestedTime, symbol]);

  useEffect(() => {
    if (mode === "replay") void refresh();
  }, [mode, refresh]);

  useEffect(() => {
    if (mode !== "live") return;
    requestVersion.current += 1;
    setError(null);
    setSnapshot(null);
    setMeter(null);
    setLiveStatus("connecting");
    setLiveReason("Đang mở luồng dữ liệu Binance Spot.");
    let active = true;
    const stream = new EventSource(`${API_ROOT}/api/v1/live/${symbol}/stream?horizon_seconds=${horizon}`);
    stream.onmessage = (event) => {
      if (!active) return;
      try {
        const update = JSON.parse(event.data) as LiveUpdate;
        setLiveStatus(update.status);
        setLiveReason(update.reason ?? null);
        setSnapshot(update.snapshot);
        setMeter(update.meter);
      } catch {
        setLiveStatus("error");
        setLiveReason("Không đọc được dữ liệu cập nhật từ máy chủ.");
      }
    };
    stream.onerror = () => {
      if (!active) return;
      setLiveStatus("reconnecting");
      setLiveReason("Mất kết nối tới máy chủ live; trình duyệt đang tự kết nối lại.");
    };
    return () => {
      active = false;
      stream.close();
    };
  }, [mode, symbol, horizon]);

  const symbolStatus = status?.symbols?.[symbol];
  const f = snapshot?.features ?? {};
  const imb = meter?.depth_imbalance ?? {};
  const cum = (levels: Level[]) => levels.slice(0, 10).reduce((s, [, q]) => s + (Number(q) || 0), 0);
  const scale = Math.max(cum(snapshot?.bids ?? []), cum(snapshot?.asks ?? []));
  const ready = meter?.status === "ok";
  const modelReady = Boolean(meter?.model_version);
  const stressThreshold = meter?.stress_probability_threshold;
  const stressFlag = ready && stressThreshold != null && meter?.p_stress != null && meter.p_stress >= stressThreshold;
  const bookBad = snapshot ? !snapshot.quality.book_valid || !!snapshot.quality.sequence_gap : false;

  return (
    <main className="app">
      <header className="top">
        <strong>liquidity stress meter</strong>
        <span className="mu">{mode === "live" ? `LIVE · ${liveStatusLabel(liveStatus)} · chỉ đọc · không đặt lệnh` : "replay · chỉ đọc · không đặt lệnh"}</span>
      </header>

      <section className="controls" aria-label="Điều khiển replay">
        <Segmented label="chế độ" value={mode} onChange={setMode}
          options={[{ value: "live", text: "LIVE" }, { value: "replay", text: "replay" }]} />
        <Segmented label="cặp" value={symbol} onChange={(nextSymbol) => {
          if (nextSymbol === symbol) return;
          setSymbol(nextSymbol);
          setRequestedTime("");
          setSnapshot(null);
          setMeter(null);
          setError(null);
        }}
          options={[{ value: "BTCUSDT", text: "BTC/USDT" }, { value: "ETHUSDT", text: "ETH/USDT" }]} />
        <Segmented label="horizon" value={horizon} onChange={setHorizon}
          options={[{ value: 1, text: "1s" }, { value: 5, text: "5s" }, { value: 10, text: "10s" }]} />
        {mode === "replay" ? (
          <>
            <label className="field grow">
              <span className="lbl">thời điểm replay · giờ máy</span>
              <input
                type="datetime-local"
                value={requestedTime}
                min={symbolStatus?.first_date_utc ? toLocalInput(`${symbolStatus.first_date_utc}T00:00:00Z`) : undefined}
                max={symbolStatus?.last_date_utc ? toLocalInput(`${symbolStatus.last_date_utc}T23:59:59Z`) : undefined}
                onChange={(e) => setRequestedTime(e.target.value)}
              />
            </label>
            <button type="button" className="go" onClick={() => void refresh()} disabled={busy}>{busy ? "đang tải…" : "nạp replay"}</button>
          </>
        ) : (
          <div className={`live-indicator ${liveStatus === "live" ? "is-live" : ""}`} role="status">
            <span>{liveStatusLabel(liveStatus)}</span>
            <small>{liveReason ?? "Binance Spot · WebSocket công khai"}</small>
          </div>
        )}
      </section>

      {mode === "replay" && error && <p className="alert" role="alert"><b>✕ lỗi</b> {error}</p>}
      {mode === "live" && ["stale", "error", "disabled", "stopped"].includes(liveStatus) && liveReason &&
        <p className="alert" role="alert"><b>✕ live</b> {liveReason}</p>}

      <section className="stats" aria-label="Chỉ số chính">
        <Stat label="mid-price" value={num(f.mid_price, 2)} source="(ask1 + bid1) / 2" />
        <Stat label="spread" value={f.relative_spread_bps == null ? "—" : num(f.relative_spread_bps, 4)} unit=" bps" source="ask1 − bid1 trên mid" />
        <Stat label="depth imbalance · l10" value={num(imb.L10, 4, true)} source="chênh lệch qty bid/ask, −1 đến +1" />
        <Stat label="order flow · ofi 5s" value={num(meter?.ofi_5s, 2, true)} source="ofi theo sự kiện, cộng 5 giây" />
      </section>

      <section className="bookgrid" aria-label="Sổ lệnh">
        <BookSide side="bid" levels={snapshot?.bids ?? []} scale={scale} />
        <BookSide side="ask" levels={snapshot?.asks ?? []} scale={scale} />
        <aside className="imb">
          <h3>imbalance theo lớp</h3>
          {[1, 5, 10, 20].map((d) => {
            const v = Number(imb[`L${d}`]);
            const ok = Number.isFinite(v);
            return (
              <div className="imbrow" key={d}>
                <span>L{d}</span>
                <div className="center"><i className={v >= 0 ? "data" : "ink"} style={{ width: `${ok ? Math.min(50, Math.abs(v) * 50) : 0}%`, left: v >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(v) * 50)}%` }} /></div>
                <b>{num(ok ? v : null, 3, true)}</b>
              </div>
            );
          })}
          <p className="src">dương: bid dày hơn · âm: ask dày hơn</p>
        </aside>
      </section>

      <section className="models">
        <article>
          <h3>hướng mid-price · {horizon}s</h3>
          {ready ? (
            <>
              <ProbRow mark="▲" label="up" value={meter?.p_up} tone="data" />
              <ProbRow mark="▬" label="flat" value={meter?.p_flat} tone="mute" />
              <ProbRow mark="▼" label="down" value={meter?.p_down} tone="ink" />
            </>
          ) : (
            <p className="empty"><b>— chưa có xác suất.</b> {meter?.reason ?? "Replay hoặc classifier chưa sẵn sàng."}</p>
          )}
          <span className="src">nguồn · {meter?.model_version ?? "chưa có model"} · δ fit trên tập train</span>
        </article>
        <article>
          <h3>xác suất stress · {horizon}s</h3>
          {ready ? (
            <>
              <strong className={`big ${stressFlag ? "breach" : ""}`}>{pct(meter?.p_stress)}</strong>
              {stressThreshold != null ? (
                <span className={stressFlag ? "flag breach" : "flag"}>
                  {stressFlag ? `■ vượt ngưỡng ${pct(stressThreshold)}` : `□ dưới ngưỡng ${pct(stressThreshold)}`}
                </span>
              ) : (
                <span className="flag">Chưa có ngưỡng quyết định trong model artifact</span>
              )}
              <div className="track"><i className={stressFlag ? "breach" : "data"} style={{ width: `${Math.min(100, Number(meter?.p_stress ?? 0) * 100)}%` }} /></div>
            </>
          ) : (
            <p className="empty"><b>— meter chưa sẵn sàng.</b> {meter?.reason ?? "Cần replay data và model artifact tương thích."}</p>
          )}
          <span className="src">nguồn · spread, price impact, depth top-10, rv · ngưỡng chọn trên dự báo walk-forward validation</span>
        </article>
      </section>

      <section className="quality" aria-label="Trạng thái dữ liệu">
        <div><span className="lbl">{mode === "live" ? "nguồn dữ liệu" : "replay data"}</span>
          <strong>{mode === "live" ? liveStatusLabel(liveStatus) : symbolStatus?.status === "ready" ? "sẵn sàng" : "chưa có"}</strong>
          <span className="src">{mode === "live" ? liveReason ?? "Binance Spot · WebSocket công khai" : symbolStatus ? `${num(symbolStatus.row_count, 0)} dòng · ${symbolStatus.partition_count} partition` : "chưa có manifest"}</span></div>
        <div><span className="lbl">{mode === "live" ? "cập nhật gần nhất" : "khoảng quan sát"}</span>
          <strong>{mode === "live" ? formatTime(snapshot?.timestamp_utc) : `${symbolStatus?.first_date_utc ?? "—"} → ${symbolStatus?.last_date_utc ?? "—"}`}</strong>
          <span className="src">{mode === "live" ? "thời gian sàn · UTC hiển thị theo giờ Việt Nam" : formatTime(snapshot?.timestamp_utc)}</span></div>
        <div><span className="lbl">chất lượng book</span>
          <strong className={bookBad ? "breach" : ""}>{!snapshot ? "đang chờ dữ liệu" : bookBad ? "✕ book lỗi / có gap" : "✓ hợp lệ"}</strong>
          <span className="src">{snapshot?.quality.seconds_since_gap != null ? `${snapshot.quality.seconds_since_gap}s từ gap gần nhất` : "sequence và độ tươi"}</span></div>
        <div><span className="lbl">model artifact</span><strong>{modelReady ? "sẵn sàng" : "chưa có"}</strong>
          <span className="src">{meter?.model_version ?? meter?.reason ?? "chưa nạp model"}</span></div>
      </section>

      <footer className="foot">
        <p>{mode === "live"
          ? "Nguồn live là Binance Spot public WebSocket, lưu capture liên tục và không đặt lệnh. Các xác suất chỉ xuất hiện sau khi đủ dữ liệu rolling 5 phút và có model tương thích."
          : "Replay đọc dữ liệu đã lưu. Dữ liệu capture live không thể khôi phục sổ lệnh L2 của những ngày trước thời điểm capture."}</p>
        <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">api · 127.0.0.1:8000/docs</a>
      </footer>
    </main>
  );
}
