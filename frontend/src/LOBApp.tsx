import { useCallback, useEffect, useState } from "react";

type SymbolCode = "BTCUSDT" | "ETHUSDT";
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
  const [symbol, setSymbol] = useState<SymbolCode>("BTCUSDT");
  const [horizon, setHorizon] = useState(5);
  const [requestedTime, setRequestedTime] = useState("");
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [meter, setMeter] = useState<Meter | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await getJson<DataStatus>("/api/v1/data-status"));
      let next: Snapshot | null = null;
      if (requestedTime) {
        const ts = new Date(requestedTime).toISOString();
        const r = await getJson<{ snapshot: Snapshot | null; reason?: string }>(
          `/api/v1/replay/${symbol}/at?timestamp_utc=${encodeURIComponent(ts)}&depth=20`,
        );
        next = r.snapshot;
        if (!next && r.reason) setError(r.reason);
      } else {
        const r = await getJson<{ snapshots: Snapshot[] }>(`/api/v1/replay/${symbol}?limit=1&depth=20`);
        next = r.snapshots.at(-1) ?? null;
      }
      setSnapshot(next);
      if (next) {
        setRequestedTime((prev) => prev || toLocalInput(next!.timestamp_utc));
        setMeter(await getJson<Meter>(`/api/v1/meter/${symbol}?timestamp_utc=${encodeURIComponent(next.timestamp_utc)}&horizon_seconds=${horizon}`));
      } else {
        setMeter(await getJson<Meter>(`/api/v1/meter/${symbol}?horizon_seconds=${horizon}`));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }, [horizon, requestedTime, symbol]);

  useEffect(() => { void refresh(); }, [refresh]);

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
        <span className="mu">replay · chỉ đọc · không đặt lệnh</span>
      </header>

      <section className="controls" aria-label="Điều khiển replay">
        <Segmented label="cặp" value={symbol} onChange={setSymbol}
          options={[{ value: "BTCUSDT", text: "BTC/USDT" }, { value: "ETHUSDT", text: "ETH/USDT" }]} />
        <Segmented label="horizon" value={horizon} onChange={setHorizon}
          options={[{ value: 1, text: "1s" }, { value: 5, text: "5s" }, { value: 10, text: "10s" }]} />
        <label className="field grow">
          <span className="lbl">thời điểm replay · giờ máy</span>
          <input type="datetime-local" value={requestedTime} onChange={(e) => setRequestedTime(e.target.value)} />
        </label>
        <button type="button" className="go" onClick={() => void refresh()} disabled={busy}>{busy ? "đang tải…" : "nạp replay"}</button>
      </section>

      {error && <p className="alert" role="alert"><b>✕ lỗi</b> {error}</p>}

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
        <div><span className="lbl">replay data</span><strong>{symbolStatus?.status === "ready" ? "sẵn sàng" : "chưa có"}</strong>
          <span className="src">{symbolStatus ? `${num(symbolStatus.row_count, 0)} dòng · ${symbolStatus.partition_count} partition` : "chưa có manifest"}</span></div>
        <div><span className="lbl">khoảng quan sát</span><strong>{symbolStatus?.first_date_utc ?? "—"} → {symbolStatus?.last_date_utc ?? "—"}</strong>
          <span className="src">{formatTime(snapshot?.timestamp_utc)}</span></div>
        <div><span className="lbl">chất lượng book</span>
          <strong className={bookBad ? "breach" : ""}>{!snapshot ? "đang chờ dữ liệu" : bookBad ? "✕ book lỗi / có gap" : "✓ hợp lệ"}</strong>
          <span className="src">{snapshot?.quality.seconds_since_gap != null ? `${snapshot.quality.seconds_since_gap}s từ gap gần nhất` : "sequence và độ tươi"}</span></div>
        <div><span className="lbl">model artifact</span><strong>{modelReady ? "sẵn sàng" : "chưa có"}</strong>
          <span className="src">{meter?.model_version ?? meter?.reason ?? "chưa nạp model"}</span></div>
      </section>

      <footer className="foot">
        <p>API đọc partition Parquet đã chuẩn hóa và classifier đã lưu. Dữ liệu L2 chỉ được thu từ lúc bắt đầu capture; xác suất chỉ coi là kết quả nghiên cứu sau khi có đủ tuần huấn luyện, kiểm định walk-forward và tuần test 2026-11-23 → 2026-11-29 được chấm một lần.</p>
        <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">api · 127.0.0.1:8000/docs</a>
      </footer>
    </main>
  );
}
