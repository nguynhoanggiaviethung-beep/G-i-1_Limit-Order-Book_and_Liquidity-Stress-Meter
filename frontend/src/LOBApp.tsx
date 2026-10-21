import { useCallback, useEffect, useState } from "react";
import { Activity, AlertTriangle, Clock3, Database, RefreshCw, ShieldCheck, Waves } from "lucide-react";

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

function number(value: unknown, digits = 4) {
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(parsed)
    : "—";
}

function percent(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "—";
}

function formatTime(value?: string | null) {
  if (!value) return "Chưa có dữ liệu";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : new Intl.DateTimeFormat("vi-VN", { dateStyle: "medium", timeStyle: "medium", timeZone: "Asia/Ho_Chi_Minh" }).format(date);
}

function toLocalInput(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <article className="meter-metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}

function Probability({ label, value, color }: { label: string; value: number | null | undefined; color: string }) {
  const shown = Number.isFinite(value) ? Math.max(0, Math.min(1, Number(value))) : null;
  return <div className="meter-probability">
    <div><span>{label}</span><strong>{shown === null ? "—" : percent(shown)}</strong></div>
    <div className="meter-probability-track"><i style={{ width: `${(shown ?? 0) * 100}%`, background: color }} /></div>
  </div>;
}

function BookTable({ side, levels }: { side: "bid" | "ask"; levels: Level[] }) {
  let runningDepth = 0;
  return <section className={`meter-book-side ${side}`}>
    <header><strong>{side === "bid" ? "Bid depth" : "Ask depth"}</strong><span>Giá · Khối lượng · Cộng dồn</span></header>
    <div className="meter-book-header"><span>#</span><span>Giá (USDT)</span><span>Qty</span><span>Depth</span></div>
    {levels.slice(0, 10).map(([price, quantity], index) => {
      runningDepth += Number(quantity) || 0;
      return <div className="meter-book-row" key={`${side}-${price}`}>
        <span>{String(index + 1).padStart(2, "0")}</span>
        <strong>{number(price, 2)}</strong>
        <span>{number(quantity, 5)}</span>
        <span>{number(runningDepth, 5)}</span>
      </div>;
    })}
    {!levels.length && <div className="meter-empty-row">Chưa có snapshot hợp lệ.</div>}
  </section>;
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
      const dataStatus = await getJson<DataStatus>("/api/v1/data-status");
      setStatus(dataStatus);
      let next: Snapshot | null = null;
      if (requestedTime) {
        const timestamp = new Date(requestedTime).toISOString();
        const response = await getJson<{ status: string; snapshot: Snapshot | null; reason?: string }>(
          `/api/v1/replay/${symbol}/at?timestamp_utc=${encodeURIComponent(timestamp)}&depth=20`,
        );
        next = response.snapshot;
        if (!next && response.reason) setError(response.reason);
      } else {
        const response = await getJson<{ snapshots: Snapshot[] }>(`/api/v1/replay/${symbol}?limit=1&depth=20`);
        next = response.snapshots.at(-1) ?? null;
      }
      setSnapshot(next);
      if (next) {
        setRequestedTime((previous) => previous || toLocalInput(next!.timestamp_utc));
        setMeter(await getJson<Meter>(
          `/api/v1/meter/${symbol}?timestamp_utc=${encodeURIComponent(next.timestamp_utc)}&horizon_seconds=${horizon}`,
        ));
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
  const features = snapshot?.features ?? {};
  const imbalances = meter?.depth_imbalance ?? {};

  return <main className="meter-app">
    <header className="meter-header">
      <a className="meter-brand" href="#top"><span className="meter-brand-mark"><Activity size={20} /></span><span><strong>LIQUIDITY STRESS METER</strong><small>CRYPTO LIMIT ORDER BOOK RESEARCH</small></span></a>
      <div className="meter-header-note"><span className={`meter-dot ${snapshot ? "ready" : ""}`} /> HISTORICAL REPLAY · READ ONLY</div>
    </header>

    <div className="meter-content" id="top">
      <section className="meter-intro">
        <div><span className="meter-eyebrow">MICROSTRUCTURE / RESEARCH BLUEPRINT 05</span><h1>Order book state.<br />Short-horizon stress.</h1>
          <p>Replay BTCUSDT and ETHUSDT books to inspect depth, imbalance, order flow, mid-price direction, and liquidity-stress probability.</p>
        </div>
        <div className="meter-controls">
          <label><span>SYMBOL</span><select value={symbol} onChange={(event) => setSymbol(event.target.value as SymbolCode)}><option value="BTCUSDT">BTC / USDT</option><option value="ETHUSDT">ETH / USDT</option></select></label>
          <label><span>HORIZON</span><select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}><option value={1}>1 second</option><option value={5}>5 seconds · primary</option><option value={10}>10 seconds</option></select></label>
          <label className="meter-time-control"><span>REPLAY TIME · LOCAL</span><input type="datetime-local" value={requestedTime} onChange={(event) => setRequestedTime(event.target.value)} /></label>
          <button type="button" className="meter-refresh" onClick={() => void refresh()} disabled={busy}><RefreshCw size={15} className={busy ? "meter-spin" : ""} />{busy ? "ĐANG TẢI" : "NẠP REPLAY"}</button>
        </div>
      </section>

      {error && <div className="meter-alert"><AlertTriangle size={17} /><span>{error}</span></div>}

      <section className="meter-state-strip">
        <div><Database size={16} /><span>REPLAY DATA</span><strong>{symbolStatus?.status === "ready" ? "AVAILABLE" : "NOT AVAILABLE"}</strong><small>{symbolStatus ? `${number(symbolStatus.row_count, 0)} rows · ${symbolStatus.partition_count} ngày-partitions` : "Chưa có manifest"}</small></div>
        <div><Clock3 size={16} /><span>OBSERVED RANGE</span><strong>{symbolStatus?.first_date_utc ?? "—"} → {symbolStatus?.last_date_utc ?? "—"}</strong><small>{formatTime(snapshot?.timestamp_utc)}</small></div>
        <div><ShieldCheck size={16} /><span>BOOK QUALITY</span><strong>{snapshot?.quality.book_valid ? "VALID SNAPSHOT" : "WAITING FOR DATA"}</strong><small>{snapshot?.quality.seconds_since_gap != null ? `${snapshot.quality.seconds_since_gap}s since last gap` : "Sequence and freshness metadata"}</small></div>
        <div><Waves size={16} /><span>MODEL ARTIFACT</span><strong>{meter?.status === "ok" ? "READY" : "UNAVAILABLE"}</strong><small>{meter?.model_version ?? meter?.reason ?? "No trained model loaded"}</small></div>
      </section>

      <section className="meter-section-heading"><div><span>01 / LIVE REPLAY SNAPSHOT</span><h2>Liquidity and depth</h2></div><p>{snapshot ? `${symbol} · ${formatTime(snapshot.timestamp_utc)}` : "Waiting for normalized one-second replay data"}</p></section>
      <section className="meter-metrics-grid">
        <Metric label="MID-PRICE" value={number(features.mid_price, 2)} detail="(Best ask + best bid) / 2" />
        <Metric label="SPREAD" value={features.relative_spread_bps == null ? "—" : `${number(features.relative_spread_bps, 3)} bps`} detail="Relative spread at the best quote" />
        <Metric label="DEPTH IMBALANCE · L10" value={number(imbalances.L10, 4)} detail="Signed quantity imbalance · range −1 to +1" />
        <Metric label="ORDER FLOW · OFI 5S" value={number(meter?.ofi_5s, 3)} detail="Event-level OFI summed over five seconds" />
      </section>

      <section className="meter-book-grid">
        <BookTable side="bid" levels={snapshot?.bids ?? []} />
        <BookTable side="ask" levels={snapshot?.asks ?? []} />
        <aside className="meter-depth-summary">
          <span className="meter-eyebrow">DEPTH IMBALANCE BY LEVEL</span>
          {[1, 5, 10, 20].map((depth) => {
            const value = imbalances[`L${depth}`];
            const numeric = Number.isFinite(value) ? Number(value) : 0;
            const width = `${Math.min(50, Math.abs(numeric) * 50)}%`;
            return <div className="meter-imbalance-row" key={depth}><span>L{depth}</span><div><i className={numeric >= 0 ? "bid-imbalance" : "ask-imbalance"} style={{ width, left: numeric >= 0 ? "50%" : `calc(50% - ${width})` }} /></div><strong>{number(value, 3)}</strong></div>;
          })}
          <small>Positive means bid depth is larger; negative means ask depth is larger.</small>
        </aside>
      </section>

      <section className="meter-model-grid">
        <article className="meter-model-panel">
          <header><div><span className="meter-eyebrow">02 / MID-PRICE DIRECTION</span><h2>Probability by class</h2></div><span className="meter-horizon-tag">{horizon}s</span></header>
          {meter?.status === "ok" ? <div className="meter-probability-list">
            <Probability label="UP" value={meter.p_up} color="#387a67" />
            <Probability label="FLAT" value={meter.p_flat} color="#8a8d91" />
            <Probability label="DOWN" value={meter.p_down} color="#b24b42" />
          </div> : <div className="meter-unavailable"><AlertTriangle size={18} /><strong>Chưa có xác suất dự báo</strong><p>{meter?.reason ?? "Replay row hoặc trained classifier chưa sẵn sàng."}</p></div>}
          <footer>Three-class target uses a train-fitted flat-band delta. Probabilities appear only after the matching saved model is available.</footer>
        </article>
        <article className="meter-model-panel stress-panel">
          <header><div><span className="meter-eyebrow">03 / LIQUIDITY STRESS</span><h2>Stress probability</h2></div><span className="meter-horizon-tag">{horizon}s</span></header>
          {meter?.status === "ok" ? <div className="meter-stress-readout"><strong>{percent(meter.p_stress)}</strong><span>COMPOSITE STRESS PROBABILITY</span><div className="meter-probability-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(meter.p_stress ?? 0) * 100))}%`, background: "#c55d3b" }} /></div></div> : <div className="meter-unavailable"><AlertTriangle size={18} /><strong>Meter chưa sẵn sàng</strong><p>{meter?.reason ?? "Cần replay data và model artifact tương thích."}</p></div>}
          <footer>Stress combines spread widening, fixed-notional price impact, top-10 depth depletion, and trailing realized variance.</footer>
        </article>
      </section>

      <section className="meter-note"><div><span className="meter-eyebrow">RESEARCH BOUNDARY</span><h2>Replay first. Live feed later.</h2><p>The active API reads historical Parquet partitions and saved classifiers. The source dataset and trained artifacts are not included in this checkout. Data quality, label prevalence, calibration, and the untouched December 2025 test must be established before probabilities are presented as research results.</p></div><div className="meter-note-meta"><strong>DATA SOURCE</strong><span>{status?.source ?? "CryptoHFTData L2 + trades"}</span><strong>API CONTRACT</strong><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">127.0.0.1:8000/docs</a></div></section>
      <footer className="meter-footer"><span>LIMIT ORDER BOOK & LIQUIDITY STRESS METER</span><span>Read-only historical replay · No trading orders</span></footer>
    </div>
  </main>;
}
