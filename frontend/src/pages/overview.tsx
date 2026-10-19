import { ArrowDownRight, ArrowUpRight, RefreshCw, Waves } from "lucide-react";
import { useForecast, useForecastHistory, useLiveMarket, useRegime, useMeta } from "../api/queries";
import { useForecastMutation } from "../api/queries";
import type { ForecastRequest } from "../api/types";
import { useFilters } from "../app/filters";
import { ForecastPlot } from "../components/charts/ForecastPlot";
import { ApiState, PageIntro, Provenance, SectionHeader, StatTile } from "../components/PageParts";
import { formatDate, formatNumber, formatPercent } from "../components/format";
import { riskForConfidence } from "../api/client";

export function OverviewPage() {
  const { filters } = useFilters();
  const live = useLiveMarket(filters.symbol);
  const regime = useRegime(filters.symbol, filters.mode);
  const meta = useMeta(filters.mode);
  const researchForecast = useForecast(filters, filters.mode === "research");
  const mutation = useForecastMutation();
  const history = useForecastHistory(filters, filters.mode === "research");
  const forecast = filters.mode === "research" ? researchForecast.data : mutation.data;
  const risk = riskForConfidence(forecast?.var_es, filters.confidence);
  const request: ForecastRequest = { symbol: filters.symbol, model: filters.model, horizon_days: filters.horizon, mode: filters.mode, confidence_levels: [filters.confidence] };

  return <div className="page-stack">
    <PageIntro eyebrow="BẢNG ĐIỀU KHIỂN / 01" title="Tổng quan thị trường" description="Biến động, dự báo và rủi ro đuôi từ dữ liệu nghiên cứu đóng băng hoặc nguồn Binance đang cập nhật." actions={<Provenance />} />
    <div className="overview-grid">
      <StatTile label={`GIÁ NẾN 5 PHÚT · ${filters.symbol}`} value={live.data?.candle ? `$${formatNumber(Number(live.data.candle.price), 2)}` : "—"} detail={live.data?.candle ? <>Nến gần nhất · {formatDate(live.data.candle.close_time_utc ?? live.data.candle.open_time_utc, true)}</> : live.isLoading ? "Đang nhận dữ liệu Binance…" : "Chưa có nến trực tiếp"} tone="blue" />
      <StatTile label={`RV DỰ BÁO · ${filters.horizon} NGÀY`} value={forecast ? formatNumber(forecast.rv_forecast, 4) : "—"} detail={forecast ? `${filters.model} · phần trăm bình phương` : filters.mode === "production" ? "Chạy forecast để xem giá trị mới" : "Chưa có forecast phù hợp"} />
      <StatTile label={`VaR · ${Math.round(filters.confidence * 1000) / 10}%`} value={risk ? formatPercent(risk.var_loss_percent) : "—"} detail={risk ? `ES ${formatPercent(risk.expected_shortfall_loss_percent)} · ${risk.method}` : "Ước lượng rủi ro chưa khả dụng"} tone="red" />
      <StatTile label="REGIME HIỆN TẠI" value={regime.data?.regime ?? "—"} detail={regime.data ? `Ngày dữ liệu ${formatDate(regime.data.latest_available_date_utc)}` : "Chờ nhãn regime từ backend"} />
    </div>
    <div className="dashboard-banner">
      <div className="banner-icon"><Waves size={19} /></div>
      <div><strong>{filters.mode === "research" ? "Nghiên cứu walk-forward" : "Dữ liệu production rolling"}</strong><p>{filters.mode === "research" ? "Forecast đọc từ kết quả out-of-sample tại mốc snapshot; số thực tế chưa quan sát được sẽ để trống." : "Production dùng các ngày UTC hoàn chỉnh gần nhất. Mỗi yêu cầu forecast chạy mô hình trên dữ liệu đã đồng bộ ở máy này."}</p></div>
      {filters.mode === "production" && <button className="primary-button" onClick={() => mutation.mutate(request)} disabled={mutation.isPending}><RefreshCw size={15} className={mutation.isPending ? "spin" : ""} />{mutation.isPending ? "Đang tính…" : "Tạo forecast"}</button>}
    </div>
    <SectionHeader title="Diễn biến dự báo" note={`${filters.symbol} · ${filters.model} · horizon ${filters.horizon} ngày`} />
    <ApiState loading={history.isLoading} error={filters.mode === "research" ? history.error : undefined}>
      {history.data && <ForecastPlot points={history.data.points} />}
      {filters.mode === "production" && <div className="note-line">Lịch sử production chỉ gồm những ngày đã hoàn tất; dự báo hiện tại sẽ xuất hiện sau khi bấm “Tạo forecast”.</div>}
    </ApiState>
    <div className="overview-lower">
      <section className="info-panel"><div className="panel-heading"><h3>Nguồn & phiên bản</h3><span className="badge">{filters.mode === "research" ? "CỐ ĐỊNH" : "ROLLING"}</span></div>
        <dl className="detail-list"><div><dt>Phạm vi</dt><dd>{meta.data?.source ?? "—"}</dd></div><div><dt>Mốc nghiên cứu</dt><dd>{formatDate(meta.data?.as_of_utc)}</dd></div><div><dt>Phiên bản dữ liệu</dt><dd className="mono">{meta.data?.data_version?.slice(0, 16) ?? "—"}</dd></div></dl>
      </section>
      <section className="info-panel"><div className="panel-heading"><h3>Quan sát cuối cùng</h3><span className={`connection-pill ${live.data?.connection_status === "connected" ? "connected" : ""}`}><span className="status-dot" />{live.data?.connection_status ?? "đang kết nối"}</span></div>
        <dl className="detail-list"><div><dt>Mở cửa</dt><dd>{live.data?.candle ? formatDate(live.data.candle.open_time_utc, true) : "—"}</dd></div><div><dt>Đóng cửa</dt><dd>{live.data?.candle?.is_closed ? "Đã đóng" : live.data?.candle ? "Đang hình thành" : "—"}</dd></div><div><dt>Phạm vi high / low</dt><dd>{live.data?.candle ? `$${formatNumber(Number(live.data.candle.high))} / $${formatNumber(Number(live.data.candle.low))}` : "—"}</dd></div><div><dt>Nhận cập nhật</dt><dd>{live.data?.last_message_at_utc ? formatDate(live.data.last_message_at_utc, true) : "—"}</dd></div></dl>
      </section>
    </div>
    <div className="direction-legend"><span><ArrowUpRight size={13} /> Dự báo tăng</span><span><ArrowDownRight size={13} /> Mức loss VaR/ES luôn hiển thị theo chiều tổn thất dương</span></div>
  </div>;
}
