import { ArrowDownRight, ArrowUpRight, RefreshCw, Waves } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useForecast, useForecastHistory, useLiveMarket, useRegime, useMeta } from "../api/queries";
import { useForecastMutation } from "../api/queries";
import type { ForecastRequest } from "../api/types";
import { useFilters } from "../app/filters";
import { ForecastPlot } from "../components/charts/ForecastPlot";
import { ApiState, PageIntro, Provenance, SectionHeader, StatTile } from "../components/PageParts";
import { formatDate, formatNumber, formatPercent } from "../components/format";
import { riskForConfidence } from "../api/client";
import { BasicOverviewPage } from "./basic";

export function OverviewPage() {
  const { filters } = useFilters();
  return filters.audience === "basic" ? <BasicOverviewPage /> : <ExpertOverviewPage />;
}

function ExpertOverviewPage() {
  const { filters, resolvedModel } = useFilters();
  const live = useLiveMarket(filters.symbol);
  const regime = useRegime(filters.symbol, filters.mode);
  const meta = useMeta(filters.mode);
  const researchForecast = useForecast(filters, resolvedModel, filters.mode === "research");
  const mutation = useForecastMutation();
  const history = useForecastHistory(filters, resolvedModel, filters.mode === "research");
  const forecast = filters.mode === "research" ? researchForecast.data : mutation.data;
  const risk = riskForConfidence(forecast?.var_es, filters.confidence, "filtered_historical_simulation")
    ?? riskForConfidence(forecast?.var_es, filters.confidence);
  const request: ForecastRequest = { symbol: filters.symbol, model: resolvedModel ?? "har", horizon_days: filters.horizon, mode: filters.mode, confidence_levels: [0.95, 0.975, 0.99] };

  return <div className="page-stack">
    <PageIntro eyebrow="KẾT QUẢ NGHIÊN CỨU" title="Tổng quan kết quả" description="Dùng các mục dưới đây để trình bày lần lượt toàn bộ bảng và hình nghiên cứu. Các kết quả này được tính trên dữ liệu lịch sử đã lưu." actions={<Provenance />} />
    <nav className="research-guide" aria-label="Lộ trình xem kết quả nghiên cứu">
      <Link to="/evaluation" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}><strong>1 · So sánh mô hình</strong><span>T3–T6 · Sai số, kiểm định và độ hiệu chỉnh; bắt đầu bằng so sánh với HAR.</span></Link>
      <Link to="/risk" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}><strong>2 · Kiểm tra rủi ro</strong><span>T7–T8 · Tần suất vượt VaR và các bài kiểm tra Expected Shortfall.</span></Link>
      <Link to="/regimes" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}><strong>3 · Trạng thái và chiến lược</strong><span>T9–T10 · Kết quả theo trạng thái thị trường và chiến lược kinh tế.</span></Link>
      <Link to="/method" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}><strong>4 · Dữ liệu và phương pháp</strong><span>T1–T2, T11 và F1–F9 · Mô tả dữ liệu, kiểm tra độ bền và toàn bộ hình nghiên cứu.</span></Link>
    </nav>
    <div className="overview-grid">
      <StatTile label={`GIÁ NẾN 5 PHÚT · ${filters.symbol}`} value={live.data?.candle ? `$${formatNumber(Number(live.data.candle.price), 2)}` : "—"} detail={live.data?.candle ? <>Nến gần nhất · {formatDate(live.data.candle.close_time_utc ?? live.data.candle.open_time_utc, true)}</> : live.isLoading ? "Đang nhận dữ liệu Binance…" : "Chưa có nến trực tiếp"} tone="blue" />
      <StatTile label={`BIẾN ĐỘNG DỰ BÁO · ${filters.horizon} NGÀY`} value={forecast ? formatNumber(forecast.rv_forecast, 4) : "—"} detail={forecast ? "Phương sai dự báo · phần trăm bình phương" : filters.mode === "production" ? "Chạy dự báo để xem giá trị mới" : "Chưa có dự báo phù hợp"} />
      <StatTile label={`NGƯỠNG LỖ VaR · ${Math.round(filters.confidence * 1000) / 10}%`} value={risk ? formatPercent(risk.var_loss_percent) : "—"} detail={risk ? `ES · lỗ trung bình nhóm kỳ xấu: ${formatPercent(risk.expected_shortfall_loss_percent)}` : "Chưa có ước lượng rủi ro"} tone="red" />
      <StatTile label="TRẠNG THÁI THỊ TRƯỜNG" value={regime.data?.regime ?? "—"} detail={regime.data ? `Dữ liệu đến ${formatDate(regime.data.latest_available_date_utc)}` : "Chưa có nhãn trạng thái"} />
    </div>
    <div className="dashboard-banner">
      <div className="banner-icon"><Waves size={19} /></div>
      <div><strong>{filters.mode === "research" ? "Kiểm tra dự báo theo từng mốc quá khứ" : "Dữ liệu hiện tại từ Binance"}</strong><p>{filters.mode === "research" ? "Mỗi dự báo chỉ dùng dữ liệu có sẵn tại thời điểm đó rồi được đối chiếu với kết quả thực tế sau này. Đây là kiểm tra lịch sử, không phải dự báo trực tiếp." : "Dự báo dùng những ngày dữ liệu Binance đã hoàn tất gần nhất."}</p></div>
      {filters.mode === "production" && <button className="primary-button" onClick={() => mutation.mutate(request)} disabled={mutation.isPending || !resolvedModel}><RefreshCw size={15} className={mutation.isPending ? "spin" : ""} />{mutation.isPending ? "Đang tính…" : "Tạo forecast"}</button>}
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
