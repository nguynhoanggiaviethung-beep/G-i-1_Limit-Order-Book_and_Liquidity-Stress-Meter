import { Play, RefreshCw } from "lucide-react";
import { useForecast, useForecastHistory, useForecastMutation } from "../api/queries";
import { useFilters } from "../app/filters";
import { ForecastPlot } from "../components/charts/ForecastPlot";
import { ApiState, PageIntro, SectionHeader, StatTile } from "../components/PageParts";
import { formatDate, formatNumber, formatPercent } from "../components/format";
import { riskForConfidence } from "../api/client";
import type { ForecastRequest } from "../api/types";
import { ModelSelector } from "../components/ModelSelector";
import { RiskExceptionPlot } from "../components/charts/RiskExceptionPlot";
import { RiskFanChart } from "../components/charts/RiskFanChart";

export function ForecastPage() {
  const { filters, resolvedModel } = useFilters();
  const mutation = useForecastMutation();
  const researchForecast = useForecast(filters, resolvedModel, filters.mode === "research");
  const history = useForecastHistory(filters, resolvedModel, Boolean(resolvedModel));
  const forecast = filters.mode === "research" ? researchForecast.data : mutation.data;
  const risk = riskForConfidence(forecast?.var_es, filters.confidence, "filtered_historical_simulation")
    ?? riskForConfidence(forecast?.var_es, filters.confidence);
  const request: ForecastRequest = { symbol: filters.symbol, model: resolvedModel ?? "har", horizon_days: filters.horizon, mode: filters.mode, confidence_levels: [0.95, 0.975, 0.99] };
  return <div className="page-stack">
    <PageIntro eyebrow="DỰ BÁO / 02" title="Dự báo theo kỳ hạn" description={filters.mode === "research" ? "Xem dự báo đã tạo tại từng mốc trong dữ liệu lịch sử và đối chiếu với kết quả thực tế khi đã có." : "Chạy mô hình trên dữ liệu Binance mới nhất đã hoàn tất."} actions={filters.mode === "production" ? <button className="primary-button" onClick={() => mutation.mutate(request)} disabled={mutation.isPending || !resolvedModel}><Play size={14} />{mutation.isPending ? "Đang chạy mô hình…" : "Tạo dự báo"}</button> : undefined} />
    <ModelSelector />
        <div className="forecast-context"><span className="context-chip">{filters.symbol}</span><span className="context-chip">{resolvedModel ?? "Đang chọn mô hình"}</span><span className="context-chip">{filters.horizon} ngày</span><span className={`context-chip ${filters.mode === "production" ? "chip-blue" : ""}`}>{filters.mode === "production" ? "Dự báo hiện tại" : "Kết quả nghiên cứu"}</span></div>
    <ApiState loading={filters.mode === "research" && researchForecast.isLoading} error={filters.mode === "research" ? researchForecast.error : mutation.error}>
      {forecast && <>
        <div className="overview-grid">
          <StatTile label="RV DỰ BÁO" value={formatNumber(forecast.rv_forecast, 5)} detail="Đơn vị: phần trăm bình phương" tone="blue" />
          <StatTile label="SIGMA XẤP XỈ" value={formatPercent(forecast.sigma_h_approx, 4)} detail="Độ lệch chuẩn lợi suất theo horizon" />
          <StatTile label="VaR" value={risk ? formatPercent(risk.var_loss_percent) : "—"} detail={risk?.method ?? "Không có estimate phù hợp"} tone="red" />
          <StatTile label="EXPECTED SHORTFALL" value={risk ? formatPercent(risk.expected_shortfall_loss_percent) : "—"} detail={`${Math.round(filters.confidence * 100)}% confidence`} tone="red" />
        </div>
        <div className="two-column-grid">
          <section className="info-panel"><div className="panel-heading"><h3>Thời điểm dự báo</h3><span className="badge">{forecast.status}</span></div><dl className="detail-list"><div><dt>Origin UTC</dt><dd>{formatDate(forecast.origin_time, true)}</dd></div><div><dt>Khoảng mục tiêu</dt><dd>{formatDate(forecast.target_start)} — {formatDate(forecast.target_end)}</dd></div><div><dt>Actual RV</dt><dd>{forecast.actual_rv_mean == null ? "Chưa quan sát" : formatNumber(forecast.actual_rv_mean, 5)}</dd></div><div><dt>Feed gần nhất</dt><dd>{forecast.source_data_as_of_utc ? formatDate(forecast.source_data_as_of_utc, true) : "Snapshot research"}</dd></div></dl></section>
          <section className="info-panel"><div className="panel-heading"><h3>Định danh kết quả</h3><span className="mono muted">{forecast.mode}</span></div><dl className="detail-list"><div><dt>Forecast ID</dt><dd className="mono">{forecast.forecast_id}</dd></div><div><dt>Data version</dt><dd className="mono">{forecast.data_version?.slice(0, 20) ?? "—"}</dd></div><div><dt>Config hash</dt><dd className="mono">{forecast.config_hash?.slice(0, 20) ?? "—"}</dd></div></dl></section>
        </div>
        {forecast.note && <p className="note-line">{forecast.note}</p>}
        {risk && <RiskFanChart estimates={forecast.var_es ?? []} preferredMethod={risk.method} />}
      </>}
    </ApiState>
    {filters.horizon === 30 && <div className="basic-warning" role="note">Kỳ hạn 30 ngày chỉ mang tính mô tả; các khoảng dự báo chồng lấn và không đại diện cho nhiều quan sát độc lập.</div>}
    <SectionHeader title="Chuỗi lịch sử" note="Đường dự báo và quan sát chỉ hiển thị khi backend có dữ liệu tương ứng." />
    <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !history.data?.points.length}>
      {history.data && <ForecastPlot points={history.data.points} />}
    </ApiState>
    <SectionHeader title="Lợi suất và ngưỡng VaR" note="Tối đa 60 ngày có đủ cả lợi suất thực tế và VaR.">
      {risk && <span className="badge">{risk.method} · {Math.round(filters.confidence * 1000) / 10}%</span>}
    </SectionHeader>
    <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !history.data?.points.some((point) => point.actual_return_percent != null && point.var_loss_percent != null)}>
      {history.data && <RiskExceptionPlot points={history.data.points} />}
    </ApiState>
    {mutation.isPending && <div className="state-box"><RefreshCw className="spin" size={16} /> Backend đang tải dữ liệu và chạy mô hình. Bạn có thể tiếp tục xem các trang khác sau khi hoàn tất.</div>}
  </div>;
}
