import { Play, RefreshCw } from "lucide-react";
import { useForecastHistory, useForecastMutation } from "../api/queries";
import { useFilters } from "../app/filters";
import { ForecastPlot } from "../components/charts/ForecastPlot";
import { ApiState, PageIntro, SectionHeader, StatTile } from "../components/PageParts";
import { formatDate, formatNumber, formatPercent } from "../components/format";
import { riskForConfidence } from "../api/client";
import type { ForecastRequest } from "../api/types";

export function ForecastPage() {
  const { filters } = useFilters();
  const mutation = useForecastMutation();
  const history = useForecastHistory(filters);
  const forecast = mutation.data;
  const risk = riskForConfidence(forecast?.var_es, filters.confidence);
  const request: ForecastRequest = { symbol: filters.symbol, model: filters.model, horizon_days: filters.horizon, mode: filters.mode, confidence_levels: [filters.confidence] };
  return <div className="page-stack">
    <PageIntro eyebrow="DỰ BÁO / 02" title="Forecast theo kỳ hạn" description="Tạo hoặc đọc dự báo trực tiếp từ backend. Đổi chế độ, tài sản, mô hình và kỳ hạn ở thanh trên cùng." actions={<button className="primary-button" onClick={() => mutation.mutate(request)} disabled={mutation.isPending}><Play size={14} />{mutation.isPending ? "Đang chạy mô hình…" : "Tạo dự báo"}</button>} />
    <div className="forecast-context"><span className="context-chip">{filters.symbol}</span><span className="context-chip">{filters.model}</span><span className="context-chip">{filters.horizon} ngày</span><span className={`context-chip ${filters.mode === "production" ? "chip-blue" : ""}`}>{filters.mode}</span></div>
    <ApiState error={mutation.error}>
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
      </>}
    </ApiState>
    <SectionHeader title="Chuỗi lịch sử" note="Đường dự báo và quan sát chỉ hiển thị khi backend có dữ liệu tương ứng." />
    <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !history.data?.points.length}>
      {history.data && <ForecastPlot points={history.data.points} />}
    </ApiState>
    {mutation.isPending && <div className="state-box"><RefreshCw className="spin" size={16} /> Backend đang tải dữ liệu và chạy mô hình. Bạn có thể tiếp tục xem các trang khác sau khi hoàn tất.</div>}
  </div>;
}
