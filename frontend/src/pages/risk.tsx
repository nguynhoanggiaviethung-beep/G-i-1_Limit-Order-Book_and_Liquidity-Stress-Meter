import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import { useForecastHistory, useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { ForecastPlot } from "../components/charts/ForecastPlot";
import { DataTable } from "../components/DataTable";
import { ApiState, PageIntro, SectionHeader } from "../components/PageParts";
import { formatPercent } from "../components/format";

export function RiskPage() {
  const [selected, setSelected] = useState("T7");
  const { filters } = useFilters();
  const table = useTable(selected);
  const history = useForecastHistory(filters);
  const points = history.data?.points ?? [];
  const violations = points.filter((point) => point.actual_loss_percent != null && point.var_loss_percent != null && point.actual_loss_percent > point.var_loss_percent);
  const coverage = points.length ? (1 - violations.length / points.length) * 100 : null;
  return <div className="page-stack">
    <PageIntro eyebrow="RỦI RO / 04" title="VaR, ES & backtest" description="Đo lường loss tail và so sánh coverage từ các kết quả point-in-time đã lưu." actions={<span className="badge badge-red"><ShieldAlert size={13} /> LOSS-POSITIVE</span>} />
    <div className="overview-grid three-tiles">
      <article className="stat-tile"><div className="stat-label">MẪU HIỆN TRONG CHUỖI</div><div className="stat-value">{points.length ? points.length.toLocaleString("vi-VN") : "—"}</div><div className="stat-detail">{filters.symbol} · {filters.horizon} ngày</div></article>
      <article className="stat-tile tone-red"><div className="stat-label">VƯỢT NGƯỠNG VaR</div><div className="stat-value">{points.length ? violations.length.toLocaleString("vi-VN") : "—"}</div><div className="stat-detail">Số lần actual loss lớn hơn VaR</div></article>
      <article className="stat-tile tone-blue"><div className="stat-label">COVERAGE THỰC NGHIỆM</div><div className="stat-value">{coverage == null ? "—" : formatPercent(coverage)}</div><div className="stat-detail">Tính từ chuỗi matching model/risk filter</div></article>
    </div>
    <div className="callout warning-callout"><ShieldAlert size={16} /><span>Backtest chỉ tính trên các quan sát có actual loss và VaR cùng kỳ. Mức coverage tự mô tả ở đây không thay thế Kupiec, Christoffersen hoặc kiểm định ES trong bảng xuất chính thức.</span></div>
    <SectionHeader title="VaR & Expected Shortfall" note="Các ước lượng và kiểm định chính thức do pipeline backend xuất." />
    <div className="tab-list" role="tablist" aria-label="Bảng rủi ro">
      {[{ id: "T7", label: "T7 · VaR coverage" }, { id: "T8", label: "T8 · ES backtests" }].map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    <ApiState loading={table.isLoading} error={table.error}>{table.data && <DataTable table={table.data} />}</ApiState>
    <SectionHeader title="Lịch sử VaR đã ghép" note="Nến dự báo/loss được ghép theo origin UTC và phương pháp đã lưu." />
    <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !points.length}>{history.data && <ForecastPlot points={points} title="RV và forecast theo thời gian" />}</ApiState>
  </div>;
}
