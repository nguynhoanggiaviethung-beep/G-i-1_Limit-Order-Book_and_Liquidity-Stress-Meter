import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import { useForecastHistory, useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { RiskExceptionPlot } from "../components/charts/RiskExceptionPlot";
import { DataTable } from "../components/DataTable";
import { ApiState, PageIntro, SectionHeader } from "../components/PageParts";

export function RiskPage() {
  const [selected, setSelected] = useState("T7");
  const { filters, resolvedModel } = useFilters();
  const table = useTable(selected);
  const history = useForecastHistory(filters, resolvedModel, Boolean(resolvedModel));
  const points = history.data?.points ?? [];
  return <div className="page-stack">
    <PageIntro eyebrow="RỦI RO / 04" title="VaR, ES & backtest" description="Đo lường loss tail và so sánh coverage từ các kết quả point-in-time đã lưu." actions={<span className="badge badge-red"><ShieldAlert size={13} /> LOSS-POSITIVE</span>} />
    <div className="callout"><ShieldAlert size={16} /><span>VaR và Expected Shortfall là kết quả đã xuất từ pipeline backend. Mức 30 ngày được xem là mô tả; p-value và quyết định kiểm định được ẩn vì các quan sát chồng lấn.</span></div>
    <SectionHeader title="VaR & Expected Shortfall" note="Các ước lượng và kiểm định chính thức do pipeline backend xuất." />
    <div className="tab-list" role="tablist" aria-label="Bảng rủi ro">
      {[{ id: "T7", label: "T7 · VaR coverage" }, { id: "T8", label: "T8 · ES backtests" }].map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    {filters.horizon === 30 && <div className="basic-warning" role="note">Kỳ hạn 30 ngày chỉ mang tính mô tả; không diễn giải kiểm định thống kê như bằng chứng kết luận.</div>}
    <ApiState loading={table.isLoading} error={table.error}>{table.data && <DataTable table={table.data} descriptive={filters.horizon === 30} />}</ApiState>
    <SectionHeader title="Lợi suất và ngưỡng VaR" note="Dấu × đánh dấu lần lợi suất vượt ngưỡng. Dữ liệu chỉ hiển thị khi cùng có actual và VaR." />
    <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !points.some((point) => point.actual_return_percent != null && point.var_loss_percent != null)}>{history.data && <RiskExceptionPlot points={points} />}</ApiState>
  </div>;
}
