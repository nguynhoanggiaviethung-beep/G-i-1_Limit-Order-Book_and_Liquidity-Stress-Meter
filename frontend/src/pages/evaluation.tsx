import { useState } from "react";
import { BarChart3, Info } from "lucide-react";
import { useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { DataTable } from "../components/DataTable";
import { MetricBarChart } from "../components/charts/EchartsCharts";
import { ApiState, PageIntro, SectionHeader } from "../components/PageParts";
import { Provenance } from "../components/PageParts";

const evaluationTables = [
  { id: "T3", label: "T3 · Volatility OOS" },
  { id: "T4", label: "T4 · Diebold–Mariano" },
  { id: "T5", label: "T5 · MCS" },
  { id: "T6", label: "T6 · Mincer–Zarnowitz" },
];

export function EvaluationPage() {
  const [selected, setSelected] = useState("T3");
  const { filters } = useFilters();
  const table = useTable(selected);
  return <div className="page-stack">
    <PageIntro eyebrow="ĐÁNH GIÁ / 03" title="So sánh mô hình" description="Bảng kết quả walk-forward và kiểm định thống kê được nạp từ artifacts của backend." actions={<Provenance compact />} />
    <div className="callout"><Info size={16} /><span>Các bảng là kết quả đã xuất từ pipeline nghiên cứu. Mốc dữ liệu và cấu hình được hiển thị ở trên; trình duyệt không tự tính lại metric.</span></div>
    {filters.horizon === 30 && <div className="basic-warning" role="note"><Info size={16} /><span>Kỳ hạn 30 ngày chỉ mang tính mô tả. Các p-value và quyết định kiểm định đã được ẩn vì số quan sát độc lập thấp và cửa sổ dự báo chồng lấn.</span></div>}
    {selected === "T4" && <p className="note-line">T4 hiện xuất kiểm định Diebold–Mariano/HLN của từng mô hình so với random walk. Backend chưa xuất ma trận so sánh từng cặp, nên giao diện không suy diễn thêm các ô.</p>}
    <SectionHeader title="Bảng đánh giá" note={`${filters.symbol} · ${filters.horizon} ngày · ${filters.mode}`}><span className="table-total"><BarChart3 size={15} /> T3–T6</span></SectionHeader>
    <div className="tab-list" role="tablist" aria-label="Bảng đánh giá mô hình">
      {evaluationTables.map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    <ApiState loading={table.isLoading} error={table.error}>
      {table.data && <>{selected === "T3" && <MetricBarChart table={table.data} title="QLIKE OOS · thấp hơn là tốt hơn" />}<DataTable table={table.data} filterModel={false} descriptive={filters.horizon === 30} sortBy={selected === "T3" ? "qlike" : undefined} /></>}
    </ApiState>
  </div>;
}
