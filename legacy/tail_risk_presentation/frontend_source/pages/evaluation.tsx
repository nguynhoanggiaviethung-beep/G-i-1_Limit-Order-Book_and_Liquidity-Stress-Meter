import { useState } from "react";
import { BarChart3, Info } from "lucide-react";
import { useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { DataTable } from "../components/DataTable";
import { MetricBarChart } from "../components/charts/EchartsCharts";
import { ApiState, PageIntro, SectionHeader } from "../components/PageParts";
import { Provenance } from "../components/PageParts";
import { formatNumber } from "../components/format";
import { modelLabel } from "../api/modelSelection";

const evaluationTables = [
  { id: "T3", label: "T3 · So sánh dự báo" },
  { id: "T4", label: "T4 · Kiểm tra khác biệt" },
  { id: "T5", label: "T5 · Nhóm mô hình tốt" },
  { id: "T6", label: "T6 · Độ hiệu chỉnh" },
];

export function EvaluationPage() {
  const [selected, setSelected] = useState("T3");
  const { filters } = useFilters();
  const table = useTable(selected);
  const comparisonRows = selected === "T3" ? table.data?.rows
    .filter((row) => row.symbol === filters.symbol && Number(row.horizon_days) === filters.horizon)
    .map((row) => ({ model: row.model ?? "", qlike: Number(row.qlike), nObs: Number(row.n_obs) }))
    .filter((row) => Number.isFinite(row.qlike)) ?? [] : [];
  const har = comparisonRows.find((row) => row.model === "har");
  const comparableRows = har && Number.isFinite(har.nObs)
    ? comparisonRows.filter((row) => row.nObs === har.nObs)
    : comparisonRows;
  const best = [...comparableRows].sort((a, b) => a.qlike - b.qlike)[0];
  const relativeGain = har && best && har.qlike !== 0 ? (har.qlike - best.qlike) / Math.abs(har.qlike) * 100 : undefined;
  const explanation = selected === "T3"
    ? har && best
      ? `T3 so sánh các mô hình trên ${filters.symbol}, kỳ hạn ${filters.horizon} ngày. HAR là mốc tham chiếu truyền thống (QLIKE ${formatNumber(har.qlike, 4)} trên ${Math.round(har.nObs).toLocaleString("vi-VN")} kỳ); ${best.model === "har" ? "HAR có điểm thấp nhất trong nhóm cùng số kỳ kiểm tra" : `${modelLabel(best.model)} có QLIKE thấp nhất trong nhóm cùng ${Math.round(har.nObs).toLocaleString("vi-VN")} kỳ`}${relativeGain != null ? `, thấp hơn mốc HAR khoảng ${formatNumber(relativeGain, 1)}%` : ""}. Điểm QLIKE thấp hơn cho biết dự báo biến động sát số liệu thực tế hơn trong giai đoạn kiểm tra này, không đảm bảo sẽ luôn tốt hơn về sau.`
      : "T3 là bảng so sánh chính. HAR đóng vai trò mốc tham chiếu truyền thống; QLIKE càng thấp thì dự báo biến động càng sát dữ liệu thực tế trong giai đoạn kiểm tra. Cột số kỳ cho biết lượng dữ liệu được dùng để chấm từng mô hình."
    : selected === "T4"
      ? "T4 kiểm tra sai số của từng mô hình so với mốc random walk. Giá trị p đã hiệu chỉnh Holm giúp hạn chế kết luận nhầm khi kiểm tra nhiều mô hình; bảng này không phải so sánh từng cặp mô hình với nhau."
      : selected === "T5"
        ? "T5 cho biết mô hình nào còn nằm trong nhóm có chất lượng dự báo tương đương tốt nhất theo thủ tục MCS. Được đưa vào nhóm không có nghĩa mô hình chắc chắn thắng; hãy đọc kết quả cùng điểm QLIKE ở T3."
        : "T6 kiểm tra mức khớp giữa biến động dự báo và biến động quan sát được. Hệ số chặn và độ dốc mô tả độ lệch có hệ thống; kiểm định chung cho biết hai hệ số có phù hợp với mức lý tưởng hay không.";
  return <div className="page-stack">
    <PageIntro eyebrow="ĐÁNH GIÁ / 03" title="So sánh mô hình" description="Bắt đầu từ T3 để xem mô hình nào dự báo biến động sát dữ liệu thực tế hơn HAR; T4–T6 kiểm tra độ khác biệt và độ phù hợp của dự báo." actions={<Provenance compact />} />
    <div className="callout"><Info size={16} /><span>{explanation}</span></div>
    {filters.horizon === 30 && <div className="basic-warning" role="note"><Info size={16} /><span>Kỳ hạn 30 ngày chỉ mang tính mô tả. Các p-value và quyết định kiểm định đã được ẩn vì số quan sát độc lập thấp và cửa sổ dự báo chồng lấn.</span></div>}
    <SectionHeader title="Kết quả cho lựa chọn hiện tại" note={`${filters.symbol} · ${filters.horizon} ngày · kết quả nghiên cứu đã lưu`}><span className="table-total"><BarChart3 size={15} /> T3–T6</span></SectionHeader>
    <div className="tab-list" role="tablist" aria-label="Bảng đánh giá mô hình">
      {evaluationTables.map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    <ApiState loading={table.isLoading} error={table.error}>
      {table.data && <>{selected === "T3" && <MetricBarChart table={table.data} title="Sai số QLIKE · thấp hơn là tốt hơn" />}<DataTable table={table.data} filterModel={false} descriptive={filters.horizon === 30} sortBy={selected === "T3" ? "qlike" : undefined} /></>}
    </ApiState>
  </div>;
}
