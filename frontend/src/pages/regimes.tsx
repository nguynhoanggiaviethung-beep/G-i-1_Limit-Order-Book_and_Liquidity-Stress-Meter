import { useState } from "react";
import { CircleGauge, Info } from "lucide-react";
import { useRegime, useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { RegimeHeatmap } from "../components/charts/EchartsCharts";
import { DataTable } from "../components/DataTable";
import { ApiState, PageIntro, SectionHeader, StatTile } from "../components/PageParts";
import { formatDate } from "../components/format";

export function RegimesPage() {
  const [selected, setSelected] = useState("T9");
  const { filters } = useFilters();
  const regime = useRegime(filters.symbol, filters.mode);
  const table = useTable(selected);
  return <div className="page-stack">
    <PageIntro eyebrow="THỊ TRƯỜNG / 05" title="Trạng thái thị trường và chiến lược" description="Xem số ngày thuộc từng trạng thái thị trường, rồi so sánh kết quả các chiến lược được đánh giá trong nghiên cứu." actions={<span className="badge"><CircleGauge size={13} /> THEO THỜI ĐIỂM</span>} />
    <ApiState loading={regime.isLoading} error={regime.error}>
      {regime.data && <>
        <div className="overview-grid three-tiles">
          <StatTile label="TRẠNG THÁI GẦN NHẤT" value={regime.data.regime} detail={filters.symbol} tone="blue" />
          <StatTile label="NGÀY DỮ LIỆU CUỐI" value={formatDate(regime.data.latest_available_date_utc)} detail={`tính đến ${formatDate(regime.data.as_of_utc, true)} UTC`} />
          <StatTile label="BỘ DỮ LIỆU" value={regime.data.source === "frozen_snapshot" ? "Nghiên cứu cố định" : "Dữ liệu cập nhật"} detail={regime.data.mode} />
        </div>
        <RegimeHeatmap counts={regime.data.counts} />
        <div className="callout"><Info size={16} /><span>Biểu đồ đếm số ngày dữ liệu thuộc từng trạng thái thị trường. Trạng thái mới nhất chỉ dùng dữ liệu đã hoàn tất tại mốc cập nhật; ngày đang diễn ra chưa được đưa vào phân loại.</span></div>
      </>}
    </ApiState>
    <SectionHeader title="Kết quả theo trạng thái và chiến lược" note="T9 mô tả kết quả theo trạng thái thị trường; T10 so sánh các chiến lược kinh tế." />
    <div className="tab-list" role="tablist" aria-label="Bảng regime và chiến lược">
      {[{ id: "T9", label: "T9 · Kết quả theo trạng thái" }, { id: "T10", label: "T10 · So sánh chiến lược" }].map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    {filters.horizon === 30 && <div className="basic-warning" role="note">Kết quả 30 ngày chỉ mang tính mô tả; không diễn giải p-value như một kết luận.</div>}
    <ApiState loading={table.isLoading} error={table.error}>{table.data && <DataTable table={table.data} descriptive={filters.horizon === 30} />}</ApiState>
  </div>;
}
