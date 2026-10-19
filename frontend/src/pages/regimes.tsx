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
    <PageIntro eyebrow="REGIME / 05" title="Chế độ thị trường & chiến lược" description="Nhãn trạng thái được tạo theo quy tắc nghiên cứu và được chặn theo thời điểm dữ liệu sẵn có." actions={<span className="badge"><CircleGauge size={13} /> POINT-IN-TIME</span>} />
    <ApiState loading={regime.isLoading} error={regime.error}>
      {regime.data && <>
        <div className="overview-grid three-tiles">
          <StatTile label="REGIME GẦN NHẤT" value={regime.data.regime} detail={filters.symbol} tone="blue" />
          <StatTile label="NGÀY FEATURE SẴN CÓ" value={formatDate(regime.data.latest_available_date_utc)} detail={`as of ${formatDate(regime.data.as_of_utc, true)}`} />
          <StatTile label="NGUỒN" value={regime.data.source === "frozen_snapshot" ? "Snapshot" : "Rolling data"} detail={regime.data.mode} />
        </div>
        <RegimeHeatmap counts={regime.data.counts} />
        <div className="callout"><Info size={16} /><span>Regime dùng ngày feature cuối cùng đã có đủ dữ liệu tại mốc “as of”. Ngày UTC đang chạy không được gán nhãn trước khi hoàn tất.</span></div>
      </>}
    </ApiState>
    <SectionHeader title="Kết quả theo regime & chiến lược" note="Các bảng T9/T10 lấy trực tiếp từ báo cáo nghiên cứu." />
    <div className="tab-list" role="tablist" aria-label="Bảng regime và chiến lược">
      {[{ id: "T9", label: "T9 · Regime metrics" }, { id: "T10", label: "T10 · Economic strategies" }].map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    <ApiState loading={table.isLoading} error={table.error}>{table.data && <DataTable table={table.data} />}</ApiState>
  </div>;
}
