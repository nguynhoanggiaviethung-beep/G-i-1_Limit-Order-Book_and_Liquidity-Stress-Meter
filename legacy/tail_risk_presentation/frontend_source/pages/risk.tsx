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
    <PageIntro eyebrow="RỦI RO / 04" title="Kiểm tra ngưỡng lỗ" description="Xem số lần khoản lỗ thực tế vượt VaR và các phép kiểm tra mức lỗ trung bình trong những kỳ xấu nhất." actions={<span className="badge badge-red"><ShieldAlert size={13} /> KẾT QUẢ LỊCH SỬ</span>} />
    <div className="callout"><ShieldAlert size={16} /><span>T7 đối chiếu số lần vượt VaR thực tế với số lần dự kiến theo mức tin cậy; T8 kiểm tra chất lượng ước lượng ES ở phần lỗ nghiêm trọng. Đây là kết quả trên dữ liệu quá khứ. Kỳ hạn 30 ngày chỉ mang tính mô tả vì các cửa sổ chồng lấn.</span></div>
    <SectionHeader title="Kết quả kiểm định rủi ro" note="Các bảng này được tính trước trong nghiên cứu và tải từ backend." />
    <div className="tab-list" role="tablist" aria-label="Bảng rủi ro">
      {[{ id: "T7", label: "T7 · Số lần vượt ngưỡng VaR" }, { id: "T8", label: "T8 · Kiểm tra mức lỗ ES" }].map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    {filters.horizon === 30 && <div className="basic-warning" role="note">Kỳ hạn 30 ngày chỉ mang tính mô tả; không diễn giải kiểm định thống kê như bằng chứng kết luận.</div>}
    <ApiState loading={table.isLoading} error={table.error}>{table.data && <DataTable table={table.data} descriptive={filters.horizon === 30} />}</ApiState>
    <SectionHeader title="Lỗ thực tế so với ngưỡng VaR" note="Tối đa 60 ngày có đủ cả lợi suất thực tế và VaR." />
    <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !points.some((point) => point.actual_return_percent != null && point.var_loss_percent != null)}>{history.data && <RiskExceptionPlot points={points} />}</ApiState>
  </div>;
}
