import { useMemo } from "react";
import { flexRender, getCoreRowModel, getPaginationRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";
import { Download, ChevronLeft, ChevronRight } from "lucide-react";
import type { CsvTable } from "../api/types";
import { useFilters } from "../app/filters";
import { modelLabel } from "../api/modelSelection";

const readableHeaders: Record<string, string> = {
  symbol: "Tài sản", horizon_days: "Kỳ hạn (ngày)", model: "Mô hình", benchmark: "Mốc so sánh", interval: "Khung dữ liệu",
  candle_rows: "Số nến", snapshot_data_version: "Phiên bản dữ liệu", quality_status: "Chất lượng dữ liệu", variant: "Biến thể kiểm tra",
  n_obs: "Số kỳ kiểm tra", n_independent_approx: "Số kỳ độc lập ước tính", n_forecast_floored: "Số dự báo được chặn",
  qlike: "QLIKE (thấp hơn tốt hơn)", mae: "Sai số tuyệt đối TB", mse: "Sai số bình phương TB", rmse: "Căn sai số bình phương TB",
  volatility_mae: "Sai số tuyệt đối biến động", volatility_rmse: "Sai số biến động", log_rv_mae: "Sai số tuyệt đối log-RV", log_rv_rmse: "Sai số log-RV",
  r2_oos_vs_benchmark: "R² ngoài mẫu so với mốc", mincer_zarnowitz: "Kết quả hiệu chỉnh", method: "Phương pháp", confidence: "Mức tin cậy",
  expected_exceedances: "Số lần vượt dự kiến", exceptions: "Số lần vượt thực tế", exception_rate: "Tỷ lệ vượt thực tế", quantile_loss: "Sai số phân vị",
  kupiec_pvalue: "Kiểm định Kupiec", christoffersen_cc_pvalue: "Kiểm định độc lập vượt ngưỡng", dq_pvalue: "Kiểm định động VaR",
  fz0_mean: "Điểm kiểm tra VaR/ES", mean_es: "ES trung bình", acerbi_szekely_pvalue: "Kiểm định ES", acerbi_szekely_interpretation: "Diễn giải kiểm định ES",
  pvalue: "Giá trị p", pvalue_holm_global: "Giá trị p đã hiệu chỉnh Holm", holm_reject_5pct: "Khác biệt có ý nghĩa 5%",
  mean_loss_difference: "Chênh lệch sai số trung bình", statistic: "Thống kê kiểm định", hln_corrected: "Đã hiệu chỉnh HLN",
  included: "Thuộc nhóm tốt nhất", status: "Trạng thái", regime: "Trạng thái thị trường", strategy: "Chiến lược",
  buy_and_hold: "Mua và giữ", utility_fee_vs_buy_and_hold_daily_percent: "Hiệu quả so với mua và giữ (%/ngày)", risk_method: "Cách tính rủi ro",
  evaluation_scope: "Phạm vi đánh giá", assumptions: "Giả định", table_source: "Nguồn bảng", budget_percent: "Ngân sách rủi ro (%)",
  margin_percent: "Tỷ lệ ký quỹ (%)", mean_margin_percent: "Ký quỹ trung bình (%)", max_margin_percent: "Ký quỹ cao nhất (%)",
  margin_breaches: "Số lần thiếu ký quỹ", breach_rate: "Tỷ lệ thiếu ký quỹ", mean_shortfall_percent: "Mức thiếu trung bình (%)",
  max_shortfall_percent: "Mức thiếu lớn nhất (%)", mean_daily_capital_cost_percent: "Chi phí vốn trung bình/ngày (%)",
  mean_absolute_margin_change_percent: "Mức thay đổi ký quỹ trung bình (%)", procyclicality_ratio: "Tỷ lệ biến động theo chu kỳ",
  fixed_mean_margin_percent: "Ký quỹ cố định trung bình (%)", fixed_max_margin_percent: "Ký quỹ cố định cao nhất (%)",
  fixed_margin_breaches: "Số lần thiếu ký quỹ cố định", fixed_breach_rate: "Tỷ lệ thiếu ký quỹ cố định",
  fixed_mean_shortfall_percent: "Mức thiếu trung bình với mức cố định (%)", fixed_max_shortfall_percent: "Mức thiếu lớn nhất với mức cố định (%)",
  fixed_mean_daily_capital_cost_percent: "Chi phí vốn/ngày với mức cố định (%)", fixed_mean_absolute_margin_change_percent: "Mức thay đổi ký quỹ cố định (%)",
  fixed_procyclicality_ratio: "Tỷ lệ chu kỳ với mức cố định", assumption_es_buffer: "Giả định đệm ES",
  assumption_fixed_margin_percent_of_notional: "Giả định ký quỹ cố định (%)", assumption_annual_capital_cost: "Giả định chi phí vốn năm",
  assumption_margin_floor_percent: "Mức ký quỹ tối thiểu giả định (%)", breaches: "Số lần vi phạm", mean_position_weight: "Tỷ trọng vị thế trung bình",
  floor_mean: "Mức sàn dự báo trung bình", reliability: "Độ tin cậy", warning: "Cảnh báo", offset: "Độ lệch ngày bắt đầu",
  n_obs_offset: "Số kỳ theo độ lệch",
  acerbi_szekely_z2: "Thống kê kiểm định ES", max_e_value: "E-value lớn nhất", e_backtest_reject_5pct: "Kiểm định ES bác bỏ ở 5%",
  es_evaluation_error: "Lỗi đánh giá ES", multiplicity_family: "Nhóm kiểm định",
  first_candle_utc: "Nến đầu tiên (UTC)", last_candle_close_utc: "Nến cuối (UTC)", missing_5m_intervals: "Khoảng nến 5 phút bị thiếu",
  complete_utc_days: "Số ngày UTC đầy đủ", n_complete_days: "Số ngày UTC đầy đủ", return_mean_percent: "Lợi suất trung bình (%)", return_std_percent: "Độ lệch chuẩn lợi suất (%)",
  return_skewness: "Độ lệch phân phối lợi suất", return_excess_kurtosis: "Độ nhọn vượt chuẩn", rv_mean_percent_squared: "RV trung bình (%²)",
  rv_median_percent_squared: "RV trung vị (%²)", rv_95pct_percent_squared: "RV phân vị 95% (%²)", rv_max_percent_squared: "RV lớn nhất (%²)",
};

function displayHeader(header: string) {
  return readableHeaders[header.toLowerCase()] ?? header.replaceAll("_", " ");
}

function displayCell(header: string, value: string) {
  if (!value) return "—";
  const key = header.toLowerCase();
  if (key === "model") return modelLabel(value);
  if (key === "symbol") return value === "BTCUSDT" ? "Bitcoin (BTCUSDT)" : value === "ETHUSDT" ? "Ethereum (ETHUSDT)" : value;
  if (["method", "risk_method"].includes(key)) {
    const methods: Record<string, string> = {
      filtered_historical_simulation: "Mô phỏng lịch sử có lọc",
      historical_simulation: "Mô phỏng lịch sử",
      param_normal: "Phân phối chuẩn",
      param_t: "Phân phối t",
      param_skewt: "Phân phối t lệch",
      evt: "Giá trị cực trị",
      linear_qr: "Hồi quy phân vị tuyến tính",
      lightgbm_quantile: "LightGBM phân vị",
      qrf: "Rừng ngẫu nhiên phân vị",
    };
    return methods[value] ?? value.replaceAll("_", " ");
  }
  if (key === "status" && value === "ok") return "Đã hoàn tất";
  if (["included", "holm_reject_5pct", "hln_corrected", "e_backtest_reject_5pct", "buy_and_hold"].includes(key)) {
    const yes = value.toLowerCase() === "true";
    const labels: Record<string, [string, string]> = {
      included: ["Có", "Không"],
      holm_reject_5pct: ["Có ý nghĩa thống kê", "Chưa đủ bằng chứng"],
      hln_corrected: ["Đã hiệu chỉnh", "Chưa hiệu chỉnh"],
      e_backtest_reject_5pct: ["Có dấu hiệu bác bỏ", "Không phát hiện dấu hiệu bác bỏ"],
      buy_and_hold: ["Có", "Không"],
    };
    if (value.toLowerCase() === "true" || value.toLowerCase() === "false") return labels[key][yes ? 0 : 1];
  }
  return value;
}

export function downloadCsv(table: CsvTable) {
  const escape = (value: string) => `"${value.replaceAll('"', '""')}"`;
  const content = [table.headers, ...table.rows.map((row) => table.headers.map((header) => row[header] ?? ""))]
    .map((row) => row.map(escape).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob(["\uFEFF", content], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = table.name; anchor.click(); URL.revokeObjectURL(url);
}

export function filterTable(table: CsvTable, filters: { symbol: string; horizon: number; model: string; confidence: number }, filterModel = true): CsvTable {
  const headers = new Set(table.headers.map((header) => header.toLowerCase()));
  return {
    ...table,
    rows: table.rows.filter((row) => {
      const lower = Object.fromEntries(Object.entries(row).map(([key, value]) => [key.toLowerCase(), value]));
      if (headers.has("symbol") && lower.symbol && lower.symbol !== filters.symbol) return false;
      if (headers.has("horizon_days") && lower.horizon_days && Number(lower.horizon_days) !== filters.horizon) return false;
      if (filterModel && headers.has("model") && lower.model && lower.model !== filters.model) return false;
      if (headers.has("confidence") && lower.confidence && Math.abs(Number(lower.confidence) - filters.confidence) > 0.0001) return false;
      return true;
    }),
  };
}

export function DataTable({ table, maxHeight = 460, filterModel = true, descriptive = false, sortBy }: { table: CsvTable; maxHeight?: number; filterModel?: boolean; descriptive?: boolean; sortBy?: string }) {
  const { filters, resolvedModel } = useFilters();
  const visible = filterTable(table, { ...filters, model: resolvedModel ?? "" }, filterModel);
  const preparedTable = descriptive ? {
    ...visible,
    headers: visible.headers.filter((header) => !/(pvalue|p_value|holm_reject|joint_pvalue|mincer_zarnowitz)/i.test(header)),
  } : visible;
  const visibleTable = sortBy ? {
    ...preparedTable,
    rows: [...preparedTable.rows].sort((a, b) => Number(a[sortBy]) - Number(b[sortBy])),
  } : preparedTable;
  const columns = useMemo<ColumnDef<Record<string, string>>[]>(() => visibleTable.headers.map((header) => ({
    accessorKey: header,
    header: displayHeader(header),
    cell: (context) => {
      const value = context.getValue<string>();
      const readableValue = displayCell(header, value);
      return <span title={readableValue}>{readableValue}</span>;
    },
  })), [visibleTable.headers]);
  const instance = useReactTable({ data: visibleTable.rows, columns, getCoreRowModel: getCoreRowModel(), getPaginationRowModel: getPaginationRowModel(), initialState: { pagination: { pageSize: 12 } } });
  return <section className="table-panel" aria-label={`Bảng ${table.name}`}>
    <div className="table-scroll" style={{ maxHeight }}>
      <table className="data-table">
        <thead>{instance.getHeaderGroups().map((group) => <tr key={group.id}>{group.headers.map((header) => <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>)}</tr>)}</thead>
        <tbody>
          {instance.getRowModel().rows.map((row) => <tr key={row.id}>{row.getVisibleCells().map((cell) => <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}
          {!visibleTable.rows.length && <tr><td className="empty-cell" colSpan={visibleTable.headers.length || 1}>Không có dòng phù hợp với bộ lọc hiện tại.</td></tr>}
        </tbody>
      </table>
    </div>
    <div className="table-footer"><span>{visibleTable.rows.length.toLocaleString("vi-VN")} / {table.rows.length.toLocaleString("vi-VN")} dòng · {table.name}</span><div className="table-actions">
      <button className="icon-button" aria-label="Trang trước" onClick={() => instance.previousPage()} disabled={!instance.getCanPreviousPage()}><ChevronLeft size={16} /></button>
      <span>Trang {instance.getState().pagination.pageIndex + 1}/{instance.getPageCount() || 1}</span>
      <button className="icon-button" aria-label="Trang sau" onClick={() => instance.nextPage()} disabled={!instance.getCanNextPage()}><ChevronRight size={16} /></button>
      <button className="text-button" onClick={() => downloadCsv(visibleTable)}><Download size={14} /> Tải CSV</button>
    </div></div>
  </section>;
}
