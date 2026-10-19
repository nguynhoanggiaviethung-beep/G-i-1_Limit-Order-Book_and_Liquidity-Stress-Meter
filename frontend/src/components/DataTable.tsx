import { useMemo } from "react";
import { flexRender, getCoreRowModel, getPaginationRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";
import { Download, ChevronLeft, ChevronRight } from "lucide-react";
import type { CsvTable } from "../api/types";
import { useFilters } from "../app/filters";

export function downloadCsv(table: CsvTable) {
  const escape = (value: string) => `"${value.replaceAll('"', '""')}"`;
  const content = [table.headers, ...table.rows.map((row) => table.headers.map((header) => row[header] ?? ""))]
    .map((row) => row.map(escape).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob(["\uFEFF", content], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = table.name; anchor.click(); URL.revokeObjectURL(url);
}

export function filterTable(table: CsvTable, filters: { symbol: string; horizon: number; model: string; confidence: number }): CsvTable {
  const headers = new Set(table.headers.map((header) => header.toLowerCase()));
  return {
    ...table,
    rows: table.rows.filter((row) => {
      const lower = Object.fromEntries(Object.entries(row).map(([key, value]) => [key.toLowerCase(), value]));
      if (headers.has("symbol") && lower.symbol && lower.symbol !== filters.symbol) return false;
      if (headers.has("horizon_days") && lower.horizon_days && Number(lower.horizon_days) !== filters.horizon) return false;
      if (headers.has("model") && lower.model && lower.model !== filters.model) return false;
      if (headers.has("confidence") && lower.confidence && Math.abs(Number(lower.confidence) - filters.confidence) > 0.0001) return false;
      return true;
    }),
  };
}

export function DataTable({ table, maxHeight = 460 }: { table: CsvTable; maxHeight?: number }) {
  const { filters } = useFilters();
  const visibleTable = filterTable(table, filters);
  const columns = useMemo<ColumnDef<Record<string, string>>[]>(() => visibleTable.headers.map((header) => ({
    accessorKey: header,
    header: header.replaceAll("_", " "),
    cell: (context) => {
      const value = context.getValue<string>();
      return <span title={value}>{value || "—"}</span>;
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
