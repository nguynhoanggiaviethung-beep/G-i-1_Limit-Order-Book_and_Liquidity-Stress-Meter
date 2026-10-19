import * as echarts from "echarts/core";
import { BarChart, HeatmapChart } from "echarts/charts";
import { GridComponent, TooltipComponent, VisualMapComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/lib/core";
import type { CsvTable } from "../../api/types";
import { useFilters } from "../../app/filters";
import { filterTable } from "../DataTable";

echarts.use([BarChart, HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

export function MetricBarChart({ table, title }: { table: CsvTable; title: string }) {
  const { filters } = useFilters();
  const visible = filterTable(table, filters);
  const category = visible.headers.find((header) => /model|regime|strategy|method/i.test(header)) ?? visible.headers[0];
  const metric = visible.headers.find((header) => /qlike|rmse|mae|coverage|violation|score|loss/i.test(header) && header !== category)
    ?? visible.headers.find((header) => header !== category);
  const data = (visible.rows ?? []).slice(0, 12).map((row) => ({ name: row[category] ?? "—", value: Number(row[metric ?? ""]) })).filter((item) => Number.isFinite(item.value));
  const option = {
    animation: false,
    grid: { left: 150, right: 24, top: 12, bottom: 24 },
    tooltip: { trigger: "axis" as const, axisPointer: { type: "shadow" as const } },
    xAxis: { type: "value" as const, axisLabel: { color: "#656862" }, splitLine: { lineStyle: { color: "#e5e3dd" } } },
    yAxis: { type: "category" as const, data: data.map((item) => item.name).reverse(), axisLabel: { color: "#343632", width: 130, overflow: "truncate" as const }, axisLine: { show: false }, axisTick: { show: false } },
    series: [{ type: "bar" as const, data: data.map((item) => item.value).reverse(), itemStyle: { color: "#316fca" }, barMaxWidth: 19 }],
  };
  return <div className="chart-panel"><div className="panel-heading"><h3>{title}</h3><span className="mono muted">{metric || ""}</span></div>
    {data.length ? <ReactEChartsCore echarts={echarts} option={option} style={{ height: Math.max(240, data.length * 30) }} notMerge lazyUpdate /> : <div className="chart-empty">Bảng chưa có cột số để vẽ biểu đồ.</div>}
  </div>;
}

export function RegimeHeatmap({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts);
  const option = {
    animation: false,
    grid: { left: 110, right: 25, top: 18, bottom: 28 },
    tooltip: { position: "top" as const },
    xAxis: { type: "category" as const, data: ["Số ngày"], axisLabel: { color: "#656862" }, splitArea: { show: true } },
    yAxis: { type: "category" as const, data: entries.map(([name]) => name), axisLabel: { color: "#343632" }, splitArea: { show: true } },
    visualMap: { min: 0, max: Math.max(1, ...entries.map(([, count]) => count)), calculable: false, orient: "horizontal" as const, left: "center", bottom: 0, inRange: { color: ["#e8edf5", "#316fca"] } },
    series: [{ type: "heatmap" as const, data: entries.map(([, count], index) => [0, index, count]), label: { show: true, color: "#20211f" }, emphasis: { itemStyle: { shadowBlur: 0 } } }],
  };
  return <div className="chart-panel"><div className="panel-heading"><h3>Số ngày theo trạng thái thị trường</h3><span className="mono muted">quan sát sẵn có</span></div>
    {entries.length ? <ReactEChartsCore echarts={echarts} option={option} style={{ height: Math.max(190, entries.length * 45) }} notMerge lazyUpdate /> : <div className="chart-empty">Chưa có nhãn regime.</div>}
  </div>;
}
