import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { ForecastPoint } from "../../api/types";

export function ForecastPlot({ points, title = "Độ biến động thực tế và dự báo" }: { points: ForecastPoint[]; title?: string }) {
  const host = useRef<HTMLDivElement>(null);
  const values = useMemo<uPlot.AlignedData>(() => {
    const sorted = [...points].sort((a, b) => Date.parse(a.origin_time) - Date.parse(b.origin_time));
    return [
      sorted.map((point) => Date.parse(point.origin_time) / 1000),
      sorted.map((point) => point.actual_rv_mean ?? null),
      sorted.map((point) => point.rv_forecast ?? null),
    ];
  }, [points]);
  useEffect(() => {
    if (!host.current || !values[0].length) return;
    const element = host.current;
    const plot = new uPlot({
      width: element.clientWidth,
      height: 270,
      scales: { x: { time: true } },
      axes: [{ stroke: "#777b77", grid: { stroke: "#e5e3dd", width: 1 } }, { stroke: "#777b77", grid: { stroke: "#e5e3dd", width: 1 } }],
      series: [
        {},
        { label: "RV thực tế", stroke: "#20211f", width: 1.5, points: { show: false } },
        { label: "Dự báo", stroke: "#316fca", width: 2, points: { show: false } },
      ],
      legend: { show: true },
    }, values, element);
    const resize = new ResizeObserver(() => plot.setSize({ width: element.clientWidth, height: 270 }));
    resize.observe(element);
    return () => { resize.disconnect(); plot.destroy(); };
  }, [values]);
  return <div className="chart-panel"><div className="panel-heading"><h3>{title}</h3><span className="mono muted">{points.length.toLocaleString("vi-VN")} điểm</span></div>
    {points.length ? <div ref={host} className="uplot-host" /> : <div className="chart-empty">Chưa có chuỗi dữ liệu cho bộ lọc này.</div>}
  </div>;
}
