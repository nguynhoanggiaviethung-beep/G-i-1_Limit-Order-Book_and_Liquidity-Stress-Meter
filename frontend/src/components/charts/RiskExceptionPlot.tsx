import type { ForecastPoint } from "../../api/types";
import { formatDate, formatPercent } from "../format";

const WIDTH = 1000;
const HEIGHT = 300;
const LEFT = 52;
const RIGHT = 20;
const TOP = 20;
const BOTTOM = 40;

export function RiskExceptionPlot({ points }: { points: ForecastPoint[] }) {
  const rows = points
    .filter((point) => point.actual_return_percent != null && point.var_loss_percent != null)
    .sort((a, b) => Date.parse(a.origin_time) - Date.parse(b.origin_time))
    .slice(-60);
  if (!rows.length) return <div className="chart-empty">Chưa có chuỗi lợi suất và VaR ghép theo cùng thời điểm.</div>;

  const values = rows.flatMap((point) => [point.actual_return_percent ?? 0, -(point.var_loss_percent ?? 0)]);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const spread = Math.max(max - min, 0.1);
  const yMin = min - spread * 0.12;
  const yMax = max + spread * 0.12;
  const innerWidth = WIDTH - LEFT - RIGHT;
  const innerHeight = HEIGHT - TOP - BOTTOM;
  const step = innerWidth / Math.max(rows.length, 1);
  const x = (index: number) => LEFT + step * (index + 0.5);
  const barWidth = Math.max(2, Math.min(16, step * 0.62));
  const y = (value: number) => TOP + (yMax - value) / (yMax - yMin) * innerHeight;
  const line = (getValue: (index: number) => number | null) => rows.map((_, index) => {
    const value = getValue(index);
    return value == null ? "" : `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
  }).filter(Boolean).join(" ");
  const varPath = line((index) => rows[index].var_loss_percent == null ? null : -rows[index].var_loss_percent);
  const exceptions = rows.flatMap((point, index) => point.actual_return_percent! < -point.var_loss_percent! ? [{ point, index }] : []);
  const gridValues = [yMin + (yMax - yMin) * 0.25, (yMin + yMax) / 2, yMin + (yMax - yMin) * 0.75];

  return <figure className="risk-exception-chart">
    <div className="risk-chart-legend"><span><i className="legend-actual" />Lãi / lỗ ngày</span><span><i className="legend-var" />Ngưỡng lỗ VaR (đường xanh)</span><span className="legend-exception">× Vượt ngưỡng</span></div>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby="risk-exception-title risk-exception-desc" preserveAspectRatio="none">
      <title id="risk-exception-title">Lãi lỗ ngày và ngưỡng VaR lịch sử</title>
      <desc id="risk-exception-desc">{rows.length} quan sát gần nhất; {exceptions.length} ngày lỗ vượt ngưỡng VaR. Cột thể hiện lợi suất ngày và các lần vượt ngưỡng được đánh dấu bằng dấu nhân màu đỏ.</desc>
      {gridValues.map((value, index) => <g key={index}><line x1={LEFT} x2={WIDTH - RIGHT} y1={y(value)} y2={y(value)} className="risk-grid-line" /><text x={LEFT - 8} y={y(value) + 3} textAnchor="end" className="risk-axis-label">{formatPercent(value, 1)}</text></g>)}
      <line x1={LEFT} x2={WIDTH - RIGHT} y1={y(0)} y2={y(0)} className="risk-zero-line" />
      <path d={varPath} className="risk-var-line" />
      {rows.map((point, index) => {
        const value = point.actual_return_percent ?? 0;
        const top = Math.min(y(0), y(value));
        const height = Math.max(1, Math.abs(y(value) - y(0)));
        const isException = value < -(point.var_loss_percent ?? 0);
        return <rect key={`${point.origin_time}-${index}`} x={x(index) - barWidth / 2} y={top} width={barWidth} height={height} className={isException ? "risk-return-exception" : value >= 0 ? "risk-return-positive" : "risk-return-negative"}>
          <title>{`${formatDate(point.origin_time)} · ${formatPercent(value, 2)}${isException ? " · vượt VaR" : ""}`}</title>
        </rect>;
      })}
      {exceptions.map(({ point, index }) => <text key={`${point.origin_time}-${index}`} x={x(index)} y={y(point.actual_return_percent!)} textAnchor="middle" dominantBaseline="central" className="risk-exception-mark">×</text>)}
      <text x={LEFT} y={HEIGHT - 10} className="risk-axis-label">{formatDate(rows[0].origin_time)}</text>
      <text x={WIDTH - RIGHT} y={HEIGHT - 10} textAnchor="end" className="risk-axis-label">{formatDate(rows[rows.length - 1].origin_time)}</text>
    </svg>
    <figcaption>Trong {rows.length.toLocaleString("vi-VN")} ngày đã kiểm tra gần nhất, mỗi cột là lợi suất thực tế. Đường xanh đứt nằm dưới mức 0 vì VaR được hiển thị thành ngưỡng lỗ; dấu × là ngày khoản lỗ vượt đường ngưỡng.</figcaption>
  </figure>;
}
