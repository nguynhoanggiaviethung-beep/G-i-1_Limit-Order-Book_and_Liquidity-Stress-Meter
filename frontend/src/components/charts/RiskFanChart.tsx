import type { RiskForecast } from "../../api/types";
import { formatPercent } from "../format";

const WIDTH = 680;
const HEIGHT = 205;
const LEFT = 52;
const RIGHT = 22;
const TOP = 22;
const BOTTOM = 38;
const levels = [0.95, 0.975, 0.99];

export function RiskFanChart({ estimates, preferredMethod }: { estimates: RiskForecast[]; preferredMethod: string }) {
  const methodRows = levels.map((confidence) => estimates.find((row) => row.confidence === confidence && row.method === preferredMethod));
  if (methodRows.some((row) => !row)) return <div className="state-box">Chưa đủ các mức tin cậy để vẽ fan chart.</div>;
  const rows = methodRows as RiskForecast[];
  const maximum = Math.max(...rows.map((row) => row.expected_shortfall_loss_percent), 0.1) * 1.15;
  const innerWidth = WIDTH - LEFT - RIGHT;
  const innerHeight = HEIGHT - TOP - BOTTOM;
  const x = (index: number) => LEFT + index / (levels.length - 1) * innerWidth;
  const y = (value: number) => TOP + (maximum - value) / maximum * innerHeight;
  const varPoints = rows.map((row, index) => `${x(index)},${y(row.var_loss_percent)}`);
  const esPoints = rows.map((row, index) => `${x(index)},${y(row.expected_shortfall_loss_percent)}`);
  const band = [...esPoints, ...varPoints.slice().reverse()].join(" ");
  return <section className="risk-fan-panel" aria-label="Fan chart rủi ro">
    <div className="panel-heading"><h3>Fan chart rủi ro</h3><span className="mono muted">{preferredMethod} · loss-positive</span></div>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby="fan-chart-title fan-chart-description">
      <title id="fan-chart-title">VaR và Expected Shortfall theo mức tin cậy</title>
      <desc id="fan-chart-description">Dải giữa VaR và Expected Shortfall ở các mức tin cậy 95, 97,5 và 99 phần trăm.</desc>
      {[0.25, 0.5, 0.75].map((fraction) => <line key={fraction} x1={LEFT} x2={WIDTH - RIGHT} y1={TOP + innerHeight * fraction} y2={TOP + innerHeight * fraction} className="risk-grid-line" />)}
      <polygon points={band} className="risk-fan-band" />
      <polyline points={varPoints.join(" ")} className="risk-var-line" />
      <polyline points={esPoints.join(" ")} className="risk-es-line" />
      {rows.map((row, index) => <g key={row.confidence}>
        <circle cx={x(index)} cy={y(row.var_loss_percent)} r="3" className="risk-var-dot" />
        <circle cx={x(index)} cy={y(row.expected_shortfall_loss_percent)} r="3" className="risk-es-dot" />
        <text x={x(index)} y={HEIGHT - 13} textAnchor="middle" className="risk-axis-label">{formatPercent(row.confidence * 100, 1)}</text>
      </g>)}
      <text x={LEFT} y={TOP - 7} className="risk-axis-label">Lỗ dự báo (%)</text>
    </svg>
    <div className="risk-chart-legend"><span><i className="legend-var" />VaR</span><span><i className="legend-es" />Expected Shortfall</span></div>
  </section>;
}
