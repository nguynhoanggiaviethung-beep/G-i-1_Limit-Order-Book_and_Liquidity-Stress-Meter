export const numberFormat = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 4 });
export const percentFormat = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 3 });

export function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("vi-VN", { maximumFractionDigits: digits }).format(value);
}

export function formatPercent(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "—" : `${formatNumber(value, digits)}%`;
}

export function formatDate(value: string | null | undefined, withTime = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("vi-VN", withTime
    ? { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }
    : { dateStyle: "medium", timeZone: "UTC" }).format(date) + (withTime ? " UTC" : "");
}

export function shortHash(value: string | null | undefined) {
  return value ? `${value.slice(0, 8)}…${value.slice(-4)}` : "chưa có";
}
