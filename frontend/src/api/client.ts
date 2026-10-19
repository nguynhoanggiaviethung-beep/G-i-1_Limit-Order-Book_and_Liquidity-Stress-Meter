import type {
  AppMode,
  CsvTable,
  ForecastHistory,
  ForecastRequest,
  ForecastResponse,
  MetaResponse,
  RegimeResponse,
  RiskForecast,
} from "./types";

const API_ROOT = import.meta.env.VITE_API_ROOT ?? "";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const detail = await response.json();
      message = typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? detail);
    } catch {
      // Keep the HTTP status as the useful fallback for non-JSON errors.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function getMeta(mode: AppMode) {
  return request<MetaResponse>(`/api/v1/meta?mode=${mode}`);
}

export function getLiveMarket(symbol: string) {
  return request<import("./types").LiveMarketResponse>(`/api/v1/live-market-data/${symbol}`);
}

export function getRegime(symbol: string, mode: AppMode) {
  return request<RegimeResponse>(`/api/v1/regimes/${symbol}?mode=${mode}`);
}

export function createForecast(body: ForecastRequest) {
  return request<ForecastResponse>("/api/v1/forecasts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getForecastHistory(args: {
  symbol: string;
  model: string;
  horizon_days: number;
  mode: AppMode;
  confidence: number;
  risk_method?: string;
  limit?: number;
}) {
  const query = new URLSearchParams({
    symbol: args.symbol,
    model: args.model,
    horizon_days: String(args.horizon_days),
    mode: args.mode,
    confidence: String(args.confidence),
    risk_method: args.risk_method ?? "filtered_historical_simulation",
    limit: String(args.limit ?? 365),
  });
  return request<ForecastHistory>(`/api/v1/forecasts/history?${query}`);
}

export async function getCsvTable(tableId: string): Promise<CsvTable> {
  const response = await fetch(`${API_ROOT}/api/v1/results/tables/${tableId}`, { headers: { Accept: "text/csv" } });
  if (!response.ok) throw new ApiError(`${response.status} ${response.statusText}`, response.status);
  const text = await response.text();
  return parseCsv(text, response.headers.get("content-disposition")?.match(/filename="?([^";]+)"?/)?.[1] ?? `${tableId}.csv`);
}

export async function getResultFigureUrl(figureId: string) {
  const response = await fetch(`${API_ROOT}/api/v1/results/figures/${figureId}`);
  if (!response.ok) throw new ApiError(`${response.status} ${response.statusText}`, response.status);
  return URL.createObjectURL(await response.blob());
}

export function parseCsv(text: string, name = "table.csv"): CsvTable {
  const records: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted && char === '"' && text[index + 1] === '"') {
      cell += '"'; index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell); cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell); cell = "";
      if (row.some((item) => item.length)) records.push(row);
      row = [];
    } else {
      cell += char;
    }
  }
  if (cell.length || row.length) { row.push(cell); records.push(row); }
  const headers = (records.shift() ?? []).map((header) => header.trim());
  return {
    name,
    headers,
    rows: records.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))),
  };
}

export function riskForConfidence(rows: RiskForecast[] | null | undefined, confidence: number) {
  return rows?.find((row) => Math.abs(row.confidence - confidence) < 0.0001);
}
