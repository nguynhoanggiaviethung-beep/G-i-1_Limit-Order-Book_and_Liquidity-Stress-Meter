import type { components } from "./schema";

export type ForecastRequest = components["schemas"]["ForecastRequest"];
export type MarketSymbol = ForecastRequest["symbol"];
export type AppMode = ForecastRequest["mode"];
export type ForecastModel = ForecastRequest["model"];
export type Horizon = ForecastRequest["horizon_days"];
export type Confidence = NonNullable<ForecastRequest["confidence_levels"]>[number];
export type AudienceMode = "basic" | "expert";
export type ModelSelection = ForecastModel | "auto";

export interface Filters {
  audience: AudienceMode;
  symbol: MarketSymbol;
  horizon: Horizon;
  model: ModelSelection;
  confidence: Confidence;
  mode: AppMode;
}

export interface MetaResponse {
  mode: AppMode;
  as_of_utc: string;
  data_version: string | null;
  config_hash: string | null;
  symbols: MarketSymbol[];
  horizons_days: Horizon[];
  models: string[];
  confidence_levels: Confidence[];
  source: string;
}

export interface RiskForecast {
  confidence: number;
  method: string;
  var_loss_percent: number;
  expected_shortfall_loss_percent: number;
  reliability?: string;
}

export interface ForecastResponse {
  forecast_id: string;
  mode: AppMode;
  symbol: MarketSymbol;
  model: string;
  horizon_days: Horizon;
  origin_time: string;
  target_start: string;
  target_end: string;
  generated_at_utc?: string;
  source_data_as_of_utc?: string;
  source_data_age_seconds?: number;
  data_version: string | null;
  config_hash?: string | null;
  actual_rv_mean: number | null;
  rv_forecast: number;
  sigma_h_approx: number;
  status: string;
  var_es: RiskForecast[] | null;
  note?: string;
  risk_unit?: string;
  rv_percentile?: number | null;
  rv_level?: "low" | "moderate" | "high" | "very_high" | null;
  rv_reference_n?: number;
  rv_reference_as_of_utc?: string | null;
}

export interface ForecastPoint {
  origin_time: string;
  target_end: string;
  actual_rv_mean: number | null;
  rv_forecast: number | null;
  sigma_h_approx: number | null;
  actual_return_percent?: number | null;
  actual_loss_percent?: number | null;
  var_loss_percent?: number | null;
  expected_shortfall_loss_percent?: number | null;
  status: string;
}

export interface ForecastHistory {
  symbol: MarketSymbol;
  model: string;
  horizon_days: Horizon;
  mode: AppMode;
  risk_method: string;
  confidence: number;
  data_version: string | null;
  config_hash: string | null;
  points: ForecastPoint[];
  note?: string;
}

export interface LiveMarketResponse {
  symbol: MarketSymbol;
  connection_status: string;
  last_message_at_utc?: string | null;
  candle?: {
    open_time_utc: string | null;
    close_time_utc: string | null;
    open: string;
    high: string;
    low: string;
    price: string;
    base_volume: string;
    is_closed: boolean;
  } | null;
}

export interface RegimeResponse {
  symbol: MarketSymbol;
  mode: AppMode;
  as_of_utc: string;
  latest_available_date_utc: string;
  regime: string;
  data_version: string | null;
  source: string;
  counts: Record<string, number>;
}

export interface CsvTable {
  name: string;
  headers: string[];
  rows: Record<string, string>[];
}
