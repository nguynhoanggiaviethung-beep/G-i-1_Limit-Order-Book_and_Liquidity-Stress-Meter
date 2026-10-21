import type { CsvTable, ForecastModel, Horizon, MarketSymbol } from "./types";

export interface ModelScore {
  model: ForecastModel;
  qlike: number;
  nObs: number | null;
}

export function scoresForSelection(
  table: CsvTable | undefined,
  symbol: MarketSymbol,
  horizon: Horizon,
): ModelScore[] {
  if (!table) return [];
  return table.rows
    .filter((row) => row.symbol === symbol && Number(row.horizon_days) === horizon)
    .map((row) => ({
      model: row.model as ForecastModel,
      qlike: Number(row.qlike),
      nObs: Number.isFinite(Number(row.n_obs)) ? Number(row.n_obs) : null,
    }))
    .filter((row) => Number.isFinite(row.qlike))
    .sort((a, b) => a.qlike - b.qlike);
}

export function modelFamily(model: string): "simple" | "classical" | "machine-learning" {
  if (["random_walk", "rolling_mean_7", "rolling_mean_30", "ewma_094"].includes(model)) return "simple";
  if (["elasticnet", "random_forest", "lightgbm_l2", "lightgbm_qlike", "xgboost"].includes(model)) return "machine-learning";
  return "classical";
}

export function modelLabel(model: string): string {
  const labels: Record<string, string> = {
    random_walk: "Theo hôm qua",
    rolling_mean_7: "Trung bình 7 ngày",
    rolling_mean_30: "Trung bình 30 ngày",
    ewma_094: "EWMA 0,94",
    har: "HAR",
    har_log: "HAR log",
    har_j: "HAR-J",
    har_cj: "HAR-CJ",
    shar: "SHAR",
    harq: "HARQ",
    garch_t: "GARCH-t",
    gjr_t: "GJR-GARCH-t",
    egarch_t: "EGARCH-t",
    garch_skewt: "GARCH skew-t",
    elasticnet: "Elastic Net",
    random_forest: "Rừng ngẫu nhiên",
    lightgbm_l2: "LightGBM L2",
    lightgbm_qlike: "LightGBM QLIKE",
    xgboost: "XGBoost",
  };
  return labels[model] ?? model.replaceAll("_", " ");
}
