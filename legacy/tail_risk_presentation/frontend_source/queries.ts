import { useMutation, useQuery } from "@tanstack/react-query";
import { createForecast, getCsvTable, getForecastHistory, getLiveMarket, getMeta, getRegime } from "./client";
import type { AppMode, Filters, ForecastModel, ForecastRequest } from "./types";

export const keys = {
  meta: (mode: AppMode) => ["meta", mode] as const,
  live: (symbol: string) => ["live-market", symbol] as const,
  regime: (symbol: string, mode: AppMode) => ["regime", symbol, mode] as const,
  history: (filters: Filters) => ["history", filters] as const,
  table: (tableId: string) => ["table", tableId] as const,
};

export function useMeta(mode: AppMode) {
  return useQuery({ queryKey: keys.meta(mode), queryFn: () => getMeta(mode), staleTime: 60_000 });
}
export function useLiveMarket(symbol: string) {
  return useQuery({ queryKey: keys.live(symbol), queryFn: () => getLiveMarket(symbol), refetchInterval: 5_000, staleTime: 2_000 });
}
export function useRegime(symbol: string, mode: AppMode) {
  return useQuery({ queryKey: keys.regime(symbol, mode), queryFn: () => getRegime(symbol, mode), staleTime: mode === "research" ? Infinity : 60_000 });
}
export function useForecast(filters: Filters, model: ForecastModel | undefined, enabled = false) {
  const body: ForecastRequest = {
    symbol: filters.symbol,
    model: model ?? "har",
    horizon_days: filters.horizon,
    mode: filters.mode,
    confidence_levels: [0.95, 0.975, 0.99],
  };
  return useQuery({
    queryKey: ["forecast", filters.symbol, filters.horizon, filters.mode, model],
    queryFn: () => createForecast(body),
    enabled: enabled && Boolean(model),
    retry: false,
    staleTime: 60_000,
    refetchOnWindowFocus: filters.mode === "production",
    refetchInterval: filters.mode === "production" ? 5 * 60_000 : false,
  });
}
export function useForecastMutation() {
  return useMutation({ mutationFn: (body: ForecastRequest) => createForecast(body) });
}
export function useForecastHistory(filters: Filters, model: ForecastModel | undefined, enabled = true, limit = 365) {
  return useQuery({
    queryKey: [...keys.history(filters), model, limit],
    queryFn: () => getForecastHistory({
      symbol: filters.symbol, model: model ?? "har", horizon_days: filters.horizon, mode: filters.mode,
      confidence: filters.confidence,
      limit,
    }),
    enabled,
    retry: false,
    staleTime: filters.mode === "research" ? 60_000 : 15_000,
  });
}
export function useTable(tableId: string, enabled = true) {
  return useQuery({ queryKey: keys.table(tableId), queryFn: () => getCsvTable(tableId), enabled, retry: false, staleTime: Infinity });
}
