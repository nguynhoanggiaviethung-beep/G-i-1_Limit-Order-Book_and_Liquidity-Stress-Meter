import { createContext, useContext } from "react";
import type { CsvTable, Filters, ForecastModel } from "../api/types";

export interface FilterContextValue {
  filters: Filters;
  update: (patch: Partial<Filters>) => void;
  resolvedModel?: ForecastModel;
  models: ForecastModel[];
  modelTable?: CsvTable;
  recommendedModel?: ForecastModel;
  recommendedQlike?: number;
  selectedQlike?: number;
}

export const FilterContext = createContext<FilterContextValue | null>(null);

export function useFilters() {
  const value = useContext(FilterContext);
  if (!value) throw new Error("Filter controls are only available inside the application router.");
  return value;
}
