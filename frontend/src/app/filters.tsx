import { createContext, useContext } from "react";
import type { Filters } from "../api/types";

export interface FilterContextValue {
  filters: Filters;
  update: (patch: Partial<Filters>) => void;
}

export const FilterContext = createContext<FilterContextValue | null>(null);

export function useFilters() {
  const value = useContext(FilterContext);
  if (!value) throw new Error("Filter controls are only available inside the application router.");
  return value;
}
