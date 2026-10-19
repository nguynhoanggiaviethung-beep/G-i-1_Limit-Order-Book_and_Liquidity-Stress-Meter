import { createRootRoute, createRoute, createRouter, lazyRouteComponent, Link, Outlet, RouterProvider, useRouter, useRouterState } from "@tanstack/react-router";
import { Activity, ArrowUpRight, BarChart3, BookOpenText, CircleGauge, LayoutDashboard, ShieldAlert } from "lucide-react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMemo } from "react";
import { useMeta } from "../api/queries";
import type { AppMode, Confidence, Filters, ForecastModel, Horizon, MarketSymbol } from "../api/types";
import { FilterContext } from "./filters";
import { Provenance } from "../components/PageParts";
import { SelectField } from "../components/SelectField";

const fallback: Filters = { symbol: "BTCUSDT", horizon: 1, model: "har", confidence: 0.95, mode: "research" };
function validateFilters(search: Record<string, unknown>): Filters {
  return {
    symbol: search.symbol === "ETHUSDT" ? "ETHUSDT" : "BTCUSDT",
    horizon: [1, 7, 30].includes(Number(search.horizon)) ? Number(search.horizon) as Horizon : fallback.horizon,
    model: typeof search.model === "string" ? search.model as ForecastModel : fallback.model,
    confidence: [0.95, 0.975, 0.99].includes(Number(search.confidence)) ? Number(search.confidence) as Confidence : fallback.confidence,
    mode: search.mode === "production" ? "production" : "research",
  };
}

function RootLayout() {
  const filters = rootRoute.useSearch();
  const router = useRouter();
  const path = useRouterState({ select: (state) => state.location.pathname });
  const meta = useMeta(filters.mode);
  const update = (patch: Partial<Filters>) => {
    void router.navigate({
      to: path as never,
      search: (previous: Record<string, unknown>) => ({ ...previous, ...patch }),
      replace: true,
    } as never);
  };
  const filterValue = useMemo(() => ({ filters, update }), [filters]);
  const navItems = [
    { to: "/", label: "Tổng quan", icon: LayoutDashboard },
    { to: "/forecast", label: "Dự báo", icon: Activity },
    { to: "/evaluation", label: "Đánh giá mô hình", icon: BarChart3 },
    { to: "/risk", label: "Rủi ro & backtest", icon: ShieldAlert },
    { to: "/regimes", label: "Regime & chiến lược", icon: CircleGauge },
    { to: "/method", label: "Phương pháp & dữ liệu", icon: BookOpenText },
  ] as const;
  const title = navItems.find((item) => item.to === path)?.label ?? "Tail Risk Lab";
  const modelOptions = (meta.data?.models?.length ? meta.data.models : ["random_walk", "har", "har_log", "har_j", "har_cj", "shar", "harq", "elasticnet", "random_forest", "lightgbm_l2", "lightgbm_qlike", "xgboost", "garch_t", "gjr_t", "egarch_t", "garch_skewt"]).map((model) => ({ value: model, label: model.replaceAll("_", " ") }));
  return <FilterContext.Provider value={filterValue}>
    <div className="app-frame">
      <aside className="sidebar">
        <Link to="/" search={(previous) => ({ ...previous, ...filters })} className="brand"><div className="brand-mark"><Activity size={19} /></div><div><strong>TAIL RISK</strong><small>CRYPTO RESEARCH LAB</small></div></Link>
        <div className="sidebar-label">KHÔNG GIAN LÀM VIỆC</div>
        <nav aria-label="Điều hướng chính" className="side-nav">
          {navItems.map((item) => <Link key={item.to} to={item.to} search={(previous) => ({ ...previous, ...filters })} activeOptions={{ exact: item.to === "/" }} className="nav-link" activeProps={{ className: "nav-link nav-active" }}>
            <item.icon size={17} strokeWidth={1.8} /><span>{item.label}</span>{item.to === "/forecast" && <ArrowUpRight className="nav-arrow" size={14} />}
          </Link>)}
        </nav>
        <div className="sidebar-bottom"><div className="sidebar-status"><span className="status-dot" /><div><strong>API local</strong><small>127.0.0.1:8000</small></div></div><div className="sidebar-version">FRONTEND · M0—M6</div></div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <div className="breadcrumb"><span>CRYPTO VOLATILITY</span><b aria-hidden="true">/</b><strong>{title.toUpperCase()}</strong></div>
          <div className="top-controls">
            <div className="mode-switch" role="group" aria-label="Chế độ dữ liệu">
              <button className={filters.mode === "research" ? "selected" : ""} onClick={() => update({ mode: "research" as AppMode })}>Nghiên cứu</button>
              <button className={filters.mode === "production" ? "selected" : ""} onClick={() => update({ mode: "production" as AppMode })}>Production</button>
            </div>
            <SelectField label="Tài sản" value={filters.symbol} options={[{ value: "BTCUSDT", label: "BTC / USDT" }, { value: "ETHUSDT", label: "ETH / USDT" }]} onChange={(value) => update({ symbol: value as MarketSymbol })} />
            <SelectField label="Kỳ hạn" value={String(filters.horizon)} options={[{ value: "1", label: "1 ngày" }, { value: "7", label: "7 ngày" }, { value: "30", label: "30 ngày" }]} onChange={(value) => update({ horizon: Number(value) as Horizon })} />
            <SelectField label="Mô hình" value={filters.model} options={modelOptions} onChange={(value) => update({ model: value as ForecastModel })} />
            <SelectField label="Mức tin cậy" value={String(filters.confidence)} options={[{ value: "0.95", label: "95%" }, { value: "0.975", label: "97,5%" }, { value: "0.99", label: "99%" }]} onChange={(value) => update({ confidence: Number(value) as Confidence })} />
          </div>
        </header>
        <main className="content-area"><Outlet /></main>
        <footer className="app-footer"><span>Tail Risk Lab <b aria-hidden="true">·</b> Binance Spot 5m</span><Provenance compact /></footer>
      </div>
      <nav className="mobile-nav" aria-label="Điều hướng di động">
        {navItems.filter((item) => ["/", "/forecast", "/evaluation", "/risk", "/regimes"].includes(item.to)).map((item) => <Link key={item.to} to={item.to} search={(previous) => ({ ...previous, ...filters })} activeOptions={{ exact: item.to === "/" }} activeProps={{ className: "mobile-active" }}><item.icon size={18} /><span>{item.to === "/evaluation" ? "Mô hình" : item.to === "/regimes" ? "Regime" : item.label.split(" ")[0]}</span></Link>)}
      </nav>
    </div>
  </FilterContext.Provider>;
}

const rootRoute = createRootRoute({ validateSearch: validateFilters, component: RootLayout });
const overviewRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: lazyRouteComponent(() => import("../pages/overview"), "OverviewPage") });
const forecastRoute = createRoute({ getParentRoute: () => rootRoute, path: "/forecast", component: lazyRouteComponent(() => import("../pages/forecast"), "ForecastPage") });
const evaluationRoute = createRoute({ getParentRoute: () => rootRoute, path: "/evaluation", component: lazyRouteComponent(() => import("../pages/evaluation"), "EvaluationPage") });
const riskRoute = createRoute({ getParentRoute: () => rootRoute, path: "/risk", component: lazyRouteComponent(() => import("../pages/risk"), "RiskPage") });
const regimesRoute = createRoute({ getParentRoute: () => rootRoute, path: "/regimes", component: lazyRouteComponent(() => import("../pages/regimes"), "RegimesPage") });
const methodRoute = createRoute({ getParentRoute: () => rootRoute, path: "/method", component: lazyRouteComponent(() => import("../pages/method"), "MethodPage") });
const routeTree = rootRoute.addChildren([overviewRoute, forecastRoute, evaluationRoute, riskRoute, regimesRoute, methodRoute]);
export const router = createRouter({ routeTree, defaultPreload: "intent" });
declare module "@tanstack/react-router" { interface Register { router: typeof router; } }

const queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } });
export function AppRouter() {
  return <QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider>;
}
