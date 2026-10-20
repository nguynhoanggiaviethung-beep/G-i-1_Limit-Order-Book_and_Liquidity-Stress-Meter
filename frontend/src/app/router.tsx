import { createRootRoute, createRoute, createRouter, lazyRouteComponent, Link, Outlet, RouterProvider, useRouter, useRouterState } from "@tanstack/react-router";
import { Activity, BarChart3, BookOpenText, CircleGauge, LayoutDashboard, ShieldAlert } from "lucide-react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo } from "react";
import { useMeta, useTable } from "../api/queries";
import { scoresForSelection } from "../api/modelSelection";
import type { AudienceMode, Confidence, Filters, ForecastModel, Horizon, MarketSymbol, ModelSelection } from "../api/types";
import { FilterContext } from "./filters";
import { Provenance } from "../components/PageParts";
import { SelectField } from "../components/SelectField";

const modelNames: ForecastModel[] = ["random_walk", "rolling_mean_7", "rolling_mean_30", "ewma_094", "har", "har_log", "har_j", "har_cj", "shar", "harq", "elasticnet", "random_forest", "lightgbm_l2", "lightgbm_qlike", "xgboost", "garch_t", "gjr_t", "egarch_t", "garch_skewt"];
const fallback: Filters = { audience: "basic", symbol: "BTCUSDT", horizon: 1, model: "auto", confidence: 0.95, mode: "production" };

function validateFilters(search: Record<string, unknown>): Filters {
  const model = search.model === "auto" || modelNames.includes(search.model as ForecastModel)
    ? search.model as ModelSelection
    : fallback.model;
  return {
    audience: search.audience === "expert" ? "expert" : "basic",
    symbol: search.symbol === "ETHUSDT" ? "ETHUSDT" : "BTCUSDT",
    horizon: [1, 7, 30].includes(Number(search.horizon)) ? Number(search.horizon) as Horizon : fallback.horizon,
    model,
    confidence: [0.95, 0.975, 0.99].includes(Number(search.confidence)) ? Number(search.confidence) as Confidence : fallback.confidence,
    mode: search.mode === "research" && search.audience === "expert" ? "research" : "production",
  };
}

function DashboardModeSwitch({ audience, change }: { audience: AudienceMode; change: (value: AudienceMode) => void }) {
  return <div className="audience-switch" role="group" aria-label="Chế độ dashboard">
    <button type="button" className={audience === "basic" ? "selected" : ""} aria-pressed={audience === "basic"} onClick={() => change("basic")}>Dự báo</button>
    <button type="button" className={audience === "expert" ? "selected" : ""} aria-pressed={audience === "expert"} onClick={() => change("expert")}>Kết quả nghiên cứu</button>
  </div>;
}

function RootLayout() {
  const filters = rootRoute.useSearch();
  const router = useRouter();
  const path = useRouterState({ select: (state) => state.location.pathname });
  const meta = useMeta(filters.mode);
  const modelTable = useTable("T3", filters.model === "auto" || filters.audience === "basic" || path === "/forecast");
  const scores = useMemo(() => scoresForSelection(modelTable.data, filters.symbol, filters.horizon), [modelTable.data, filters.symbol, filters.horizon]);
  const recommendedModel = scores[0]?.model;
  const recommendedQlike = scores[0]?.qlike;
  const resolvedModel = filters.model === "auto" ? recommendedModel : filters.model;
  const selectedQlike = scores.find((score) => score.model === resolvedModel)?.qlike;
  const models = (meta.data?.models?.length ? meta.data.models : modelNames) as ForecastModel[];

  const update = useCallback((patch: Partial<Filters>) => {
    void router.navigate({
      to: path as never,
      search: (previous: Record<string, unknown>) => ({ ...previous, ...patch }),
      replace: true,
    } as never);
  }, [router, path]);
  const changeAudience = useCallback((audience: AudienceMode) => {
    const destination = audience === "basic" ? "/" : path;
    void router.navigate({
      to: destination as never,
      search: (previous: Record<string, unknown>) => ({ ...previous, audience, mode: audience === "basic" ? "production" : "research" }),
    } as never);
  }, [router, path]);
  useEffect(() => {
    if (filters.audience === "basic" && (path !== "/" || filters.mode !== "production" || ![0.95, 0.99].includes(filters.confidence))) {
      void router.navigate({ to: "/", search: (previous: Record<string, unknown>) => ({ ...previous, audience: "basic", mode: "production", confidence: previous.confidence === 0.99 ? 0.99 : 0.95 }) } as never);
    } else if (filters.audience === "expert" && filters.mode !== "research") {
      void router.navigate({ to: path as never, search: (previous: Record<string, unknown>) => ({ ...previous, audience: "expert", mode: "research" }) } as never);
    }
  }, [filters.audience, filters.mode, path, router]);
  const filterValue = useMemo(() => ({ filters, update, resolvedModel, recommendedModel, recommendedQlike, selectedQlike, models, modelTable: modelTable.data }), [filters, update, resolvedModel, recommendedModel, recommendedQlike, selectedQlike, models, modelTable.data]);
  const navItems = [
    { to: "/", label: "Tổng quan", icon: LayoutDashboard },
    { to: "/forecast", label: "Dự báo", icon: Activity },
    { to: "/evaluation", label: "Đánh giá mô hình", icon: BarChart3 },
    { to: "/risk", label: "Rủi ro và backtest", icon: ShieldAlert },
    { to: "/regimes", label: "Trạng thái và chiến lược", icon: CircleGauge },
    { to: "/method", label: "Phương pháp và dữ liệu", icon: BookOpenText },
  ] as const;
  const title = navItems.find((item) => item.to === path)?.label ?? "Tail-Risk";

  return <FilterContext.Provider value={filterValue}>
    {filters.audience === "basic" ? <div className="basic-app">
      <header className="basic-topbar">
        <Link to="/" search={(previous) => ({ ...previous, ...filters, audience: "basic", mode: "production" })} className="basic-brand"><strong>TAIL-RISK</strong><span>Biến động và rủi ro đuôi</span></Link>
        <div className="basic-header-controls"><DashboardModeSwitch audience={filters.audience} change={changeAudience} /><span className="basic-production-badge"><span className="status-dot" />GIÁ BINANCE · NẾN 5 PHÚT</span></div>
      </header>
      <main className="basic-content"><Outlet /></main>
      <footer className="basic-footer"><Provenance compact /></footer>
    </div> : <div className="expert-app">
      <aside className="sidebar">
        <Link to="/" search={(previous) => ({ ...previous, ...filters, audience: "basic", mode: "production" })} className="brand"><div className="brand-mark"><Activity size={19} /></div><div><strong>TAIL-RISK</strong><small>Biến động và rủi ro đuôi</small></div></Link>
        <div className="sidebar-label">KẾT QUẢ NGHIÊN CỨU</div>
        <nav aria-label="Điều hướng chính" className="side-nav">
          {navItems.map((item) => <Link key={item.to} to={item.to} search={(previous) => ({ ...previous, ...filters })} activeOptions={{ exact: item.to === "/" }} className="nav-link" activeProps={{ className: "nav-link nav-active" }}>
            <item.icon size={17} strokeWidth={1.8} /><span>{item.label}</span>
          </Link>)}
        </nav>
        <div className="sidebar-bottom"><div className="sidebar-status"><span className="status-dot" /><div><strong>API local</strong><small>127.0.0.1:8000</small></div></div><Provenance compact /></div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <div className="breadcrumb"><span>TAIL-RISK</span><b aria-hidden="true">/</b><strong>{title.toUpperCase()}</strong></div>
          <div className="top-controls">
            <DashboardModeSwitch audience={filters.audience} change={changeAudience} />
            <SelectField label="Tài sản" value={filters.symbol} options={[{ value: "BTCUSDT", label: "Bitcoin" }, { value: "ETHUSDT", label: "Ethereum" }]} onChange={(value) => update({ symbol: value as MarketSymbol })} />
            <SelectField label="Kỳ hạn" value={String(filters.horizon)} options={[{ value: "1", label: "1 ngày" }, { value: "7", label: "7 ngày" }, { value: "30", label: "30 ngày" }]} onChange={(value) => update({ horizon: Number(value) as Horizon })} />
            <SelectField label="Mức tin cậy" value={String(filters.confidence)} options={[{ value: "0.95", label: "95%" }, { value: "0.975", label: "97,5%" }, { value: "0.99", label: "99%" }]} onChange={(value) => update({ confidence: Number(value) as Confidence })} />
          </div>
        </header>
        <main className="content-area"><Outlet /></main>
        <footer className="app-footer"><span>Binance Spot 5 phút <b aria-hidden="true">·</b> số lỗ hiển thị theo chiều tổn thất</span><Provenance compact /></footer>
      </div>
      <nav className="mobile-nav" aria-label="Điều hướng di động">
        {navItems.filter((item) => ["/", "/forecast", "/evaluation", "/risk", "/regimes"].includes(item.to)).map((item) => <Link key={item.to} to={item.to} search={(previous) => ({ ...previous, ...filters })} activeOptions={{ exact: item.to === "/" }} activeProps={{ className: "mobile-active" }}><item.icon size={18} /><span>{item.to === "/evaluation" ? "Mô hình" : item.to === "/regimes" ? "Regime" : item.label.split(" ")[0]}</span></Link>)}
      </nav>
    </div>}
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
