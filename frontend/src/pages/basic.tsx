import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { AlertCircle, Database } from "lucide-react";
import { riskForConfidence } from "../api/client";
import { scoresForSelection } from "../api/modelSelection";
import { useForecast, useForecastHistory, useLiveMarket, useMeta, useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { ModelSelector } from "../components/ModelSelector";
import { formatDate, formatNumber, formatPercent, shortHash } from "../components/format";
import { modelLabel } from "../api/modelSelection";
import { ApiState, Provenance } from "../components/PageParts";
import { RiskExceptionPlot } from "../components/charts/RiskExceptionPlot";
import type { MarketSymbol } from "../api/types";

const horizonOptions = [
  { value: 1 as const, label: "Hôm nay" },
  { value: 7 as const, label: "7 ngày tới" },
  { value: 30 as const, label: "30 ngày tới" },
];

const money = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 });
const levelLabels: Record<string, string> = {
  low: "Thấp",
  moderate: "Trung bình",
  high: "Cao",
  very_high: "Rất cao",
};
const riskMethod = "filtered_historical_simulation";

function amountFrom(row: Record<string, string> | undefined, key: string) {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : undefined;
}

export function BasicOverviewPage() {
  const { filters, update, resolvedModel, recommendedQlike, selectedQlike, modelTable } = useFilters();
  const meta = useMeta("production");
  const live = useLiveMarket(filters.symbol);
  const qlikeTable = useTable("T3");
  const forecastQuery = useForecast(filters, resolvedModel, Boolean(resolvedModel));
  const forecast = forecastQuery.data;
  const scoreTable = qlikeTable.data ?? modelTable;
  const dailyModel = filters.model === "auto"
    ? scoresForSelection(scoreTable, filters.symbol, 1)[0]?.model
    : resolvedModel;
  const researchFilters = useMemo(() => ({ ...filters, mode: "research" as const, horizon: 1 as const }), [filters]);
  const history = useForecastHistory(researchFilters, dailyModel, Boolean(dailyModel), 60);
  const riskTable = useTable("T7", Boolean(resolvedModel));
  const [amountText, setAmountText] = useState("10000000");
  const amount = Number(amountText) || 0;
  const confidence: 0.95 | 0.99 = filters.confidence === 0.99 ? 0.99 : 0.95;
  const risk = riskForConfidence(forecast?.var_es, confidence, riskMethod)
    ?? riskForConfidence(forecast?.var_es, confidence);
  const hasFhs = risk?.method === riskMethod;
  const sigmaMoney = forecast && amount > 0 ? amount * forecast.sigma_h_approx / 100 : null;
  const varMoney = risk && hasFhs && amount > 0 ? amount * risk.var_loss_percent / 100 : null;
  const esMoney = risk && hasFhs && amount > 0 ? amount * risk.expected_shortfall_loss_percent / 100 : null;
  const symbolLabel = filters.symbol === "BTCUSDT" ? "Bitcoin" : "Ethereum";
  const confidenceLabel = confidence === 0.99 ? "99%" : "95%";
  const confidenceFrequency = confidence === 0.99 ? "100" : "20";
  const coverageRow = riskTable.data?.rows.find((row) =>
    row.symbol === filters.symbol
    && row.model === resolvedModel
    && Number(row.horizon_days) === filters.horizon
    && row.method === riskMethod
    && Math.abs(Number(row.confidence) - confidence) < 0.0001
    && row.status === "ok",
  );
  const observedRate = amountFrom(coverageRow, "exception_rate");
  const observedCount = amountFrom(coverageRow, "exceptions");
  const observedN = amountFrom(coverageRow, "n_obs");
  const qlikeRows = scoreTable?.rows.filter((row) => row.symbol === filters.symbol && Number(row.horizon_days) === filters.horizon) ?? [];
  const harQlike = amountFrom(qlikeRows.find((row) => row.model === "har"), "qlike");
  const currentQlike = filters.model === "auto" ? recommendedQlike : selectedQlike;
  const comparisonImprovement = harQlike && currentQlike != null ? ((harQlike - currentQlike) / harQlike) * 100 : undefined;
  const maxQlike = Math.max(harQlike ?? 0, currentQlike ?? 0, Number.EPSILON);
  const forecastTime = forecast?.source_data_as_of_utc ?? live.data?.last_message_at_utc;
  const currentModelName = resolvedModel ? modelLabel(resolvedModel) : "đang chọn mô hình";
  const dailyModelName = dailyModel ? modelLabel(dailyModel) : "đang chọn mô hình";

  return <div className="basic-page">
    <section className="basic-intro">
      <div className="basic-updated mono">
        PRODUCTION · BINANCE SPOT 5 PHÚT
        {forecastTime && <span> · DỮ LIỆU ĐẾN {formatDate(forecastTime, true)}</span>}
      </div>
      <h1>{filters.horizon === 1 ? "Hôm nay" : `Trong ${filters.horizon} ngày tới`}, {symbolLabel} có thể dao động mạnh đến đâu?</h1>
      <div className="basic-controls">
        <fieldset className="basic-control-group">
          <legend>Tài sản</legend>
          <div className="segmented-control">
            <button type="button" aria-pressed={filters.symbol === "BTCUSDT"} className={filters.symbol === "BTCUSDT" ? "selected" : ""} onClick={() => update({ symbol: "BTCUSDT" as MarketSymbol })}>Bitcoin</button>
            <button type="button" aria-pressed={filters.symbol === "ETHUSDT"} className={filters.symbol === "ETHUSDT" ? "selected" : ""} onClick={() => update({ symbol: "ETHUSDT" as MarketSymbol })}>Ethereum</button>
          </div>
        </fieldset>
        <fieldset className="basic-control-group">
          <legend>Nhìn trước</legend>
          <div className="segmented-control">
            {horizonOptions.map((option) => <button type="button" key={option.value} aria-pressed={filters.horizon === option.value} className={filters.horizon === option.value ? "selected" : ""} onClick={() => update({ horizon: option.value })}>{option.label}</button>)}
          </div>
        </fieldset>
        <label className="basic-control-group amount-control"><span>Số tiền bạn định đầu tư (đồng)</span><input aria-label="Số tiền bạn định đầu tư bằng đồng" inputMode="numeric" value={money.format(amount)} onChange={(event) => setAmountText(event.target.value.replace(/\D/g, ""))} /></label>
      </div>
      <div className="basic-live-ticker" aria-live="polite">
        <span className={`connection-pill ${live.data?.connection_status === "connected" ? "connected" : ""}`}><span className="status-dot" />{live.data?.connection_status === "connected" ? "Binance đang kết nối" : live.data?.connection_status ?? "Đang kết nối Binance"}</span>
        {live.data?.candle && <strong>{symbolLabel} · ${formatNumber(Number(live.data.candle.price), 2)}</strong>}
        <span>{live.data?.last_message_at_utc ? `Cập nhật lúc ${formatDate(live.data.last_message_at_utc, true)}` : "Giá và forecast tự làm mới mỗi 5 phút"}</span>
      </div>
    </section>

    <ApiState loading={forecastQuery.isLoading || qlikeTable.isLoading} error={forecastQuery.error ?? qlikeTable.error}>
      {forecast && <>
        <section className="basic-level-row" aria-label="Mức biến động so với lịch sử">
          <div className="basic-level-meter">
            <span className="basic-section-label">Mức biến động {filters.horizon === 1 ? "hôm nay" : `trong ${filters.horizon} ngày`} so với lịch sử</span>
            <strong>{forecast.rv_level ? levelLabels[forecast.rv_level] : "Chưa phân loại"}</strong>
            <div className="level-track" aria-hidden="true"><i className={forecast.rv_level ? `level-${forecast.rv_level}` : ""} /></div>
            <div className="level-labels"><span>Thấp</span><span>Trung bình</span><span>Cao</span><span>Rất cao</span></div>
            {forecast.rv_percentile != null && <small>Phân vị {formatPercent(forecast.rv_percentile, 0)} trong {forecast.rv_reference_n?.toLocaleString("vi-VN") ?? "—"} dự báo walk-forward lịch sử.</small>}
          </div>
          <p className="basic-explanation">
            {sigmaMoney !== null
              ? <>{symbolLabel} thường dao động khoảng <b>{formatPercent(forecast.sigma_h_approx, 2)}</b> trong {filters.horizon === 1 ? "một ngày" : filters.horizon + " ngày"}. Với {money.format(amount)} đồng, mức thay đổi tương ứng khoảng <b>{money.format(sigmaMoney)} đồng</b> lên hoặc xuống.</>
              : `${symbolLabel} thường dao động khoảng ${formatPercent(forecast.sigma_h_approx, 2)} trong kỳ dự báo này.`}
          </p>
        </section>

        <section className="basic-metrics" aria-labelledby="basic-metrics-title">
          <div className="basic-metrics-heading"><div><span className="basic-section-label">CÁC CON SỐ HỆ THỐNG ĐÃ TÍNH</span><h2 id="basic-metrics-title">Mỗi con số nghĩa là gì</h2></div>
            <div className="confidence-toggle" role="group" aria-label="Mức tin cậy của VaR và Expected Shortfall">
              <button type="button" aria-pressed={confidence === 0.95} className={confidence === 0.95 ? "selected" : ""} onClick={() => update({ confidence: 0.95 })}>Ngày xấu: 1 trong 20</button>
              <button type="button" aria-pressed={confidence === 0.99} className={confidence === 0.99 ? "selected" : ""} onClick={() => update({ confidence: 0.99 })}>1 trong 100</button>
            </div>
          </div>
          <div className="basic-metric-list">
            <article className="basic-metric-row">
              <div className="basic-metric-value"><span>1 · Dao động thường gặp</span><strong>± {formatPercent(forecast.sigma_h_approx, 2)}</strong><small>{sigmaMoney !== null ? `Khoảng ± ${money.format(sigmaMoney)} đồng trên ${money.format(amount)} đồng` : "Độ lệch chuẩn dự báo"}</small></div>
              <div className="basic-metric-copy"><p><b>Là gì.</b> Mức lên xuống thông thường của giá trong một kỳ như kỳ đang xem; phần lớn các kỳ sẽ nằm trong khoảng này.</p><p><b>Cách đọc.</b> Số càng lớn, giá càng biến động. Đây là khoảng lên hoặc xuống, không phải mức lỗ tối đa.</p><small>Tên chuyên môn: độ lệch chuẩn (sigma) · kỳ {filters.horizon} ngày</small></div>
            </article>
            <article className="basic-metric-row basic-metric-risk">
              <div className="basic-metric-value"><span>2 · Một kỳ xấu</span><strong>{risk ? `lỗ trên ${formatPercent(risk.var_loss_percent, 2)}` : "VaR chưa có"}</strong><small>{varMoney !== null ? `Trên khoảng ${money.format(varMoney)} đồng` : risk ? "Chưa quy đổi tiền: production hiện dùng xấp xỉ chuẩn, chưa dùng FHS" : "loss-positive · % theo kỳ hạn"}</small></div>
              <div className="basic-metric-copy"><p><b>Là gì.</b> Ở mức tin cậy {confidenceLabel}, khoảng 1 trong {confidenceFrequency} kỳ có thể lỗ vượt ngưỡng này.</p><p><b>Cách đọc.</b> Đây không phải mức lỗ tối đa; đôi khi thị trường có thể lỗ nhiều hơn.</p><small>Tên chuyên môn: Value at Risk (VaR) {confidenceLabel} · loss-positive</small></div>
            </article>
            <article className="basic-metric-row basic-metric-risk">
              <div className="basic-metric-value"><span>3 · Nếu kỳ xấu đó xảy ra</span><strong>{risk ? `lỗ trung bình ${formatPercent(risk.expected_shortfall_loss_percent, 2)}` : "ES chưa có"}</strong><small>{esMoney !== null ? `Khoảng ${money.format(esMoney)} đồng trên ${money.format(amount)} đồng` : risk ? "Chưa quy đổi tiền: cần ước lượng FHS để phản ánh rủi ro đuôi" : "Expected Shortfall theo cùng mức tin cậy"}</small></div>
              <div className="basic-metric-copy"><p><b>Là gì.</b> Khi lỗ vượt ngưỡng ở mục 2, đây là mức lỗ trung bình ước tính trong những kỳ xấu đó.</p><p><b>Cách đọc.</b> Có thể dùng con số này để cân nhắc khoản lỗ lớn hơn VaR mà mình có thể chịu được.</p><small>Tên chuyên môn: Expected Shortfall (ES) {confidenceLabel}</small></div>
            </article>
            <article className="basic-metric-row">
              <div className="basic-metric-value"><span>4 · Điểm biến động gốc</span><strong>{formatNumber(forecast.rv_forecast, 2)}</strong><small>Số nền, không dùng để tính tiền</small></div>
              <div className="basic-metric-copy"><p><b>Là gì.</b> Giá trị mô hình tính ra đầu tiên để đo mức biến động dự kiến.</p><p><b>Cách đọc.</b> Đây là số nền; ba con số trên được suy ra từ nó. Tự nó không cho biết bạn có thể lỗ bao nhiêu.</p><small>Tên chuyên môn: RV dự báo = {formatNumber(forecast.rv_forecast, 5)} (% bình phương)</small></div>
            </article>
          </div>
        </section>

        {filters.horizon === 30 && <div className="basic-warning" role="note"><AlertCircle size={16} /><span>Dự báo 30 ngày là ước lượng thô, mang tính mô tả. Các quan sát chồng lấn làm giảm số mẫu độc lập; không hiển thị kết luận kiểm định cho kỳ hạn này.</span></div>}
        {risk && !hasFhs && <div className="basic-warning" role="note"><AlertCircle size={16} /><span>VaR và ES production hiện dùng xấp xỉ phân phối chuẩn. Chưa quy đổi hai số này sang tiền vì giả định chuẩn có thể đánh giá thấp rủi ro đuôi; cần FHS để thể hiện khoản lỗ tiền đáng tin cậy hơn.</span></div>}

        <section className="basic-history-section">
          <div><span className="basic-section-label">NHÌN LẠI QUÁ KHỨ</span><h2>Những ngày vượt mức “ngày xấu” gần đây</h2><p>Mỗi cột là lãi hoặc lỗ một ngày. Đường nét đứt là VaR lịch sử; dấu × là ngày lỗ vượt ngưỡng. Chuỗi OOS dùng kỳ hạn một ngày và mức tin cậy đang chọn.</p></div>
          <ApiState loading={history.isLoading} error={history.error} empty={!history.isLoading && !history.error && !history.data?.points.some((point) => point.actual_return_percent != null && point.var_loss_percent != null)}>
            {history.data && <RiskExceptionPlot points={history.data.points} />}
          </ApiState>
          {history.data && <small className="basic-source-caption">Nguồn: kết quả OOS nghiên cứu · {history.data.points.length.toLocaleString("vi-VN")} quan sát gần nhất · {dailyModelName}{filters.model === "auto" ? " (QLIKE tốt nhất cho một ngày)" : " (đã chọn thủ công)"} · FHS {confidenceLabel}.</small>}
        </section>

        <section className="basic-trust-section">
          <div className="basic-trust-heading"><span className="basic-section-label">ĐỘ TIN CẬY</span><h2>Có thể tin các con số này đến đâu?</h2></div>
          <div className="basic-trust-grid">
            <article className="basic-trust-card">
              <h3>Đã thử trên dữ liệu quá khứ</h3>
              <ApiState loading={riskTable.isLoading} error={riskTable.error} empty={!riskTable.isLoading && !riskTable.error && !coverageRow}>
                {coverageRow && <><strong className="basic-trust-value">{formatPercent((observedRate ?? 0) * 100, 1)}</strong><p>Số kỳ lỗ vượt mức VaR thực tế là {formatPercent((observedRate ?? 0) * 100, 1)}; mức dự kiến là {formatPercent((1 - confidence) * 100, 1)} ({confidenceLabel}).</p><small>{Math.round(observedCount ?? 0).toLocaleString("vi-VN")} lần vượt trong {Math.round(observedN ?? 0).toLocaleString("vi-VN")} quan sát · T7, {symbolLabel}, {filters.horizon} ngày, FHS.</small></>}
              </ApiState>
            </article>
            <article className="basic-trust-card">
              <h3>So với cách làm truyền thống</h3>
              <p className="basic-qlike-caption">QLIKE: thấp hơn là dự báo RV sát thực tế hơn.</p>
              {harQlike != null && currentQlike != null ? <div className="qlike-comparison">
                {[{ label: "Truyền thống (HAR)", value: harQlike, selected: false }, { label: `${currentModelName}${filters.model === "auto" ? " (hệ thống dùng)" : " (đang chọn)"}`, value: currentQlike, selected: true }].map((item) => <div className="qlike-row" key={item.label}>
                  <div><span>{item.label}</span><strong>{formatNumber(item.value, 3)}</strong></div>
                  <i className={item.selected ? "qlike-selected" : ""} style={{ width: `${Math.max(3, item.value / maxQlike * 100)}%` }} />
                </div>)}
                <p>{comparisonImprovement != null && comparisonImprovement >= 0 ? `Mô hình đang dùng có QLIKE thấp hơn khoảng ${formatNumber(comparisonImprovement, 0)}% so với HAR.` : comparisonImprovement != null ? `QLIKE của mô hình đang dùng cao hơn khoảng ${formatNumber(Math.abs(comparisonImprovement), 0)}% so với HAR.` : ""}</p>
                <small>Số thật: T3, {symbolLabel}, {filters.horizon} ngày · {filters.model === "auto" ? "tự chọn theo QLIKE thấp nhất" : "mô hình được chọn thủ công"}.</small>
              </div> : <div className="state-box">Đang đọc bảng T3 cho mã và kỳ hạn này…</div>}
            </article>
          </div>
          {filters.horizon === 30 && <p className="basic-source-caption">Kỳ hạn 30 ngày mang tính mô tả; số quan sát độc lập ít hơn do các cửa sổ dự báo chồng lấn.</p>}
        </section>
      </>}
    </ApiState>

    <aside className="basic-disclaimer"><strong>Lưu ý quan trọng.</strong> Đây là ước tính thống kê từ dữ liệu quá khứ, không phải lời khuyên đầu tư và không đảm bảo kết quả. Khi có tin bất ngờ, thị trường có thể biến động mạnh hơn nhiều. Dự báo 30 ngày chỉ là ước lượng thô. Chỉ đầu tư số tiền bạn chấp nhận mất.</aside>

    <details className="basic-technical">
      <summary>Chi tiết kỹ thuật <span>Đóng sẵn. Chỉ mở khi bạn muốn biết cách tính.</span></summary>
      <p className="technical-caption">Dự báo hiện tại chạy trên dữ liệu Binance production. Kết quả walk-forward đóng băng được gắn nhãn riêng để không nhầm với dự báo hiện tại.</p>
      <div className="technical-grid">
        <div><span>Mô hình production</span><strong>{currentModelName}{filters.model === "auto" ? " (tự động chọn)" : " (chọn thủ công)"}</strong><small>{currentQlike != null ? `QLIKE ${currentQlike.toFixed(4)} · thấp hơn là tốt hơn` : "Chọn theo bảng T3 cùng tài sản và kỳ hạn."}</small></div>
        <div><span>Cách tính “ngày xấu”</span><strong>{risk ? `VaR ${confidenceLabel}: ${formatPercent(risk.var_loss_percent, 3)} · ES: ${formatPercent(risk.expected_shortfall_loss_percent, 3)}` : "Chưa có VaR/ES"}</strong><small>{hasFhs ? "FHS · loss-positive, % theo kỳ hạn" : "Production đang dùng xấp xỉ chuẩn; không quy đổi VaR/ES sang tiền."}</small></div>
        <div><span>Nguồn dữ liệu</span><strong>Binance Spot 5 phút · cập nhật theo chu kỳ 5 phút</strong><small>Forecast dùng dữ liệu hoàn chỉnh mới nhất đến {formatDate(forecastTime, true)}. Mã dữ liệu: {shortHash(forecast?.data_version ?? meta.data?.data_version)} · config: {shortHash(forecast?.config_hash ?? meta.data?.config_hash)}.</small></div>
      </div>
      <ModelSelector compact />
      <section className="frozen-results">
        <div><span className="basic-section-label">KẾT QUẢ NGHIÊN CỨU · SNAPSHOT ĐÓNG BĂNG</span><h3>Đánh giá walk-forward và phương pháp</h3><p>Các bảng và biểu đồ này mô tả kết quả nghiên cứu đã lưu, không phải forecast production hiện tại.</p></div>
        <div className="frozen-result-links">
          <Link to="/evaluation" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}>T3–T6 · So sánh mô hình</Link>
          <Link to="/risk" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}>T7–T8 · Kiểm định VaR/ES</Link>
          <Link to="/method" search={(previous) => ({ ...previous, ...filters, audience: "expert", mode: "research" })}><Database size={14} />T1–T2, T9–T11 · Dữ liệu và phương pháp</Link>
        </div>
      </section>
      <div className="basic-provenance-detail"><Provenance /></div>
    </details>

    <p className="basic-regime-hint"><span className={`status-dot ${live.data?.connection_status === "connected" ? "status-live" : ""}`} /> {live.data?.connection_status === "connected" ? "Binance live đang kết nối" : `Trạng thái Binance: ${live.data?.connection_status ?? "đang kiểm tra"}`} · Forecast tự cập nhật mỗi 5 phút · phiên bản dữ liệu {shortHash(forecast?.data_version)}</p>
  </div>;
}

