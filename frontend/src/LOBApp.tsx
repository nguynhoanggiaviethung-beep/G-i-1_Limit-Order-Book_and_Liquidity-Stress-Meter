import { useCallback, useEffect, useRef, useState } from "react";

type SymbolCode = "BTCUSDT" | "ETHUSDT";
type Mode = "live" | "replay";
type AppView = "dashboard" | "research";
type Level = [string, string];
type Snapshot = {
  symbol: SymbolCode;
  timestamp_utc: string;
  bids: Level[];
  asks: Level[];
  features: Record<string, unknown>;
  quality: { book_valid: boolean; seconds_since_gap?: number | null; sequence_gap?: boolean };
};
type DataStatus = {
  status: string;
  source: string;
  symbols: Record<string, { status: string; partition_count: number; row_count: number; first_date_utc: string | null; last_date_utc: string | null; reason: string | null }>;
};
type Meter = {
  status: string;
  reason?: string | null;
  timestamp_utc?: string | null;
  horizon_seconds: number;
  spread?: string | null;
  depth_imbalance?: Record<string, number | null>;
  ofi_5s?: number | null;
  p_up?: number | null;
  p_flat?: number | null;
  p_down?: number | null;
  p_stress?: number | null;
  stress_probability_threshold?: number | null;
  model_version?: string | null;
  quality?: { book_valid?: boolean; reason?: string; seconds_since_gap?: number | null };
};
type LiveUpdate = {
  schema: string;
  symbol: SymbolCode;
  status: string;
  reason?: string | null;
  updated_at_utc: string;
  snapshot: Snapshot | null;
  meter: Meter;
  horizon_seconds: number;
};

const API_ROOT = import.meta.env.VITE_API_ROOT ?? "";
async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : body.detail?.message;
    throw new Error(detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function num(value: unknown, digits = 4, signed = false) {
  const parsed = Number(value);
  if (value == null || !Number.isFinite(parsed)) return "—";
  const text = new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(parsed);
  return signed && parsed > 0 ? `+${text}` : text;
}

function pct(value: unknown) {
  const parsed = Number(value);
  return value != null && Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "—";
}

function formatTime(value?: string | null) {
  if (!value) return "chưa có dữ liệu";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "—"
    : new Intl.DateTimeFormat("vi-VN", { dateStyle: "medium", timeStyle: "medium", timeZone: "Asia/Ho_Chi_Minh" }).format(date);
}

function toLocalInput(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function noReplayMessage(symbol: SymbolCode, timestamp: string | undefined, status: DataStatus | null, reason?: string) {
  const symbolData = status?.symbols?.[symbol];
  const first = symbolData?.first_date_utc;
  const last = symbolData?.last_date_utc;
  const range = first && last
    ? `Dữ liệu ${symbol} hiện có từ ${first} đến ${last} (UTC).`
    : `Hiện chưa có dữ liệu replay cho ${symbol}.`;
  const selected = timestamp ? ` tại ${formatTime(new Date(timestamp).toISOString())}` : "";
  const day = timestamp ? new Date(timestamp).toISOString().slice(0, 10) : null;
  const outsideRange = Boolean(day && first && last && (day < first || day > last));
  const staleBook = reason?.match(/Latest valid book is ([\d.]+)s old \(limit ([\d.]+)s\)/);
  const explanation = outsideRange
    ? "Chọn thời điểm trong khoảng dữ liệu; capture live không thể khôi phục sổ lệnh lịch sử của ngày trước đó."
    : staleBook
      ? `Snapshot gần nhất cách mốc yêu cầu ${staleBook[1]} giây, vượt ngưỡng cho phép ${staleBook[2]} giây.`
      : "Ngày này có dữ liệu, nhưng không tìm thấy book hợp lệ gần mốc đã chọn; hãy chọn giờ khác trong khoảng quan sát.";
  return `Không có replay book cho ${symbol}${selected}. ${range} ${explanation}`;
}

function liveStatusLabel(status: string) {
  switch (status) {
    case "live": return "đang nhận dữ liệu";
    case "syncing": return "đang đồng bộ sổ lệnh";
    case "reconnecting": return "đang kết nối lại";
    case "stale": return "dữ liệu bị gián đoạn";
    case "disabled": return "đã tắt nguồn live";
    case "error": return "lỗi nguồn dữ liệu";
    case "stopped": return "đã dừng";
    default: return "đang kết nối Binance";
  }
}

function Stat({ label, value, unit, source }: { label: string; value: string; unit?: string; source: string }) {
  return (
    <article className="stat">
      <span className="lbl">{label}</span>
      <strong className="big">{value}{unit && <small>{unit}</small>}</strong>
      <span className="src">nguồn · {source}</span>
    </article>
  );
}

function Segmented<T extends string | number>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; text: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="field">
      <span className="lbl">{label}</span>
      <div className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.text}</button>
        ))}
      </div>
    </div>
  );
}

function BookSide({ side, levels, scale }: { side: "bid" | "ask"; levels: Level[]; scale: number }) {
  let running = 0;
  return (
    <section className="book">
      <h3>{side === "bid" ? "bid · mua" : "ask · bán"}</h3>
      <div className="row head"><span>#</span><span>giá (usdt)</span><span>qty</span><span>cộng dồn</span></div>
      {levels.slice(0, 10).map(([price, qty], i) => {
        running += Number(qty) || 0;
        return (
          <div className="row" key={`${side}-${price}`}>
            <i className={`bar ${side}`} style={{ width: `${scale ? Math.min(100, (running / scale) * 100) : 0}%` }} />
            <span className="mu">{String(i + 1).padStart(2, "0")}</span>
            <span>{num(price, 2)}</span>
            <span>{num(qty, 5)}</span>
            <span>{num(running, 5)}</span>
          </div>
        );
      })}
      {!levels.length && <p className="empty">Chưa có snapshot hợp lệ.</p>}
    </section>
  );
}

function ProbRow({ mark, label, value, tone }: { mark: string; label: string; value?: number | null; tone: "data" | "mute" | "ink" }) {
  const v = value != null && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
  return (
    <div className="prob">
      <div><span>{mark} {label}</span><strong>{v === null ? "—" : pct(v)}</strong></div>
      <div className="track"><i className={tone} style={{ width: `${(v ?? 0) * 100}%` }} /></div>
    </div>
  );
}

const researchFigures = [
  {
    title: "Heatmap sổ lệnh · top 10",
    file: "lob_heatmap_validation.png",
    scope: "VALIDATION",
    reading: "Trục ngang là thời gian; trục dọc là giá cách mid-price bao nhiêu tick. Màu xanh càng đậm nghĩa là khối lượng trong sổ lệnh tại mức giá đó càng lớn. Mỗi khung là 300 giây liên tục, bắt đầu lúc 00:00 UTC ngày 05/10/2026.",
    numbers: "Trong khung này, spread trung vị của cả BTC và ETH là 1 tick. Độ sâu L10 trung vị: BTC có 3.801 BTC ở bid và 3.919 BTC ở ask; ETH có 33.721 ETH ở bid và 34.051 ETH ở ask. OBI L10 trung vị lần lượt là +0.002 và +0.015.",
    finding: "Thanh khoản tập trung sát mid-price và hai phía khá cân bằng trong khung được chọn. Đây là ảnh chụp cấu trúc book ở một thời điểm; các vệt sáng thay đổi theo giây và không tự chứng minh rằng độ sâu dự báo được giá.",
  },
  {
    title: "OBI L10 và lợi suất mid-price tương lai",
    file: "obi_vs_forward_return_validation.png",
    scope: "VALIDATION",
    reading: "Chia OBI L10 thành 10 nhóm từ thấp đến cao. Trục dọc là lợi suất mid-price sau 5 giây, tính theo basis point (bps); đường xanh là trung bình, đường đỏ là trung vị.",
    numbers: "Có 604,445 quan sát BTC và 604,442 quan sát ETH. Trung bình tăng từ −0.528 lên +0.492 bps giữa decile 1 và 10 với BTC; ETH tăng từ −0.642 lên +0.609 bps. Tương quan hạng Spearman giữa OBI và lợi suất là +0.419 (BTC), +0.369 (ETH). Trung vị bằng 0 ở phần lớn decile.",
    finding: "Trong tuần validation, book nghiêng về bid thường đi cùng lợi suất 5 giây cao hơn theo trung bình. Nhiều trung vị vẫn bằng 0 cho thấy chuyển động nhỏ hoặc không đổi chiếm ưu thế; chuỗi theo giây tự tương quan mạnh nên đây là quan hệ mô tả, không phải kiểm định nhân quả hay bằng chứng độc lập từng giây.",
  },
  {
    title: "Ma trận nhầm lẫn · mô hình cổ điển",
    file: "confusion_matrices_holdout.png",
    scope: "HOLDOUT ĐÃ CHẤM",
    reading: "Hàng là lớp thật, cột là lớp dự báo; ô trên đường chéo là dự báo đúng. Ô màu ghi số mẫu và tỷ lệ trong hàng đó. Hình dùng mô hình được chọn trước cho từng tác vụ.",
    numbers: "Hướng BTC (Random Forest): nhận đúng 38.6% lớp down, 86.1% flat, 37.0% up. Hướng ETH (Logistic): 40.9%, 70.0%, 49.8%. Stress XGBoost: BTC bắt được 43.1% sự kiện stress (3,574/8,287), precision 87.2%; ETH bắt 40.9% (7,844/19,193), precision 90.7%.",
    finding: "Mô hình hướng thường nhận ra lớp flat tốt hơn hai hướng biến động. Với stress, precision cao nhưng recall chỉ khoảng 41–43%: các cảnh báo thường đúng, song mô hình bỏ lỡ hơn một nửa sự kiện. Đây là tổng hợp từ holdout đã chấm một lần.",
  },
  {
    title: "Calibration · mô hình cổ điển",
    file: "calibration_curves_holdout.png",
    scope: "HOLDOUT ĐÃ CHẤM",
    reading: "Trục ngang là xác suất trung bình mô hình đưa ra trong mỗi nhóm; trục dọc là tần suất thực tế. Đường càng gần đường chéo nét đứt thì xác suất càng khớp tần suất. Brier thấp hơn biểu thị sai số xác suất thấp hơn.",
    numbers: "Brier stress là 0.026 cho BTC và 0.058 cho ETH; hướng lần lượt là 0.486 và 0.569. Ở bin stress cao nhất, BTC dự báo 97.6% và ghi nhận 97.3%; ETH dự báo 98.3% và ghi nhận 98.4%. Riêng lớp up ETH, bin cao nhất dự báo 93.1% nhưng tần suất thực tế là 53.1% (326 mẫu).",
    finding: "Xác suất stress cao nhất khá sát tần suất quan sát trong các bin đã lưu. Calibration của hướng kém đồng đều hơn; điểm lớp up ETH lệch đáng kể khỏi đường chuẩn. Brier giữa stress và hướng không nên so trực tiếp vì số lớp và phân bố nhãn khác nhau.",
  },
  {
    title: "Hiệu năng theo giai đoạn pilot",
    file: "performance_over_time_pilot.png",
    scope: "TÓM TẮT ĐÃ LƯU",
    reading: "Mỗi ô so sánh cùng một chỉ số giữa tuần validation 05–11/10 và holdout 12–18/10: hướng dùng Macro-F1, stress dùng PR-AUC. Điểm holdout lấy từ kết quả tổng hợp đã lưu.",
    numbers: "BTC: hướng 0.595 → 0.554 (−0.041), stress 0.517 → 0.612 (+0.095). ETH: hướng 0.562 → 0.535 (−0.027), stress 0.593 → 0.624 (+0.031).",
    finding: "Điểm hướng giảm ở tuần holdout, còn PR-AUC stress tăng trong pilot này. Chỉ có hai tuần được so sánh nên không thể suy ra xu hướng ổn định theo thời gian; biểu đồ cũng không phải chuỗi hiệu năng liên tục.",
  },
  {
    title: "DeepLOB so với Logistic và XGBoost",
    file: "deeplob_vs_baselines_validation.png",
    scope: "VALIDATION · DEEPLOB",
    reading: "Macro-F1 đo chất lượng dự báo ba lớp hướng; PR-AUC đánh giá xếp hạng xác suất stress khi lớp stress ít gặp. Cột cao hơn tốt hơn. Cả ba mô hình dùng cùng các điểm validation.",
    numbers: "Macro-F1 hướng BTC: DeepLOB 0.568, Logistic 0.587, XGBoost 0.582; ETH: 0.507, 0.557, 0.526. PR-AUC stress BTC: 0.230, 0.371, 0.435; ETH: 0.308, 0.452, 0.534. Mỗi tác vụ dùng tối đa 8,000 mẫu train và khoảng 2,500 điểm validation.",
    finding: "DeepLOB không vượt baseline nào trong bốn so sánh này; khoảng cách lớn nhất nằm ở stress. Đây là kết quả validation một tuần với giới hạn tính toán, chưa phải kết luận final-test.",
  },
  {
    title: "Ma trận nhầm lẫn · DeepLOB",
    file: "deeplob_confusion_validation.png",
    scope: "VALIDATION · DEEPLOB",
    reading: "Đọc theo hàng thật, cột dự báo; phần trăm là tỷ lệ nhận diện đúng theo từng lớp. Với stress, lớp 1 là stress và lớp 0 là không stress.",
    numbers: "Hướng BTC: recall down/flat/up là 53.8%/70.4%/49.9%; ETH là 53.4%/67.3%/32.2%. Stress BTC có 60 ca thật: bắt 35 (58.3%), nhưng tạo 333 cảnh báo sai; precision 9.5%. ETH có 120 ca thật: bắt 65 (54.2%), 335 cảnh báo sai; precision 16.3%.",
    finding: "DeepLOB nhận ra lớp flat tốt hơn hướng tăng, đặc biệt trên ETH. Với stress, recall khoảng 54–58% nhưng phần lớn cảnh báo là sai do nhãn dương hiếm; cần xem precision cùng recall, không chỉ nhìn tỷ lệ bắt được.",
  },
  {
    title: "Calibration · DeepLOB",
    file: "deeplob_calibration_validation.png",
    scope: "VALIDATION · DEEPLOB",
    reading: "Đường chéo là hiệu chỉnh hoàn hảo. Nếu một điểm nằm dưới đường này, mô hình dự báo xác suất cao hơn tần suất xảy ra; điểm nằm trên đường là dự báo thấp hơn tần suất.",
    numbers: "Stress BTC: Brier 0.0998, ECE 0.1964; ETH: Brier 0.1141, ECE 0.2046. Ở bin stress cao nhất, xác suất trung bình 92.9% nhưng tần suất thực tế chỉ 39.1% (BTC, 23 mẫu) và 50.0% (ETH, 46 mẫu).",
    finding: "DeepLOB có xu hướng quá tự tin với xác suất stress cao trong validation, nhất là ở các bin đuôi có ít mẫu. Không nên dùng xác suất này như mức rủi ro đã hiệu chuẩn; kết quả cũng chưa được xác nhận trên holdout.",
  },
] as const;

const researchAnswers = [
  {
    id: "RQ1",
    status: "KẾT QUẢ MỘT PHẦN",
    tone: "mixed",
    question: "OBI tĩnh có dự báo hướng giá tốt hơn baseline không?",
    answer: "OBI có thêm tín hiệu so với luôn đoán lớp phổ biến nhất, nhưng chưa vượt baseline dựa trên hướng giá trước đó.",
    evidence: "Validation 5 giây: Macro-F1 OBI-only là 0.410 (BTC) và 0.391 (ETH); class-frequency là 0.252 và 0.239; persistence cao hơn, lần lượt 0.451 và 0.412. Vì vậy, kết quả ủng hộ tín hiệu OBI nhưng chưa chứng minh OBI đủ mạnh để thắng baseline persistence.",
  },
  {
    id: "RQ2",
    status: "CẢI THIỆN NHẸ",
    tone: "partial",
    question: "OFI theo sự kiện có bổ sung thông tin ngoài OBI tĩnh không?",
    answer: "Có dấu hiệu cải thiện, nhưng mức tăng nhỏ và chưa có kiểm định riêng cho phép kết luận đó là khác biệt ổn định.",
    evidence: "Trên validation 5 giây, thêm OFI vào Logistic OBI-only tăng Macro-F1 từ 0.410 lên 0.414 ở BTC, và 0.391 lên 0.394 ở ETH. MCC tăng tương ứng 0.238→0.240 và 0.194→0.196.",
  },
  {
    id: "RQ3",
    status: "DEEPLOB CHƯA VƯỢT BASELINE",
    tone: "negative",
    question: "DeepLOB có cải thiện dự báo so với các mô hình cổ điển không?",
    answer: "Trong phép so sánh validation công bằng ở horizon 5 giây, DeepLOB không vượt Logistic hoặc XGBoost ở hướng giá lẫn stress.",
    evidence: "Macro-F1 hướng BTC: DeepLOB 0.568, Logistic 0.587, XGBoost 0.582; ETH: 0.507, 0.557, 0.526. PR-AUC stress BTC: 0.230 so với 0.371/0.435; ETH: 0.308 so với 0.452/0.534. Mô hình cổ điển được chọn đạt Macro-F1 hướng 0.554/0.535 và PR-AUC stress 0.612/0.624 trên holdout BTC/ETH đã chấm một lần. DeepLOB dùng tối đa 8.000 mẫu train và khoảng 2.500 điểm validation; chưa có kết quả holdout.",
  },
  {
    id: "RQ4",
    status: "CHƯA ĐỦ ĐỂ KHẲNG ĐỊNH ỔN ĐỊNH",
    tone: "limited",
    question: "Kết quả có ổn định giữa cặp, tuần và chế độ biến động không?",
    answer: "Kết quả thay đổi theo tuần và chế độ biến động; pilot hiện chỉ có một tuần validation và một tuần holdout nên chưa thể khẳng định khả năng khái quát dài hạn.",
    evidence: "Từ validation sang holdout ở 5 giây, Macro-F1 hướng BTC/ETH giảm 0.595→0.554 và 0.562→0.535; PR-AUC stress tăng 0.517→0.612 và 0.593→0.624. Ở holdout, Macro-F1 hướng ETH tại 10 giây là 0.468; PR-AUC stress từ 1 xuống 10 giây giảm 0.772→0.559 (BTC) và 0.748→0.587 (ETH). Theo biến động cao/thấp, PR-AUC stress là 0.757/0.178 (BTC) và 0.751/0.123 (ETH).",
  },
  {
    id: "RQ5",
    status: "KẾT LUẬN NHẠY VỚI MẤT CÂN BẰNG VÀ NHÃN",
    tone: "limited",
    question: "Kết luận có còn đúng khi xét mất cân bằng lớp, calibration và độ nhạy nhãn không?",
    answer: "Đánh giá bằng PR-AUC, MCC và calibration cho thấy mô hình stress có ích, nhưng nhãn spread-only suy biến; vì vậy kết quả phụ thuộc rõ vào cách định nghĩa stress.",
    evidence: "Holdout stress composite: tỷ lệ dương 4.7% BTC và 10.3% ETH; PR-AUC 0.612/0.624, precision 87.2%/90.7% nhưng recall chỉ 43.1%/40.9%. Brier stress là 0.026/0.058. Với nhãn spread-only Q95, tỷ lệ dương chỉ khoảng 0.05% ở mỗi cặp và mô hình không dự báo được ca dương nào. Kết quả chính vì thế dùng nhãn composite và cần được đọc cùng các giới hạn này.",
  },
] as const;

function ResearchPage() {
  return (
    <div className="research-page">
      <section className="research-intro">
        <span className="lbl">MICROSTRUCTURE / RESEARCH PILOT</span>
        <h1>Hình nghiên cứu &amp; bằng chứng</h1>
        <p>Khám phá cấu trúc sổ lệnh, tín hiệu OBI và kết quả dự báo cho BTCUSDT, ETHUSDT.</p>
      </section>

      <aside className="research-boundary" aria-label="Phạm vi kết quả nghiên cứu">
        <strong>Phạm vi cần nhớ</strong>
        <p>DeepLOB mới được so sánh trên tuần validation 2026-10-05–2026-10-11; kết quả chưa vượt Logistic/XGBoost. Holdout cổ điển đã chấm một lần và các hình của holdout chỉ dùng số liệu tổng hợp đã lưu. Không có kết quả DeepLOB trên holdout.</p>
      </aside>

      <section className="research-answers" aria-labelledby="research-answers-title">
        <div className="research-answers-head">
          <span className="lbl">KẾT LUẬN TỪ PILOT · HORIZON CHÍNH 5 GIÂY</span>
          <h2 id="research-answers-title">Trả lời câu hỏi nghiên cứu</h2>
          <p>Các câu trả lời dưới đây tóm tắt kết quả đã lưu, kèm số liệu và giới hạn cần biết. Holdout không được chấm lại.</p>
        </div>
        <div className="research-answer-list">
          {researchAnswers.map((item) => (
            <article className="research-answer" key={item.id}>
              <div className="research-answer-meta">
                <span className="research-answer-id">{item.id}</span>
                <span className={`research-answer-status ${item.tone}`}>{item.status}</span>
              </div>
              <div className="research-answer-body">
                <h3>{item.question}</h3>
                <p className="research-answer-conclusion">{item.answer}</p>
                <p className="research-answer-evidence"><strong>Bằng chứng:</strong> {item.evidence}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="research-grid" aria-label="Các hình nghiên cứu">
        {researchFigures.map((figure) => (
          <article className="research-card" key={figure.file}>
            <div className="research-card-head">
              <div><span className="lbl">{figure.scope}</span><h2>{figure.title}</h2></div>
              <a href={`/research/${figure.file}`} target="_blank" rel="noreferrer" aria-label={`Mở hình: ${figure.title}`}>mở hình ↗</a>
            </div>
            <div className="research-image"><img src={`/research/${figure.file}`} alt={figure.title} loading="lazy" /></div>
            <section className="research-analysis" aria-label={`Diễn giải: ${figure.title}`}>
              <div><span className="lbl">cách đọc</span><p>{figure.reading}</p></div>
              <div><span className="lbl">số liệu chính</span><p>{figure.numbers}</p></div>
              <div><span className="lbl">ý nghĩa nghiên cứu</span><p>{figure.finding}</p></div>
            </section>
          </article>
        ))}
      </section>
      <p className="research-footnote">Các hình được tạo từ dữ liệu validation hoặc kết quả tổng hợp đã lưu. Chúng không cập nhật theo luồng realtime của dashboard.</p>
    </div>
  );
}

export function LOBApp() {
  const [appView, setAppView] = useState<AppView>("dashboard");
  const [mode, setMode] = useState<Mode>("live");
  const [symbol, setSymbol] = useState<SymbolCode>("BTCUSDT");
  const [horizon, setHorizon] = useState(5);
  const [requestedTime, setRequestedTime] = useState("");
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [meter, setMeter] = useState<Meter | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState("starting");
  const [liveReason, setLiveReason] = useState<string | null>("Đang khởi động nguồn dữ liệu Binance Spot.");
  const requestVersion = useRef(0);

  const refresh = useCallback(async () => {
    const version = ++requestVersion.current;
    const isCurrent = () => version === requestVersion.current;
    setBusy(true);
    setError(null);
    setSnapshot(null);
    setMeter(null);
    try {
      const nextStatus = await getJson<DataStatus>("/api/v1/data-status");
      if (!isCurrent()) return;
      setStatus(nextStatus);
      let next: Snapshot | null = null;
      if (requestedTime) {
        const ts = new Date(requestedTime).toISOString();
        const r = await getJson<{ snapshot: Snapshot | null; reason?: string }>(
          `/api/v1/replay/${symbol}/at?timestamp_utc=${encodeURIComponent(ts)}&depth=20`,
        );
        if (!isCurrent()) return;
        next = r.snapshot;
        if (!next) {
          setError(noReplayMessage(symbol, ts, nextStatus, r.reason));
          return;
        }
      } else {
        const r = await getJson<{ snapshots: Snapshot[] }>(`/api/v1/replay/${symbol}?limit=1&depth=20`);
        if (!isCurrent()) return;
        next = r.snapshots.at(-1) ?? null;
        if (!next) {
          setError(noReplayMessage(symbol, undefined, nextStatus));
          return;
        }
      }
      setSnapshot(next);
      if (!requestedTime) setRequestedTime(toLocalInput(next.timestamp_utc));
      const nextMeter = await getJson<Meter>(`/api/v1/meter/${symbol}?timestamp_utc=${encodeURIComponent(next.timestamp_utc)}&horizon_seconds=${horizon}`);
      if (!isCurrent()) return;
      setMeter(nextMeter);
    } catch (cause) {
      if (isCurrent()) setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (isCurrent()) setBusy(false);
    }
  }, [horizon, requestedTime, symbol]);

  useEffect(() => {
    if (mode === "replay") void refresh();
  }, [mode, refresh]);

  useEffect(() => {
    if (mode !== "live") return;
    requestVersion.current += 1;
    setError(null);
    setSnapshot(null);
    setMeter(null);
    setLiveStatus("connecting");
    setLiveReason("Đang mở luồng dữ liệu Binance Spot.");
    let active = true;
    const stream = new EventSource(`${API_ROOT}/api/v1/live/${symbol}/stream?horizon_seconds=${horizon}`);
    stream.onmessage = (event) => {
      if (!active) return;
      try {
        const update = JSON.parse(event.data) as LiveUpdate;
        setLiveStatus(update.status);
        setLiveReason(update.reason ?? null);
        setSnapshot(update.snapshot);
        setMeter(update.meter);
      } catch {
        setLiveStatus("error");
        setLiveReason("Không đọc được dữ liệu cập nhật từ máy chủ.");
      }
    };
    stream.onerror = () => {
      if (!active) return;
      setLiveStatus("reconnecting");
      setLiveReason("Mất kết nối tới máy chủ live; trình duyệt đang tự kết nối lại.");
    };
    return () => {
      active = false;
      stream.close();
    };
  }, [mode, symbol, horizon]);

  const symbolStatus = status?.symbols?.[symbol];
  const f = snapshot?.features ?? {};
  const imb = meter?.depth_imbalance ?? {};
  const cum = (levels: Level[]) => levels.slice(0, 10).reduce((s, [, q]) => s + (Number(q) || 0), 0);
  const scale = Math.max(cum(snapshot?.bids ?? []), cum(snapshot?.asks ?? []));
  const ready = meter?.status === "ok";
  const modelReady = Boolean(meter?.model_version);
  const stressThreshold = meter?.stress_probability_threshold;
  const stressFlag = ready && stressThreshold != null && meter?.p_stress != null && meter.p_stress >= stressThreshold;
  const bookBad = snapshot ? !snapshot.quality.book_valid || !!snapshot.quality.sequence_gap : false;

  return (
    <main className="app">
      <header className="top">
        <strong>liquidity stress meter</strong>
        <nav className="view-tabs" aria-label="Khu vực dashboard">
          <button type="button" aria-current={appView === "dashboard" ? "page" : undefined} onClick={() => setAppView("dashboard")}>dashboard</button>
          <button type="button" aria-current={appView === "research" ? "page" : undefined} onClick={() => setAppView("research")}>nghiên cứu</button>
        </nav>
        <span className="mu">{mode === "live" ? `LIVE · ${liveStatusLabel(liveStatus)} · chỉ đọc · không đặt lệnh` : "replay · chỉ đọc · không đặt lệnh"}</span>
      </header>

      {appView === "dashboard" ? (
      <>
      <section className="controls" aria-label="Điều khiển replay">
        <Segmented label="chế độ" value={mode} onChange={setMode}
          options={[{ value: "live", text: "LIVE" }, { value: "replay", text: "replay" }]} />
        <Segmented label="cặp" value={symbol} onChange={(nextSymbol) => {
          if (nextSymbol === symbol) return;
          setSymbol(nextSymbol);
          setRequestedTime("");
          setSnapshot(null);
          setMeter(null);
          setError(null);
        }}
          options={[{ value: "BTCUSDT", text: "BTC/USDT" }, { value: "ETHUSDT", text: "ETH/USDT" }]} />
        <Segmented label="horizon" value={horizon} onChange={setHorizon}
          options={[{ value: 1, text: "1s" }, { value: 5, text: "5s" }, { value: 10, text: "10s" }]} />
        {mode === "replay" ? (
          <>
            <label className="field grow">
              <span className="lbl">thời điểm replay · giờ máy</span>
              <input
                type="datetime-local"
                value={requestedTime}
                min={symbolStatus?.first_date_utc ? toLocalInput(`${symbolStatus.first_date_utc}T00:00:00Z`) : undefined}
                max={symbolStatus?.last_date_utc ? toLocalInput(`${symbolStatus.last_date_utc}T23:59:59Z`) : undefined}
                onChange={(e) => setRequestedTime(e.target.value)}
              />
            </label>
            <button type="button" className="go" onClick={() => void refresh()} disabled={busy}>{busy ? "đang tải…" : "nạp replay"}</button>
          </>
        ) : (
          <div className={`live-indicator ${liveStatus === "live" ? "is-live" : ""}`} role="status">
            <span>{liveStatusLabel(liveStatus)}</span>
            <small>{liveReason ?? "Binance Spot · WebSocket công khai"}</small>
          </div>
        )}
      </section>

      {mode === "replay" && error && <p className="alert" role="alert"><b>✕ lỗi</b> {error}</p>}
      {mode === "live" && ["stale", "error", "disabled", "stopped"].includes(liveStatus) && liveReason &&
        <p className="alert" role="alert"><b>✕ live</b> {liveReason}</p>}

      <section className="stats" aria-label="Chỉ số chính">
        <Stat label="mid-price" value={num(f.mid_price, 2)} source="(ask1 + bid1) / 2" />
        <Stat label="spread" value={f.relative_spread_bps == null ? "—" : num(f.relative_spread_bps, 4)} unit=" bps" source="ask1 − bid1 trên mid" />
        <Stat label="depth imbalance · l10" value={num(imb.L10, 4, true)} source="chênh lệch qty bid/ask, −1 đến +1" />
        <Stat label="order flow · ofi 5s" value={num(meter?.ofi_5s, 2, true)} source="ofi theo sự kiện, cộng 5 giây" />
      </section>

      <section className="bookgrid" aria-label="Sổ lệnh">
        <BookSide side="bid" levels={snapshot?.bids ?? []} scale={scale} />
        <BookSide side="ask" levels={snapshot?.asks ?? []} scale={scale} />
        <aside className="imb">
          <h3>imbalance theo lớp</h3>
          {[1, 5, 10, 20].map((d) => {
            const v = Number(imb[`L${d}`]);
            const ok = Number.isFinite(v);
            return (
              <div className="imbrow" key={d}>
                <span>L{d}</span>
                <div className="center"><i className={v >= 0 ? "data" : "ink"} style={{ width: `${ok ? Math.min(50, Math.abs(v) * 50) : 0}%`, left: v >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(v) * 50)}%` }} /></div>
                <b>{num(ok ? v : null, 3, true)}</b>
              </div>
            );
          })}
          <p className="src">dương: bid dày hơn · âm: ask dày hơn</p>
        </aside>
      </section>

      <section className="models">
        <article>
          <h3>hướng mid-price · {horizon}s</h3>
          {ready ? (
            <>
              <ProbRow mark="▲" label="up" value={meter?.p_up} tone="data" />
              <ProbRow mark="▬" label="flat" value={meter?.p_flat} tone="mute" />
              <ProbRow mark="▼" label="down" value={meter?.p_down} tone="ink" />
            </>
          ) : (
            <p className="empty"><b>— chưa có xác suất.</b> {meter?.reason ?? "Replay hoặc classifier chưa sẵn sàng."}</p>
          )}
          <span className="src">nguồn · {meter?.model_version ?? "chưa có model"} · δ fit trên tập train</span>
        </article>
        <article>
          <h3>xác suất stress · {horizon}s</h3>
          {ready ? (
            <>
              <strong className={`big ${stressFlag ? "breach" : ""}`}>{pct(meter?.p_stress)}</strong>
              {stressThreshold != null ? (
                <span className={stressFlag ? "flag breach" : "flag"}>
                  {stressFlag ? `■ vượt ngưỡng ${pct(stressThreshold)}` : `□ dưới ngưỡng ${pct(stressThreshold)}`}
                </span>
              ) : (
                <span className="flag">Chưa có ngưỡng quyết định trong model artifact</span>
              )}
              <div className="track"><i className={stressFlag ? "breach" : "data"} style={{ width: `${Math.min(100, Number(meter?.p_stress ?? 0) * 100)}%` }} /></div>
            </>
          ) : (
            <p className="empty"><b>— meter chưa sẵn sàng.</b> {meter?.reason ?? "Cần replay data và model artifact tương thích."}</p>
          )}
          <span className="src">nguồn · spread, price impact, depth top-10, rv · ngưỡng chọn trên dự báo walk-forward validation</span>
        </article>
      </section>

      <section className="quality" aria-label="Trạng thái dữ liệu">
        <div><span className="lbl">{mode === "live" ? "nguồn dữ liệu" : "replay data"}</span>
          <strong>{mode === "live" ? liveStatusLabel(liveStatus) : symbolStatus?.status === "ready" ? "sẵn sàng" : "chưa có"}</strong>
          <span className="src">{mode === "live" ? liveReason ?? "Binance Spot · WebSocket công khai" : symbolStatus ? `${num(symbolStatus.row_count, 0)} dòng · ${symbolStatus.partition_count} partition` : "chưa có manifest"}</span></div>
        <div><span className="lbl">{mode === "live" ? "cập nhật gần nhất" : "khoảng quan sát"}</span>
          <strong>{mode === "live" ? formatTime(snapshot?.timestamp_utc) : `${symbolStatus?.first_date_utc ?? "—"} → ${symbolStatus?.last_date_utc ?? "—"}`}</strong>
          <span className="src">{mode === "live" ? "thời gian sàn · UTC hiển thị theo giờ Việt Nam" : formatTime(snapshot?.timestamp_utc)}</span></div>
        <div><span className="lbl">chất lượng book</span>
          <strong className={bookBad ? "breach" : ""}>{!snapshot ? "đang chờ dữ liệu" : bookBad ? "✕ book lỗi / có gap" : "✓ hợp lệ"}</strong>
          <span className="src">{snapshot?.quality.seconds_since_gap != null ? `${snapshot.quality.seconds_since_gap}s từ gap gần nhất` : "sequence và độ tươi"}</span></div>
        <div><span className="lbl">model artifact</span><strong>{modelReady ? "sẵn sàng" : "chưa có"}</strong>
          <span className="src">{meter?.model_version ?? meter?.reason ?? "chưa nạp model"}</span></div>
      </section>

      <footer className="foot">
        <p>{mode === "live"
          ? "Nguồn live là Binance Spot public WebSocket, lưu capture liên tục và không đặt lệnh. Các xác suất chỉ xuất hiện sau khi đủ dữ liệu rolling 5 phút và có model tương thích."
          : "Replay đọc dữ liệu đã lưu. Dữ liệu capture live không thể khôi phục sổ lệnh L2 của những ngày trước thời điểm capture."}</p>
        <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">api · 127.0.0.1:8000/docs</a>
      </footer>
      </>
      ) : <ResearchPage />}
    </main>
  );
}
