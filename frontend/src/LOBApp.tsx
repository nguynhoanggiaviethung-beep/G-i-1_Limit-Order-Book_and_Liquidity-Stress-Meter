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

const researchGlossary = [
  {
    term: "Sổ lệnh (order book)",
    meaning: "Danh sách mọi lệnh mua và bán đang chờ khớp trên sàn. Bên mua (bid) ghi giá họ sẵn sàng trả; bên bán (ask) ghi giá họ muốn nhận.",
  },
  {
    term: "Mid-price",
    meaning: "Điểm giữa giá mua cao nhất và giá bán thấp nhất. Nghiên cứu dự báo con số này sẽ tăng, giảm hay đứng yên sau vài giây.",
  },
  {
    term: "Spread",
    meaning: "Khoảng cách giữa giá bán thấp nhất và giá mua cao nhất. Spread càng nhỏ, mua bán càng rẻ.",
  },
  {
    term: "OBI",
    meaning: "Độ lệch giữa khối lượng đang chờ mua và chờ bán, từ −1 (toàn người bán) đến +1 (toàn người mua). Giống việc nhìn xem hàng người chờ mua hay hàng chờ bán dài hơn.",
  },
  {
    term: "OFI",
    meaning: "Dòng lệnh vừa đổ vào trong vài giây qua: bên mua hay bên bán đang thêm lệnh, rút lệnh nhanh hơn. OBI là ảnh chụp, OFI là đoạn phim ngắn.",
  },
  {
    term: "Stress thanh khoản",
    meaning: "Lúc thị trường trở nên khó giao dịch: spread giãn rộng, sổ lệnh mỏng, giá giật mạnh, hoặc một lệnh 2.000 USDT đã làm giá trượt nhiều.",
  },
  {
    term: "Persistence",
    meaning: "Cách đoán đơn giản nhất để so sánh: giả định mọi thứ sẽ giống vài giây trước. Giá vừa tăng thì đoán tiếp tục tăng; đang stress thì đoán vẫn stress.",
  },
  {
    term: "MCC",
    meaning: "Điểm chấm chất lượng dự báo hướng giá, từ −1 đến 1. Bằng 0 nghĩa là không hơn đoán mò; càng cao càng tốt. Không bị đánh lừa khi một lớp chiếm đa số.",
  },
  {
    term: "PR-AUC",
    meaning: "Điểm chấm khả năng báo trước stress, từ 0 đến 1. Đo xem mô hình có xếp những lúc sắp stress lên đầu danh sách cảnh báo hay không. Càng cao càng tốt.",
  },
  {
    term: "Development và test",
    meaning: "Development (19/04–23/08) dùng để xây và chọn mô hình. Test (24/08–27/09) được giấu đi suốt quá trình đó và chỉ mở ra chấm điểm đúng một lần, như một bài thi chưa từng thấy đề.",
  },
] as const;

const researchFigures = [
  {
    title: "Mô hình có tốt đều qua từng tuần không?",
    file: "6m/weekly_performance_6m.png",
    scope: "TEST · DỮ LIỆU CHƯA TỪNG DÙNG",
    reading: "Mỗi chấm là điểm của một tuần. Đường xanh là mô hình, đường xám là cách đoán đơn giản (persistence). Phần không tô màu là 12 tuần dùng để xây mô hình; phần tô hồng là 5 tuần test. Cột trái là dự báo hướng giá (điểm MCC), cột phải là báo trước stress (điểm PR-AUC). Khoảng dự báo: 5 giây.",
    numbers: "Ở cả 5 tuần test, đường xanh luôn nằm trên đường xám. Ví dụ với BTC: điểm hướng giá của mô hình là 0,36–0,40, cách đoán đơn giản chỉ 0,24–0,26.",
    finding: "Mô hình không chỉ giỏi trên dữ liệu đã học: sang dữ liệu mới nó vẫn tốt ngang như trước. Điểm stress lên xuống mạnh giữa các tuần, vì có tuần thị trường nhiều biến động, có tuần rất yên.",
  },
  {
    title: "Con số xác suất có đáng tin không?",
    file: "6m/calibration_test_6m.png",
    scope: "TEST · DỮ LIỆU CHƯA TỪNG DÙNG",
    reading: "Câu hỏi của hình: khi mô hình nói \"30% khả năng stress\", thì thực tế có xảy ra khoảng 30% số lần không? Trục ngang là xác suất mô hình đưa ra, trục dọc là tỷ lệ thực sự xảy ra. Nếu đường bám sát đường chéo nét đứt thì con số xác suất đáng tin.",
    numbers: "Đường stress bám rất sát đường chéo: sai lệch trung bình chỉ 0,4% (BTC) và 1,6% (ETH). Mô hình báo động khi xác suất vượt khoảng 42–45%; khi đã báo thì đúng 82–86% số lần.",
    finding: "Có thể đọc thẳng con số xác suất như một mức rủi ro. Nhưng mô hình chỉ báo trước được khoảng một nửa số lần stress, nên không có cảnh báo không có nghĩa là an toàn.",
  },
  {
    title: "Mô hình hay đoán sai kiểu gì?",
    file: "6m/confusion_matrices_test_6m.png",
    scope: "TEST · DỮ LIỆU CHƯA TỪNG DÙNG",
    reading: "Mỗi hàng là điều thực sự xảy ra, mỗi cột là điều mô hình đoán. Ô trên đường chéo (từ trái trên xuống phải dưới) là đoán đúng; ô màu càng đậm thì càng nhiều lần. Bên trái là hướng giá (down = giảm, flat = đứng yên, up = tăng); bên phải là stress (1 = có stress).",
    numbers: "Khi giá BTC thực sự tăng, mô hình đoán đúng \"tăng\" 41% số lần, đoán \"đứng yên\" 45%, và chỉ đoán ngược thành \"giảm\" 14%. Với stress, mô hình bắt được 48% số lần stress, và chỉ báo nhầm khoảng 11.650 lần trong gần 3 triệu giây bình thường.",
    finding: "Mô hình thận trọng: khi không chắc, nó đoán \"đứng yên\" chứ hiếm khi đoán ngược chiều. Với stress, nó ít khi báo động sai, đổi lại bỏ sót khoảng một nửa.",
  },
  {
    title: "OBI và OFI có thật sự giúp dự báo không?",
    file: "6m/paired_contrasts_test_6m.png",
    scope: "TEST · SO SÁNH TỪNG CẶP",
    reading: "Mỗi chấm là mức chênh lệch điểm giữa hai cách dự báo, thanh ngang hai bên là khoảng sai số. Chấm nằm bên phải đường đỏ: cách thứ nhất tốt hơn. Bên trái: tệ hơn. Ba màu là ba khoảng dự báo: 1, 5 và 10 giây.",
    numbers: "Chỉ dùng OBI thua cách đoán đơn giản ở 1 và 5 giây, và chỉ thắng nhẹ ở 10 giây (+0,02 điểm). Thêm OFI vào OBI gần như không thay đổi (tăng dưới 0,01). Kết hợp tất cả thông tin (spread, độ sâu, biến động, giao dịch) thì tăng mạnh, khoảng +0,15.",
    finding: "Riêng OBI hay riêng OFI đều yếu. Sức mạnh đến từ việc nhìn nhiều thông tin cùng lúc. Với stress, mô hình vượt cách đoán đơn giản rõ rệt ở mọi khoảng dự báo.",
  },
  {
    title: "Stress dễ đoán vào lúc nào?",
    file: "6m/stress_by_regime_test_6m.png",
    scope: "TEST · THEO TRẠNG THÁI THỊ TRƯỜNG",
    reading: "Thị trường được chia theo mức biến động (thấp, trung bình, cao) và theo phiên giao dịch giờ quốc tế (Á, Âu, Mỹ). Cột đỏ là mô hình, cột xám là cách đoán đơn giản. Cột càng cao càng báo trước stress tốt.",
    numbers: "Khi thị trường biến động mạnh, điểm của mô hình đạt khoảng 0,72 cho cả hai đồng. Khi thị trường yên, điểm chỉ còn 0,22 (BTC) và 0,07 (ETH). Phiên châu Âu dễ đoán nhất, phiên châu Á khó nhất.",
    finding: "Mô hình báo stress tốt nhất khi thị trường đã bắt đầu dậy sóng. Những cú sốc bất ngờ giữa lúc yên ả thì gần như không đoán trước được; đây là giới hạn lớn nhất.",
  },
  {
    title: "Sổ lệnh trông như thế nào?",
    file: "6m/lob_heatmap_development_6m.png",
    scope: "DỮ LIỆU XÂY MÔ HÌNH",
    reading: "Hình chụp 5 phút sổ lệnh ngày 17/08/2026. Trục ngang là thời gian. Trục dọc là khoảng cách tới giá giữa, tính bằng bước giá nhỏ nhất (tick, 0,01 USDT): phía trên là lệnh bán, phía dưới là lệnh mua. Màu càng đậm, càng nhiều tiền đang chờ ở mức giá đó.",
    numbers: "Với BTC, gần như toàn bộ lệnh dồn ở mức giá sát nhất, phía sau rất mỏng. ETH thì lệnh trải đều qua 10 mức. Khoảng cách mua–bán của cả hai gần như luôn chỉ 1 bước giá.",
    finding: "BTC trông có vẻ rất thanh khoản vì khoảng cách mua–bán siêu hẹp, nhưng một lệnh lớn sẽ nhanh chóng \"ăn\" hết lớp lệnh đầu tiên và làm giá trượt. Vì vậy, đo stress không thể chỉ nhìn spread.",
  },
  {
    title: "OBI có liên quan đến giá sắp tới không?",
    file: "6m/obi_vs_forward_return_development_6m.png",
    scope: "DỮ LIỆU XÂY MÔ HÌNH",
    reading: "Chia mọi giây thành 10 nhóm, từ lúc phe bán áp đảo (nhóm 1) đến lúc phe mua áp đảo (nhóm 10). Trục dọc là giá thay đổi bao nhiêu sau 5 giây, tính bằng bps (1 bps = 0,01%). Đường xanh là mức trung bình, đường đỏ là mức ở giữa.",
    numbers: "Đi từ nhóm 1 đến nhóm 10, giá sau 5 giây tăng đều từ −0,6 lên +0,6 bps với BTC (ETH: −0,8 lên +0,8). Tuy vậy, ở 8 trên 10 nhóm, giá ở giữa vẫn là \"không đổi\".",
    finding: "Phe nào đông hơn trong sổ lệnh thì giá có xu hướng nhích về phía đó, đúng như lý thuyết. Nhưng mức nhích dưới 0,01%, nhỏ hơn nhiều so với phí giao dịch thông thường (0,1%). Tức là có tín hiệu, nhưng không đủ để kiếm lời.",
  },
] as const;

const researchAnswers = [
  {
    id: "RQ1",
    status: "CÓ, NHƯNG YẾU",
    tone: "mixed",
    question: "Nhìn bên mua hay bên bán đông hơn (OBI) có đoán được giá sắp đi đâu không?",
    answer: "Có một chút. OBI tốt hơn đoán mò, nhưng chỉ tốt hơn cách đoán \"giá sẽ đi tiếp như vừa rồi\" khi dự báo xa 10 giây. Với 1 và 5 giây, cách đoán đơn giản đó còn tốt hơn.",
    evidence: "Điểm MCC của OBI so với cách đoán đơn giản: thua 0,09 ở 1 giây, thua 0,03 ở 5 giây, thắng 0,02 ở 10 giây (BTC; ETH tương tự). Kết quả lặp lại y hệt trên dữ liệu xây mô hình, nên không phải ngẫu nhiên.",
  },
  {
    id: "RQ2",
    status: "CÓ THÊM, NHƯNG RẤT ÍT",
    tone: "partial",
    question: "Dòng lệnh vừa đổ vào (OFI) có cho thêm thông tin ngoài OBI không?",
    answer: "Có, nhưng rất ít và biến mất nhanh. Thông tin về dòng lệnh chỉ có ích trong 1–2 giây, sau đó giá đã phản ánh xong.",
    evidence: "Thêm OFI vào OBI chỉ tăng điểm khoảng 0,005–0,007 ở 1 giây và gần bằng 0 ở 10 giây. Trong khi đó, dùng đầy đủ thông tin (spread, độ sâu sổ lệnh, biến động, giao dịch) tăng tới khoảng 0,15.",
  },
  {
    id: "RQ3",
    status: "KHÔNG",
    tone: "negative",
    question: "Mô hình học sâu DeepLOB có tốt hơn các mô hình đơn giản không?",
    answer: "Không. Mạng nơ-ron DeepLOB thua mô hình đơn giản (Logistic, XGBoost) ở mọi tuần thử nghiệm, nên không được đem đi thi trên dữ liệu test.",
    evidence: "DeepLOB thua ở 12/12 tuần với cả hai đồng, cả dự báo giá lẫn stress. Ví dụ báo trước stress của BTC: DeepLOB 0,28 điểm, XGBoost 0,54. Lý do có thể là dữ liệu huấn luyện cho DeepLOB còn ít (8.000 mẫu mỗi lần), nên kết luận chỉ đúng ở quy mô này.",
  },
  {
    id: "RQ4",
    status: "ỔN QUA THỜI GIAN, KHÁC THEO THỊ TRƯỜNG",
    tone: "mixed",
    question: "Mô hình có còn tốt khi thị trường thay đổi không?",
    answer: "Qua các tuần thì ổn định: mô hình thắng cách đoán đơn giản ở mọi tuần, kể cả 5 tuần test chưa từng thấy. Nhưng mức tốt phụ thuộc nhiều vào thị trường đang sôi động hay yên ả.",
    evidence: "Điểm dự báo giá BTC trên test (0,37) gần với lúc xây mô hình (0,36). Báo trước stress đạt 0,72 điểm khi thị trường biến động mạnh, nhưng chỉ 0,07–0,22 khi thị trường yên. Lưu ý: test chỉ dài 5 tuần.",
  },
  {
    id: "RQ5",
    status: "ACCURACY GÂY HIỂU NHẦM",
    tone: "partial",
    question: "Có nên chấm mô hình bằng tỷ lệ đoán đúng (accuracy) không?",
    answer: "Không nên. Vì phần lớn thời gian giá đứng yên và thị trường bình thường, một cách đoán \"lười\" cũng đạt tỷ lệ đúng rất cao. Phải dùng các điểm chấm công bằng hơn như MCC và PR-AUC.",
    evidence: "Luôn đoán \"giá đứng yên\" đã đúng 82% số lần ở khoảng 1 giây, gần bằng mô hình (84%), nhưng điểm MCC của nó bằng 0, tức không dự báo được gì. Tương tự, luôn đoán \"không stress\" đúng 96%. Khi dùng điểm chấm công bằng, mô hình vẫn hơn rõ, và xác suất của nó khớp với thực tế.",
  },
] as const;

function ResearchPage() {
  return (
    <div className="research-page">
      <section className="research-intro">
        <span className="lbl">MICROSTRUCTURE / NGHIÊN CỨU 6 THÁNG</span>
        <h1>Nghiên cứu nói gì?</h1>
        <p>Dùng sổ lệnh của BTC/USDT và ETH/USDT trên Binance, nghiên cứu thử trả lời hai câu: giá sẽ tăng hay giảm trong vài giây tới, và thị trường sắp trở nên khó giao dịch (stress) hay không.</p>
      </section>

      <aside className="research-boundary" aria-label="Phạm vi kết quả nghiên cứu">
        <strong>Tóm tắt trong 30 giây</strong>
        <p>Sổ lệnh có chứa tín hiệu về giá sắp tới, nhưng tín hiệu rất nhỏ, không đủ bù phí giao dịch. Điểm mạnh thật sự của mô hình là báo trước lúc thị trường sắp khó giao dịch, với xác suất đáng tin. Mọi con số dưới đây chấm trên 5 tuần dữ liệu (24/08–27/09/2026) mà mô hình chưa từng thấy, và chỉ chấm một lần.</p>
      </aside>

      <section className="research-glossary" aria-labelledby="research-glossary-title">
        <div className="research-answers-head">
          <span className="lbl">ĐỌC TRƯỚC</span>
          <h2 id="research-glossary-title">Giải thích thuật ngữ</h2>
        </div>
        <dl>
          {researchGlossary.map((item) => (
            <div key={item.term}><dt>{item.term}</dt><dd>{item.meaning}</dd></div>
          ))}
        </dl>
      </section>

      <section className="research-answers" aria-labelledby="research-answers-title">
        <div className="research-answers-head">
          <span className="lbl">KẾT LUẬN</span>
          <h2 id="research-answers-title">Năm câu hỏi nghiên cứu và câu trả lời</h2>
          <p>Mỗi câu gồm: câu trả lời ngắn, rồi con số cụ thể để kiểm chứng.</p>
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
                <p className="research-answer-evidence"><strong>Con số cụ thể:</strong> {item.evidence}</p>
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
              <div><span className="lbl">cách đọc hình</span><p>{figure.reading}</p></div>
              <div><span className="lbl">con số đáng chú ý</span><p>{figure.numbers}</p></div>
              <div><span className="lbl">điều rút ra</span><p>{figure.finding}</p></div>
            </section>
          </article>
        ))}
      </section>
      <p className="research-footnote">Các hình được tạo từ kết quả và dự báo đã lưu (backend/scripts/build_six_month_figures.py). Chúng không cập nhật theo luồng realtime của dashboard. Chi tiết phương pháp nằm trong báo cáo paper/bao_cao_nghien_cuu.md.</p>
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
