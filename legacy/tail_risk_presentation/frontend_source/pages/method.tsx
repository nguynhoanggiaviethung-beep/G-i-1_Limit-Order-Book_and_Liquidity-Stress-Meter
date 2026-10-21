import { useEffect, useState } from "react";
import { Database, Download, ExternalLink, FileText, Image as ImageIcon } from "lucide-react";
import { getResultFigureUrl } from "../api/client";
import { useMeta, useTable } from "../api/queries";
import { useFilters } from "../app/filters";
import { DataTable } from "../components/DataTable";
import { ApiState, PageIntro, SectionHeader } from "../components/PageParts";
import { formatDate, shortHash } from "../components/format";

const sourceTables = [
  { id: "T1", label: "T1 · Mô tả dữ liệu" },
  { id: "T2", label: "T2 · Chất lượng snapshot" },
  { id: "T11", label: "T11 · Robustness" },
];

function FigurePreview({ figureId }: { figureId: string }) {
  const [url, setUrl] = useState<string>();
  const [error, setError] = useState<string>();
  useEffect(() => {
    let current: string | undefined;
    let active = true;
    getResultFigureUrl(figureId).then((result) => { current = result; if (active) setUrl(result); else URL.revokeObjectURL(result); }).catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Không tải được hình."); });
    return () => { active = false; if (current) URL.revokeObjectURL(current); };
  }, [figureId]);
  return <article className="figure-card"><div className="panel-heading"><h3><ImageIcon size={15} /> {figureId}</h3><a href={url} download={`${figureId}.svg`} aria-label={`Tải ${figureId}`} className="icon-button"><Download size={15} /></a></div>
    {url ? <img src={url} alt={`Hình nghiên cứu ${figureId}`} /> : <div className="figure-empty">{error ?? "Đang tải SVG…"}</div>}
  </article>;
}

export function MethodPage() {
  const [selected, setSelected] = useState("T1");
  const { filters } = useFilters();
  const table = useTable(selected);
  const meta = useMeta(filters.mode);
  return <div className="page-stack">
    <PageIntro eyebrow="TÀI LIỆU / 06" title="Phương pháp & dữ liệu" description="Thông tin tái lập, provenance và tài sản đầu ra được phục vụ từ backend nghiên cứu." actions={<a className="secondary-button" href="/docs" target="_blank" rel="noreferrer"><ExternalLink size={14} /> API docs</a>} />
    <div className="overview-grid three-tiles">
      <article className="info-panel"><div className="panel-heading"><h3><Database size={15} /> Nguồn</h3></div><p className="large-copy">{meta.data?.source ?? "—"}</p><span className="muted">Binance Spot · nến 5 phút</span></article>
      <article className="info-panel"><div className="panel-heading"><h3>Mốc nghiên cứu</h3><span className="badge">UTC</span></div><p className="large-copy">{formatDate(meta.data?.as_of_utc)}</p><span className="muted">Snapshot được dùng cho so sánh walk-forward.</span></article>
      <article className="info-panel"><div className="panel-heading"><h3><FileText size={15} /> Hash cấu hình</h3></div><p className="large-copy mono">{shortHash(meta.data?.config_hash)}</p><span className="muted">Config version dùng để tạo kết quả.</span></article>
    </div>
    <div className="callout"><Database size={16} /><span>UI chỉ hiển thị bảng, biểu đồ và dự báo do backend trả về. Phép tính forecast, VaR/ES, coverage và các kiểm định không chạy trong trình duyệt.</span></div>
    <SectionHeader title="Dữ liệu và robustness" note="Tài sản T1, T2 và T11 do pipeline xuất." />
    <div className="tab-list" role="tablist" aria-label="Bảng dữ liệu và robustness">
      {sourceTables.map((item) => <button key={item.id} role="tab" aria-selected={selected === item.id} className={selected === item.id ? "tab-active" : ""} onClick={() => setSelected(item.id)}>{item.label}</button>)}
    </div>
    <ApiState loading={table.isLoading} error={table.error}>{table.data && <DataTable table={table.data} descriptive={filters.horizon === 30} filterModel={false} />}</ApiState>
    <SectionHeader title="Hình nghiên cứu" note="SVG gốc được tải từ endpoint artifact; thiếu hình sẽ hiển thị trạng thái API." />
    <div className="figure-grid">{["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"].map((id) => <FigurePreview key={id} figureId={id} />)}</div>
  </div>;
}
