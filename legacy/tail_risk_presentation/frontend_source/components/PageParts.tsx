import { AlertCircle, Database, LoaderCircle } from "lucide-react";
import { useMeta } from "../api/queries";
import { useFilters } from "../app/filters";
import { formatDate, shortHash } from "./format";

export function PageIntro({ eyebrow, title, description, actions }: {
  eyebrow: string; title: string; description: string; actions?: React.ReactNode;
}) {
  return <div className="page-intro">
    <div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
    {actions && <div className="page-actions">{actions}</div>}
  </div>;
}

export function Provenance({ compact = false }: { compact?: boolean }) {
  const { filters } = useFilters();
  const meta = useMeta(filters.mode);
  if (meta.isLoading) return <div className="provenance"><LoaderCircle className="spin" size={14} /> Đang đọc phiên bản dữ liệu…</div>;
  if (meta.isError || !meta.data) return <div className="provenance warning"><AlertCircle size={14} /> Không đọc được thông tin nguồn dữ liệu.</div>;
  const value = meta.data;
  return <div className={`provenance ${compact ? "provenance-compact" : ""}`} title={`Data version: ${value.data_version ?? "—"}\nConfig hash: ${value.config_hash ?? "—"}`}>
    <Database size={14} />
    <span>{filters.mode === "research" ? "Snapshot nghiên cứu" : "Dữ liệu rolling"}</span>
    <i /> <span>as-of {formatDate(value.as_of_utc)}</span>
    <code>data {shortHash(value.data_version)}</code>
    <code>config {shortHash(value.config_hash)}</code>
  </div>;
}

export function ApiState({ loading, error, empty, children }: {
  loading?: boolean; error?: unknown; empty?: boolean; children?: React.ReactNode;
}) {
  if (loading) return <div className="state-box"><LoaderCircle className="spin" size={17} /> Đang tải từ backend…</div>;
  if (error) return <div className="state-box state-error"><AlertCircle size={17} /><span>{error instanceof Error ? error.message : "Không thể tải dữ liệu từ API."}</span></div>;
  if (empty) return <div className="state-box">Chưa có kết quả cho lựa chọn hiện tại.</div>;
  return <>{children}</>;
}

export function StatTile({ label, value, detail, tone = "default" }: {
  label: string; value: React.ReactNode; detail?: React.ReactNode; tone?: "default" | "blue" | "red";
}) {
  return <article className={`stat-tile tone-${tone}`}><div className="stat-label">{label}</div><div className="stat-value">{value}</div>{detail && <div className="stat-detail">{detail}</div>}</article>;
}

export function SectionHeader({ title, note, children }: { title: string; note?: string; children?: React.ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2>{note && <p>{note}</p>}</div>{children}</div>;
}
