import { useEffect, useMemo, useState } from "react";
import { useFilters } from "../app/filters";
import { modelFamily, modelLabel, scoresForSelection } from "../api/modelSelection";
import type { ModelSelection } from "../api/types";

const groups = [
  { id: "auto", label: "Tự động", note: "Khuyến nghị theo QLIKE" },
  { id: "simple", label: "Đơn giản", note: "Theo hôm qua, trung bình, EWMA" },
  { id: "classical", label: "Thống kê kinh điển", note: "HAR và GARCH" },
  { id: "machine-learning", label: "Học máy", note: "XGBoost, LightGBM, rừng ngẫu nhiên" },
] as const;
type GroupId = typeof groups[number]["id"];

export function ModelSelector({ compact = false }: { compact?: boolean }) {
  const { filters, update, models, modelTable, resolvedModel, recommendedModel } = useFilters();
  const [activeGroup, setActiveGroup] = useState<GroupId>(filters.model === "auto" ? "auto" : modelFamily(filters.model));
  const scores = useMemo(() => scoresForSelection(modelTable, filters.symbol, filters.horizon), [modelTable, filters.symbol, filters.horizon]);
  const activeModel = filters.model === "auto" ? recommendedModel : filters.model;

  useEffect(() => {
    setActiveGroup(filters.model === "auto" ? "auto" : modelFamily(filters.model));
  }, [filters.model]);

  const options = activeGroup === "auto" ? [] : models.filter((model) => modelFamily(model) === activeGroup);
  const selectGroup = (group: GroupId) => {
    setActiveGroup(group);
    if (group === "auto") {
      update({ model: "auto" });
      return;
    }
    if (filters.model === "auto" || modelFamily(filters.model) !== group) {
      const first = scores.find((score) => modelFamily(score.model) === group)?.model ?? models.find((model) => modelFamily(model) === group);
      if (first) update({ model: first });
    }
  };

  return <section className={`model-selector ${compact ? "model-selector-compact" : ""}`} aria-label="Chọn mô hình">
    <div className="model-family-tabs" role="tablist" aria-label="Nhóm mô hình">
      {groups.map((group) => <button
        key={group.id}
        type="button"
        role="tab"
        aria-selected={activeGroup === group.id}
        className={activeGroup === group.id ? "model-family-active" : ""}
        onClick={() => selectGroup(group.id)}
      >
        <strong>{group.label}</strong><small>{group.note}</small>
      </button>)}
    </div>
    {activeGroup === "auto" ? <div className="model-recommendation" aria-live="polite">
      <span>{recommendedModel ? `Tự động chọn: ${modelLabel(recommendedModel)} (khuyến nghị)` : "Đang đọc bảng T3 để chọn mô hình…"}</span>
      {scores[0] && <code>QLIKE {scores[0].qlike.toFixed(4)}</code>}
    </div> : <div className="model-choice-row">
      <label htmlFor="expert-model-choice">Mô hình trong nhóm</label>
      <select id="expert-model-choice" value={activeModel && options.includes(activeModel) ? activeModel : ""} onChange={(event) => update({ model: event.target.value as ModelSelection })}>
        {!options.length && <option value="">Chưa có mô hình trong bảng T3</option>}
        {options.map((model) => {
          const score = scores.find((entry) => entry.model === model);
          return <option key={model} value={model}>{modelLabel(model)}{score ? ` · QLIKE ${score.qlike.toFixed(4)}` : ""}</option>;
        })}
      </select>
      <span className="model-score-note">QLIKE thấp hơn là tốt hơn. Điểm hiển thị theo T3, cùng mã và kỳ hạn.</span>
    </div>}
    {resolvedModel && filters.model === "auto" && <span className="sr-only">Mô hình được dùng: {modelLabel(resolvedModel)}</span>}
  </section>;
}
