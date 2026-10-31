"use client";
import { useMemo, useState } from "react";
import { api, ModelChoice } from "@/lib/packs";

// Smallest first within a family, using the same units the API reports.
const size = (m: ModelChoice) =>
  parseFloat(m.parameters) * (m.parameters.endsWith("B") ? 1000 : 1);

export function ModelPicker({
  models,
  value,
  onChange,
  onAdded,
}: {
  models: ModelChoice[];
  value: string;
  onChange: (id: string) => void;
  onAdded: (model: ModelChoice) => void;
}) {
  const [query, setQuery] = useState(""),
    [repo, setRepo] = useState(""),
    [adding, setAdding] = useState(false),
    [error, setError] = useState("");

  const shown = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    return models
      .filter((m) => {
        const text =
          `${m.id} ${m.label} ${m.architecture ?? ""} ${m.license ?? ""} ${m.parameters}`.toLowerCase();
        return words.every((word) => text.includes(word));
      })
      .sort(
        (a, b) =>
          Number(!!a.problem) - Number(!!b.problem) ||
          a.family.localeCompare(b.family) ||
          size(a) - size(b),
      );
  }, [models, query]);

  async function add() {
    setError("");
    setAdding(true);
    try {
      const model = await api<ModelChoice>("models", { repo });
      onAdded(model);
      setRepo("");
      setQuery("");
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "That model could not be added.",
      );
    } finally {
      setAdding(false);
    }
  }

  return (
    <div className="model-picker">
      <span className="model-picker-title">
        Model · {models.length} available
      </span>
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search by name, maker, size or license"
        aria-label="Search models"
      />
      <div className="model-list" role="radiogroup" aria-label="Model">
        {shown.map((m) => (
          <label
            key={m.id}
            className={`model-row${m.problem ? " is-blocked" : ""}`}
            title={m.problem ?? m.id}
          >
            <input
              type="radio"
              name="model"
              checked={m.id === value}
              disabled={!!m.problem}
              onChange={() => onChange(m.id)}
            />
            <span className="model-row-name">
              <strong>{m.label}</strong>
              <span className="muted small">
                {m.family} · {m.license ?? "see model card"}
              </span>
            </span>
            <span className="model-row-size">{m.parameters}</span>
            {m.problem && (
              <span className="tag">
                {m.remote_only ? "Cloud GPU" : "Needs access"}
              </span>
            )}
          </label>
        ))}
        {!shown.length && (
          <p className="muted small model-empty">
            Nothing matches. Add it from Hugging Face below.
          </p>
        )}
      </div>
      <div className="field-row">
        <input
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              if (repo.trim() && !adding) add();
            }
          }}
          placeholder="Any Hugging Face model: owner/name"
          aria-label="Hugging Face model"
        />
        <button
          type="button"
          className="button small-button"
          disabled={!repo.trim() || adding}
          onClick={add}
        >
          {adding ? "Checking…" : "Add model"}
        </button>
      </div>
      {error ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : (
        <span className="field-help">
          Forge pins the current commit and reads the training setup from the
          model&apos;s architecture. Weights download when a campaign starts.
        </span>
      )}
    </div>
  );
}
