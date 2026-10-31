"use client";
import { useEffect, useMemo, useState } from "react";
import { Dialog } from "./dialog";
import { Icon } from "./icon";
import { ModelPicker } from "./model-picker";
import { SetupWizard } from "./setup-wizard";
import { TeacherData } from "./teacher-data";
import { Campaign, Health } from "@/lib/types";
import { api, ModelChoice, Pack, Strategy, TeacherJob } from "@/lib/packs";

const DESCRIPTIONS: Record<string, string> = {
  broad_sft: "Supervised LoRA on the permitted training data",
  targeted_sft: "Supervised LoRA that trains labeled production failures first",
  distill_sft: "Supervised LoRA on verified solutions written by a teacher",
  hard_sft:
    "Supervised LoRA focused on the training questions the model misses",
  grpo: "Reward-driven practice graded by Terse math worlds (8 steps)",
  dpo: "Preference alignment: chosen over rejected replies",
};

export function NewCampaign({
  onClose,
  onCreated,
  health,
}: {
  onClose: () => void;
  onCreated: (campaign: Campaign) => void;
  health: Health | null;
}) {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [packs, setPacks] = useState<Pack[]>([]),
    [strategies, setStrategies] = useState<Strategy[]>([]),
    [models, setModels] = useState<ModelChoice[]>([]),
    [packId, setPackId] = useState("gsm8k"),
    [modelId, setModelId] = useState(health?.model?.id ?? ""),
    [chosen, setChosen] = useState<string[]>([
      "broad_sft",
      "targeted_sft",
      "grpo",
    ]),
    [teacherJobs, setTeacherJobs] = useState<TeacherJob[]>([]),
    [synthetic, setSynthetic] = useState(""),
    [dialog, setDialog] = useState<"" | "data" | "teacher">("");

  useEffect(() => {
    Promise.all([
      api<Pack[]>("packs"),
      api<Strategy[]>("strategies"),
      api<ModelChoice[]>("models"),
    ])
      .then(([p, s, m]) => {
        setPacks(p);
        setStrategies(s);
        setModels(m);
      })
      .catch((e) => setError(e.message));
  }, []);
  const pack = packs.find((p) => p.id === packId);
  const backend = health?.compute?.backend ?? "local";
  const modal = backend === "modal";

  function loadTeacherJobs(id: string) {
    api<TeacherJob[]>(`packs/${id}/teacher-data`)
      .then((jobs) => {
        setTeacherJobs(jobs);
        const done = jobs.filter((j) => j.status === "completed");
        setSynthetic((current) =>
          done.some((j) => j.id === current)
            ? current
            : (done.at(-1)?.id ?? ""),
        );
      })
      .catch(() => setTeacherJobs([]));
  }
  useEffect(() => {
    if (pack) loadTeacherJobs(pack.id);
  }, [pack]);

  // Methods that can run for this task on this backend; the director chooses among them.
  const usable = useMemo(
    () =>
      strategies.filter((s) => {
        if (
          !pack ||
          !s.kinds.includes(pack.kind) ||
          !s.backends.includes(backend)
        )
          return false;
        if (s.id === "grpo") return pack.capabilities.grpo;
        if (s.id === "dpo") return pack.capabilities.dpo;
        return true;
      }),
    [strategies, pack, backend],
  );
  // The selection, limited to what this task and backend allow (never empty).
  const selected = useMemo(() => {
    const allowed = chosen.filter((id) => usable.some((s) => s.id === id));
    return allowed.length ? allowed : usable.slice(0, 2).map((s) => s.id);
  }, [chosen, usable]);
  const doneTeacher = teacherJobs.filter((j) => j.status === "completed");
  const needsTeacher = selected.includes("distill_sft") && !synthetic;

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!pack) return;
    setError("");
    setBusy(true);
    const f = new FormData(e.currentTarget);
    try {
      const data = await api<Campaign>("campaigns", {
        name: f.get("name"),
        objective: f.get("objective"),
        task_pack: pack.id,
        ...(modelId ? { model_id: modelId } : {}),
        max_seconds: Number(f.get("budget")) * 60,
        train_steps: Number(f.get("steps")),
        dev_samples: Number(f.get("samples")),
        final_samples: Number(f.get("final_samples") || f.get("samples")),
        max_experiments: Number(f.get("experiments")),
        final_backend: f.get("final_backend") || "shared_scorer",
        strategies: selected,
        ...(selected.includes("distill_sft")
          ? { synthetic_dataset: synthetic }
          : {}),
      });
      onCreated(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Campaign could not start.");
    } finally {
      setBusy(false);
    }
  }
  const ready =
    health?.services.terse.connected && health.services.relay.connected;
  const unit = pack?.display.unit ?? "examples";
  const model = models.find((m) => m.id === (modelId || health?.model?.id));
  return (
    <>
      <Dialog
        title="Train a model for a task"
        subtitle="Choose the task and its data, define success and a budget. Follow the decisions and evidence."
        onClose={onClose}
        wide
      >
        <form onSubmit={submit} className="campaign-form" key={pack?.id}>
          <label>
            Task
            <div className="field-row">
              <select
                value={packId}
                onChange={(e) => setPackId(e.target.value)}
                aria-label="Task"
              >
                {packs.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.display.name}
                    {p.builtin ? " · built in" : ` · ${p.kind}`}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="button small-button"
                onClick={() => setDialog("data")}
              >
                <Icon name="plus" size={14} />
                Add your data
              </button>
            </div>
            {pack && (
              <span className="field-help">
                Scored by {pack.display.scoring_label}.
                {pack.builtin
                  ? " Verified starter task."
                  : ` ${pack.counts.train ?? 0} training, ${pack.counts.dev ?? 0} development and ${pack.counts.sealed_remaining ?? 0} unused sealed ${unit}.`}
              </span>
            )}
          </label>
          <div className="form-model">
            <span className="model-monogram">
              {(model?.label ?? health?.model?.label ?? "Q")[0].toUpperCase()}
            </span>
            <div>
              <strong>
                {model?.id ?? health?.model?.id ?? "Qwen/Qwen2.5-0.5B-Instruct"}
              </strong>
              <span>
                {model?.parameters ?? health?.model?.parameters ?? "494M"}{" "}
                parameters · pinned revision
              </span>
            </div>
            <span className="tag">
              {modal ? `Modal ${health?.compute?.gpu}` : "Local GPU"}
            </span>
          </div>
          <ModelPicker
            models={models}
            value={modelId || health?.model?.id || ""}
            onChange={setModelId}
            onAdded={(added) => {
              setModels((all) => [
                ...all.filter((m) => m.id !== added.id),
                added,
              ]);
              if (!added.problem) setModelId(added.id);
            }}
          />
          {model?.problem && <p className="form-error">{model.problem}</p>}
          <label>
            Campaign name
            <input
              name="name"
              defaultValue={pack?.display.name ?? "Mathematical reasoning"}
              minLength={3}
              maxLength={80}
              required
              data-autofocus
            />
          </label>
          <label>
            Research objective
            <textarea
              name="objective"
              defaultValue={
                pack?.builtin
                  ? "Improve multi-step mathematical reasoning without benchmark leakage."
                  : pack?.display.objective
              }
              rows={3}
              minLength={10}
              maxLength={500}
              required
            />
            <span className="field-help">
              The objective is immutable once the campaign begins.
            </span>
          </label>
          <div className="form-grid">
            <label>
              Compute time limit
              <select name="budget" defaultValue="60">
                <option value="10">10 minutes</option>
                <option value="30">30 minutes</option>
                <option value="60">1 hour</option>
                <option value="120">2 hours</option>
              </select>
            </label>
            <label>
              Experiment budget
              <select name="experiments" defaultValue="2">
                <option value="1">1 experiment</option>
                <option value="2">2 experiments</option>
                <option value="3">3 experiments</option>
                <option value="4">4 experiments</option>
              </select>
            </label>
          </div>
          <div className="form-grid">
            <label>
              Optimization steps per experiment
              <select name="steps" defaultValue="96">
                <option value="24">24 steps</option>
                <option value="48">48 steps</option>
                <option value="96">96 steps</option>
                <option value="192">192 steps</option>
              </select>
            </label>
            <label>
              Development examples
              <select name="samples" defaultValue="24">
                <option value="8">8 {unit} · fast check</option>
                <option value="24">24 {unit} · research run</option>
                <option value="64">64 {unit} · stronger estimate</option>
                <option value="100">100 {unit}</option>
              </select>
            </label>
          </div>
          <label>
            Final evaluation examples
            <select
              name="final_samples"
              defaultValue={pack?.builtin ? "64" : "24"}
            >
              <option value="8">8 {unit} · mechanics check</option>
              <option value="24">24 {unit}</option>
              <option value="64">64 {unit} · independent evaluation</option>
              <option value="100">100 {unit}</option>
            </select>
          </label>
          <fieldset className="method-list">
            <legend>Allowed research methods</legend>
            {usable.map((s) => (
              <label key={s.id} className="method-option">
                <input
                  type="checkbox"
                  checked={selected.includes(s.id)}
                  onChange={(e) =>
                    setChosen(
                      e.target.checked
                        ? [...selected, s.id]
                        : selected.length > 1
                          ? selected.filter((x) => x !== s.id)
                          : selected,
                    )
                  }
                />
                <span>
                  <strong>{s.label}</strong>
                  <span className="muted small">{DESCRIPTIONS[s.id]}</span>
                </span>
              </label>
            ))}
            <span className="field-help">
              The director chooses among these using measured development
              evidence.
            </span>
          </fieldset>
          {selected.includes("distill_sft") && (
            <label>
              Teacher dataset
              <div className="field-row">
                <select
                  value={synthetic}
                  onChange={(e) => setSynthetic(e.target.value)}
                  aria-label="Teacher dataset"
                >
                  {!doneTeacher.length && (
                    <option value="">None prepared yet</option>
                  )}
                  {doneTeacher.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.id} · {j.teacher} · {j.counts?.kept ?? 0} verified
                      solutions
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="button small-button"
                  onClick={() => setDialog("teacher")}
                >
                  Prepare teacher data
                </button>
              </div>
              <span className="field-help">
                Only solutions the task&apos;s checker accepted are kept, and
                the campaign re-verifies them before training.
              </span>
            </label>
          )}
          <label>
            Final evaluation
            <select name="final_backend" defaultValue="shared_scorer">
              <option value="shared_scorer">
                Task checker · sealed {unit}, queried once
              </option>
              {pack?.capabilities.lm_eval_final && model?.independent_final && (
                <option value="lm_eval_azure">
                  lm-eval 0.4.13 · isolated Azure Linux worker
                </option>
              )}
            </select>
          </label>
          <div className="form-boundary">
            <Icon name="governance" />
            <div>
              <strong>Governance is required</strong>
              <p>
                Sealed test data, a pinned task and checker, provenance checks
                and a 25 percentage point regression limit. All checkpoint
                promotions require approval.
              </p>
            </div>
          </div>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          {!ready && (
            <p className="form-error">
              Terse and Agent Relay must be connected before launch.
            </p>
          )}
          <footer className="dialog-actions">
            <span className="muted small">
              {pack?.display.benchmark_label ?? "GSM8K subset"} · seed 42 ·{" "}
              {modal
                ? `Modal ${health?.compute?.gpu} billed per second`
                : "no cloud GPU billing"}
            </span>
            <button
              type="submit"
              className="button primary"
              disabled={
                busy || !ready || !pack || needsTeacher || !!model?.problem
              }
            >
              {busy ? "Initializing campaign…" : "Launch campaign"}
              <Icon name="arrow" size={16} />
            </button>
          </footer>
        </form>
      </Dialog>
      {dialog === "data" && (
        <SetupWizard
          onClose={() => setDialog("")}
          onRegistered={(registered) => {
            setPacks((p) => [
              ...p.filter((x) => x.id !== registered.id),
              registered,
            ]);
            setPackId(registered.id);
            setDialog("");
          }}
        />
      )}
      {dialog === "teacher" && pack && (
        <TeacherData
          pack={pack}
          onClose={() => {
            setDialog("");
            loadTeacherJobs(pack.id);
          }}
        />
      )}
    </>
  );
}
