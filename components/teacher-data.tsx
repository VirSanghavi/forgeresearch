"use client";
import { useEffect, useState } from "react";
import { Dialog } from "./dialog";
import { Icon } from "./icon";
import { api, Pack, TeacherJob } from "@/lib/packs";

type Probe = {
  endpoint: string;
  model_listed: boolean;
  digest: string | null;
  reply: string;
  reasoning_returned: boolean;
};

export function TeacherData({
  pack,
  onClose,
}: {
  pack: Pack;
  onClose: () => void;
}) {
  const [kind, setKind] = useState("ollama"),
    [url, setUrl] = useState("http://127.0.0.1:11434/v1"),
    [model, setModel] = useState("qwen3.5:4b"),
    [keyEnv, setKeyEnv] = useState(""),
    [terms, setTerms] = useState(""),
    [n, setN] = useState("4"),
    [limit, setLimit] = useState("64"),
    [probe, setProbe] = useState<Probe | null>(null),
    [job, setJob] = useState<TeacherJob | null>(null),
    [busy, setBusy] = useState(""),
    [error, setError] = useState("");
  const teacher = {
    kind,
    base_url: url,
    model,
    teacher_terms: terms,
    ...(keyEnv ? { api_key_env: keyEnv } : {}),
  };
  const local = /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])/.test(url);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = setInterval(() => {
      api<TeacherJob>(`packs/${pack.id}/teacher-data/${job.id}`)
        .then(setJob)
        .catch(() => undefined);
    }, 3000);
    return () => clearInterval(timer);
  }, [job, pack.id]);

  async function run(action: "probe" | "start") {
    setError("");
    setBusy(action === "probe" ? "Contacting the teacher…" : "Starting…");
    try {
      if (action === "probe")
        setProbe(
          await api<Probe>("distill/probe", {
            teacher: {
              ...teacher,
              teacher_terms: terms || "probe only, no data is generated",
            },
          }),
        );
      else
        setJob(
          await api<TeacherJob>(`packs/${pack.id}/teacher-data`, {
            teacher,
            sampling: { n: Number(n) },
            budget: { max_prompts: Number(limit) },
          }),
        );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed.");
    } finally {
      setBusy("");
    }
  }

  const progress = job?.progress;
  return (
    <Dialog
      title="Prepare teacher data"
      subtitle={`A stronger model writes solutions to ${pack.display.name} training questions. Only solutions the task's checker accepts are kept.`}
      onClose={onClose}
      wide
    >
      <div className="campaign-form">
        <div className="form-grid">
          <label>
            Teacher server
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="ollama">Ollama</option>
              <option value="vllm">vLLM</option>
              <option value="lmstudio">LM Studio</option>
              <option value="llama_cpp">llama.cpp server</option>
              <option value="openai_compatible">
                Other OpenAI-compatible API
              </option>
            </select>
          </label>
          <label>
            Model
            <input value={model} onChange={(e) => setModel(e.target.value)} />
          </label>
        </div>
        <label>
          Endpoint
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="http://127.0.0.1:11434/v1"
          />
          <span className="field-help">
            The OpenAI-compatible root ending in /v1.{" "}
            {local
              ? "A local teacher shares this machine's memory, so it only runs while no campaign does."
              : "Training questions are sent to this hosted endpoint."}
          </span>
        </label>
        {!local && (
          <label>
            API key environment variable
            <input
              value={keyEnv}
              onChange={(e) => setKeyEnv(e.target.value)}
              placeholder="TEACHER_API_KEY"
            />
            <span className="field-help">
              The name of a variable in the lab&apos;s environment. The key
              itself is never entered here or written to disk.
            </span>
          </label>
        )}
        <label>
          Why this teacher&apos;s outputs may train another model
          <textarea
            value={terms}
            onChange={(e) => setTerms(e.target.value)}
            rows={2}
            maxLength={2000}
            placeholder="For example: Qwen3.5-4B under Apache 2.0, run on our own hardware."
          />
          <span className="field-help">
            Recorded verbatim with the dataset. Some hosted APIs forbid training
            other models on their outputs; check the provider&apos;s terms.
          </span>
        </label>
        <div className="form-grid">
          <label>
            Attempts per question
            <select value={n} onChange={(e) => setN(e.target.value)}>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="4">4 · keep the shortest correct</option>
              <option value="8">8</option>
            </select>
          </label>
          <label>
            Questions to teach
            <select value={limit} onChange={(e) => setLimit(e.target.value)}>
              <option value="8">8 · quick check</option>
              <option value="64">64</option>
              <option value="384">All training questions</option>
            </select>
          </label>
        </div>
        {probe && (
          <p className="production-note">
            <Icon name={probe.model_listed ? "check" : "alert"} size={16} />
            <span>
              {probe.model_listed
                ? `${model} is served`
                : `${model} is not listed`}{" "}
              at {probe.endpoint}
              {probe.digest ? ` · digest ${probe.digest.slice(0, 12)}` : ""}
              {probe.reasoning_returned
                ? " · returns reasoning, which is stripped before checking"
                : ""}
            </span>
          </p>
        )}
        {job && (
          <section className="production-summary" aria-label="Teacher job">
            <div>
              <span>Status</span>
              <strong>{job.status.replace("_", " ")}</strong>
              <p className="mono">{job.id}</p>
            </div>
            <div>
              <span>Questions</span>
              <strong>
                {progress
                  ? `${progress.done}/${progress.total}`
                  : (job.counts?.prompts ?? "…")}
              </strong>
              <p>sent to the teacher</p>
            </div>
            <div>
              <span>Verified solutions</span>
              <strong className="accent-text">
                {job.counts?.kept ?? progress?.kept ?? 0}
              </strong>
              <p>accepted by the task&apos;s checker</p>
            </div>
          </section>
        )}
        {job?.status === "failed" && job.error && (
          <pre className="prediction-output">{job.error}</pre>
        )}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <footer className="dialog-actions">
          <span className="muted small">
            {busy ||
              "Development and sealed questions are never sent to a teacher."}
          </span>
          <div className="field-row">
            <button
              type="button"
              className="button"
              disabled={!!busy || !url || !model}
              onClick={() => run("probe")}
            >
              Test connection
            </button>
            {job?.status === "completed" ? (
              <button
                type="button"
                className="button primary"
                onClick={onClose}
              >
                Use this dataset
                <Icon name="arrow" size={16} />
              </button>
            ) : (
              <button
                type="button"
                className="button primary"
                disabled={
                  !!busy || terms.length < 20 || job?.status === "running"
                }
                onClick={() => run("start")}
              >
                {job?.status === "running" ? "Writing solutions…" : "Start"}
                <Icon name="play" size={16} />
              </button>
            )}
          </div>
        </footer>
      </div>
    </Dialog>
  );
}
