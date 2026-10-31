"use client";
import { useMemo, useState } from "react";
import { Dialog } from "./dialog";
import { Icon } from "./icon";
import {
  api,
  columns,
  firstValues,
  INLINE_LIMIT,
  Pack,
  PackKind,
  PackPreview,
  upload,
} from "@/lib/packs";

type CheckerChoice = "number" | "marker_text" | "exact" | "json";
const CHECKERS: Record<CheckerChoice, { label: string; help: string }> = {
  number: {
    label: "Number on a final #### line",
    help: "The answer ends with '#### 42'; numbers are compared exactly.",
  },
  marker_text: {
    label: "Text after a final marker",
    help: "The answer ends with a line like 'Answer: Paris'; compared ignoring case and spacing.",
  },
  exact: {
    label: "Whole answer, exact match",
    help: "The whole reply must equal the reference, ignoring case and spacing.",
  },
  json: {
    label: "JSON with expected fields",
    help: "The reply is a JSON object; every field in the reference must match.",
  },
};
const INSTRUCTIONS: Record<CheckerChoice, string> = {
  number:
    "Answer the question. End with a final line in the form #### <number>.",
  marker_text:
    "Answer the question. End with a final line in the form Answer: <answer>.",
  exact: "Reply with the answer only.",
  json: "Reply with a JSON object containing the requested fields only.",
};

function checkerSpec(choice: CheckerChoice, reference: string[]) {
  if (choice === "number")
    return {
      extract: { kind: "final_marker", marker: "####", grammar: "number" },
      compare: { kind: "numeric" },
    };
  if (choice === "marker_text")
    return {
      extract: { kind: "final_marker", marker: "Answer:" },
      compare: {
        kind: "exact",
        normalize: ["strip", "casefold", "whitespace"],
      },
    };
  if (choice === "exact")
    return {
      extract: { kind: "whole" },
      compare: {
        kind: "exact",
        normalize: ["strip", "casefold", "whitespace"],
      },
      reference: "whole",
    };
  let fields: { path: string }[] = [];
  try {
    fields = Object.keys(JSON.parse(reference[0] ?? "{}")).map((k) => ({
      path: `/${k.replace(/~/g, "~0").replace(/\//g, "~1")}`,
    }));
  } catch {
    fields = [];
  }
  return {
    extract: { kind: "json" },
    compare: {
      kind: "json_fields",
      fields,
      json_schema: {
        type: "object",
        required: fields.map((f) => f.path.slice(1)),
      },
    },
    reference: "whole",
  };
}

function guess(values: string[]): CheckerChoice {
  const v = values.filter(Boolean);
  if (v.length && v.every((x) => x.trim().startsWith("{"))) return "json";
  if (v.length && v.every((x) => /####\s*[-\d.,]+\s*$/.test(x)))
    return "number";
  if (v.length && v.every((x) => /^answer:/im.test(x))) return "marker_text";
  return "exact";
}

export function SetupWizard({
  onClose,
  onRegistered,
}: {
  onClose: () => void;
  onRegistered: (pack: Pack) => void;
}) {
  const [step, setStep] = useState(1),
    [file, setFile] = useState<{ name: string; text: string } | null>(null),
    [format, setFormat] = useState<"csv" | "jsonl">("jsonl"),
    [kind, setKind] = useState<PackKind>("qa"),
    [question, setQuestion] = useState(""),
    [reference, setReference] = useState(""),
    [chosen, setChosen] = useState(""),
    [rejected, setRejected] = useState(""),
    [idColumn, setIdColumn] = useState(""),
    [checker, setChecker] = useState<CheckerChoice>("exact"),
    [instruction, setInstruction] = useState(INSTRUCTIONS.exact),
    [name, setName] = useState(""),
    [objective, setObjective] = useState(""),
    [license, setLicense] = useState("Proprietary"),
    [terms, setTerms] = useState(""),
    [preview, setPreview] = useState<PackPreview | null>(null),
    [uploadId, setUploadId] = useState<string | null>(null),
    [busy, setBusy] = useState(""),
    [error, setError] = useState("");
  const cols = useMemo(
    () => (file ? columns(file.text, format) : []),
    [file, format],
  );

  async function choose(f: File | undefined) {
    if (!f) return;
    const text = await f.text();
    const fmt = f.name.toLowerCase().endsWith(".csv") ? "csv" : "jsonl";
    const found = columns(text, fmt);
    const pick = (...names: string[]) =>
      found.find((c) => names.includes(c.toLowerCase())) ?? "";
    const q =
      pick("question", "prompt", "input", "text", "instruction") ||
      found[0] ||
      "";
    const a = pick(
      "answer",
      "output",
      "reference",
      "label",
      "target",
      "response",
    );
    const c = pick("chosen", "preferred", "good");
    const r = pick("rejected", "dispreferred", "bad");
    setFile({ name: f.name, text });
    setFormat(fmt);
    setQuestion(q);
    setReference(a);
    setChosen(c);
    setRejected(r);
    setIdColumn(pick("id", "uid", "ticket_id"));
    const k: PackKind =
      c && r
        ? "preference"
        : a && guess(firstValues(text, fmt, a)) === "json"
          ? "extraction"
          : "qa";
    setKind(k);
    const g = a ? guess(firstValues(text, fmt, a)) : "exact";
    setChecker(g);
    setInstruction(k === "preference" ? "Reply to the user." : INSTRUCTIONS[g]);
    setName(
      f.name
        .replace(/\.[^.]+$/, "")
        .replace(/[-_]+/g, " ")
        .slice(0, 80),
    );
    setUploadId(null);
    setPreview(null);
    setError("");
  }

  function draft() {
    const values = reference ? firstValues(file!.text, format, reference) : [];
    return {
      name,
      objective,
      kind,
      format,
      mapping:
        kind === "preference"
          ? {
              ...(idColumn ? { id: idColumn } : {}),
              input_template: `{${question}}`,
              target_template: null,
              chosen: `{${chosen}}`,
              rejected: `{${rejected}}`,
            }
          : {
              ...(idColumn ? { id: idColumn } : {}),
              input_template: `{${question}}`,
              target_template: `{${reference}}`,
            },
      instruction,
      ...(kind === "preference"
        ? {}
        : { checker: checkerSpec(checker, values) }),
      license,
      data_terms: terms,
    };
  }

  async function source() {
    if (file!.text.length <= INLINE_LIMIT) return { text: file!.text };
    if (uploadId) return { upload_id: uploadId };
    setBusy("Uploading 0%");
    const id = await upload(file!.name, file!.text, (share) =>
      setBusy(`Uploading ${Math.round(share * 100)}%`),
    );
    setUploadId(id);
    return { upload_id: id };
  }

  async function check() {
    setError("");
    try {
      const body = { draft: draft(), ...(await source()) };
      setBusy("Validating every row…");
      setPreview(await api<PackPreview>("packs/preview", body));
      setStep(3);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Validation failed.");
    } finally {
      setBusy("");
    }
  }

  async function register() {
    setError("");
    try {
      const body = { draft: draft(), ...(await source()) };
      setBusy("Registering task…");
      onRegistered(await api<Pack>("packs", body));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Registration failed.");
    } finally {
      setBusy("");
    }
  }

  const mapped =
    question && (kind === "preference" ? chosen && rejected : reference);
  const described =
    name.length >= 3 && objective.length >= 10 && terms.length >= 20 && license;
  return (
    <Dialog
      title="Add your data"
      subtitle="Upload examples of the task. Forge validates every row, keeps a sealed test set aside and registers an immutable task."
      onClose={onClose}
      wide
    >
      <div className="campaign-form">
        <ol className="wizard-steps" aria-label="Steps">
          {["File and fields", "Task details", "Check and register"].map(
            (label, i) => (
              <li
                key={label}
                className={
                  step === i + 1 ? "current" : step > i + 1 ? "done" : ""
                }
                aria-current={step === i + 1 ? "step" : undefined}
              >
                <span>{i + 1}</span>
                {label}
              </li>
            ),
          )}
        </ol>
        {step === 1 && (
          <>
            <label>
              Data file
              <input
                type="file"
                accept=".jsonl,.csv,.json,application/jsonl,text/csv"
                onChange={(e) => choose(e.target.files?.[0])}
              />
              <span className="field-help">
                JSONL (one object per line) or CSV with a header row. Each row
                is one example.
              </span>
            </label>
            {file && (
              <>
                <p className="muted small">
                  {file.name} · {format.toUpperCase()} ·{" "}
                  {cols.length
                    ? `${cols.length} columns found`
                    : "no columns found"}
                </p>
                <label>
                  What kind of task is this?
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value as PackKind)}
                  >
                    <option value="qa">Questions with reference answers</option>
                    <option value="extraction">
                      Structured extraction (JSON)
                    </option>
                    <option value="preference">
                      Preferred and rejected replies (alignment)
                    </option>
                  </select>
                </label>
                <div className="form-grid">
                  <label>
                    Question or prompt column
                    <select
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                    >
                      {cols.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  </label>
                  {kind === "preference" ? (
                    <>
                      <label>
                        Preferred reply column
                        <select
                          value={chosen}
                          onChange={(e) => setChosen(e.target.value)}
                        >
                          <option value="">Choose…</option>
                          {cols.map((c) => (
                            <option key={c}>{c}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Rejected reply column
                        <select
                          value={rejected}
                          onChange={(e) => setRejected(e.target.value)}
                        >
                          <option value="">Choose…</option>
                          {cols.map((c) => (
                            <option key={c}>{c}</option>
                          ))}
                        </select>
                      </label>
                    </>
                  ) : (
                    <label>
                      Reference answer column
                      <select
                        value={reference}
                        onChange={(e) => {
                          setReference(e.target.value);
                          const g = guess(
                            firstValues(file.text, format, e.target.value),
                          );
                          setChecker(g);
                          setInstruction(INSTRUCTIONS[g]);
                        }}
                      >
                        <option value="">Choose…</option>
                        {cols.map((c) => (
                          <option key={c}>{c}</option>
                        ))}
                      </select>
                    </label>
                  )}
                  <label>
                    Row id column (optional)
                    <select
                      value={idColumn}
                      onChange={(e) => setIdColumn(e.target.value)}
                    >
                      <option value="">Generate ids</option>
                      {cols.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  </label>
                </div>
                {kind !== "preference" && (
                  <label>
                    How is an answer checked?
                    <select
                      value={checker}
                      onChange={(e) => {
                        const c = e.target.value as CheckerChoice;
                        setChecker(c);
                        setInstruction(INSTRUCTIONS[c]);
                      }}
                    >
                      {(Object.keys(CHECKERS) as CheckerChoice[]).map((c) => (
                        <option key={c} value={c}>
                          {CHECKERS[c].label}
                        </option>
                      ))}
                    </select>
                    <span className="field-help">
                      {CHECKERS[checker].help} Every reference must pass its own
                      checker, so mapping mistakes show up in the check.
                    </span>
                  </label>
                )}
              </>
            )}
          </>
        )}
        {step === 2 && (
          <>
            <label>
              Task name
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                minLength={3}
                maxLength={80}
              />
            </label>
            <label>
              What should improve?
              <textarea
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                rows={2}
                maxLength={500}
                placeholder="For example: answer support questions with the correct refund amount."
              />
            </label>
            <label>
              Instruction shown to the model
              <textarea
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                rows={2}
                maxLength={4000}
              />
              <span className="field-help">
                Placed before every question, in training and evaluation alike.
              </span>
            </label>
            <div className="form-grid">
              <label>
                Data license
                <input
                  value={license}
                  onChange={(e) => setLicense(e.target.value)}
                  maxLength={200}
                />
              </label>
            </div>
            <label>
              Permission to train on this data
              <textarea
                value={terms}
                onChange={(e) => setTerms(e.target.value)}
                rows={2}
                maxLength={2000}
                placeholder="For example: our own anonymized support tickets; we may use them to train internal models."
              />
              <span className="field-help">
                Recorded with the task and shown in every report. At least 20
                characters.
              </span>
            </label>
          </>
        )}
        {step === 3 && preview && (
          <>
            <section className="production-summary" aria-label="Validation">
              <div>
                <span>Valid rows</span>
                <strong>{preview.valid}</strong>
                <p>of {preview.rows} read</p>
              </div>
              <div>
                <span>Training</span>
                <strong>{preview.split.train}</strong>
                <p>used to train</p>
              </div>
              <div>
                <span>Development</span>
                <strong>{preview.split.dev}</strong>
                <p>used to choose experiments</p>
              </div>
              <div>
                <span>Sealed test</span>
                <strong>{preview.split.sealed}</strong>
                <p>queried once per campaign, never reused</p>
              </div>
            </section>
            {(preview.invalid > 0 || preview.duplicates > 0) && (
              <details className="production-skipped">
                <summary>
                  {preview.invalid} invalid and {preview.duplicates} duplicate
                  rows left out
                  <Icon name="down" size={14} />
                </summary>
                <ul>
                  {preview.invalid_examples.map((r) => (
                    <li key={r.line}>
                      <span className="mono">Line {r.line}</span>
                      <span>{r.reason}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {preview.problems.map((p) => (
              <p key={p} className="form-error" role="alert">
                {p}
              </p>
            ))}
            <div className="panel wizard-samples">
              <span className="muted small">
                Training samples as the model will see them
              </span>
              {preview.samples.map((s) => (
                <div key={s.id} className="finding">
                  <p className="finding-prompt">{s.question}</p>
                  <pre className="prediction-output finding-output">
                    {s.answer}
                  </pre>
                </div>
              ))}
            </div>
            <p className="muted small mono">Checker {preview.checker}</p>
          </>
        )}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <footer className="dialog-actions">
          <span className="muted small">
            {busy || (file ? file.name : "No file chosen")}
          </span>
          <div className="field-row">
            {step > 1 && (
              <button
                type="button"
                className="button"
                onClick={() => setStep(step - 1)}
                disabled={!!busy}
              >
                Back
              </button>
            )}
            {step === 1 && (
              <button
                type="button"
                className="button primary"
                disabled={!mapped}
                onClick={() => setStep(2)}
              >
                Next
                <Icon name="arrow" size={16} />
              </button>
            )}
            {step === 2 && (
              <button
                type="button"
                className="button primary"
                disabled={!described || !!busy}
                onClick={check}
              >
                Check every row
                <Icon name="check" size={16} />
              </button>
            )}
            {step === 3 && (
              <button
                type="button"
                className="button primary"
                disabled={!!busy || !!preview?.problems.length}
                onClick={register}
              >
                Register task
                <Icon name="arrow" size={16} />
              </button>
            )}
          </div>
        </footer>
      </div>
    </Dialog>
  );
}
