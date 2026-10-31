"use client";
import { FormEvent, useCallback, useEffect, useState } from "react";
import { Icon } from "./icon";
import { api, Connection, ConnectionList } from "@/lib/packs";

const STATUS: Record<string, { label: string; tone: string }> = {
  provisioning: { label: "Starting its VM", tone: "" },
  starting: { label: "Connecting", tone: "" },
  restarting: { label: "Restarting", tone: "" },
  syncing: { label: "Connected", tone: "success" },
  stopped: { label: "Stopped", tone: "" },
  capped: { label: "Daily cap reached", tone: "danger" },
  needs_mapping: { label: "Needs your input", tone: "danger" },
  source_auth_failed: { label: "Credential refused", tone: "danger" },
  source_error: { label: "Source unavailable", tone: "danger" },
  sink_auth_failed: { label: "Token refused", tone: "danger" },
  sink_unreachable: { label: "Buffer unavailable", tone: "danger" },
  failed: { label: "Stopped trying", tone: "danger" },
  disconnect_failed: { label: "Disconnect incomplete", tone: "danger" },
};
const EMPTY = {
  name: "",
  kind: "langfuse" as "langfuse" | "http",
  url: "",
  auth_type: "bearer",
  username: "",
  header: "",
  secret: "",
  cursor_param: "",
  start_cursor: "",
  records_path: "",
  data_terms: "",
};

function ConnectionRow({
  connection,
  busy,
  act,
}: {
  connection: Connection;
  busy: boolean;
  act: (path: string, body?: unknown) => void;
}) {
  const [confirming, setConfirming] = useState(false),
    [prompt, setPrompt] = useState(""),
    [output, setOutput] = useState("");
  const status = STATUS[connection.status] ?? {
    label: connection.status.replace(/_/g, " "),
    tone: "",
  };
  const base = `overseer/connections/${connection.id}`;
  const listId = `${connection.id}-fields`;
  return (
    <li className="connection">
      <div className="connection-head">
        <strong>{connection.name}</strong>
        <span className={`status ${status.tone}`}>
          <i />
          {status.label}
        </span>
        <span className="muted small mono">
          {connection.host} ·{" "}
          {connection.provider === "local" ? "this machine" : "its own VM"}
        </span>
      </div>
      <p className="muted small">
        {connection.counts.accepted} traces received
        {connection.counts.duplicates
          ? ` · ${connection.counts.duplicates} repeats ignored`
          : ""}
        {connection.counts.skipped
          ? ` · ${connection.counts.skipped} records without a prompt and answer`
          : ""}
        {connection.counts.rejected
          ? ` · ${connection.counts.rejected} refused`
          : ""}
      </p>
      {connection.detail && <p className="small">{connection.detail}</p>}
      {connection.status === "needs_mapping" && (
        <form
          className="field-row"
          onSubmit={(e) => {
            e.preventDefault();
            act(`${base}/mapping`, { mapping: { prompt, output } });
          }}
        >
          <datalist id={listId}>
            {connection.fields_seen.map((f) => (
              <option key={f} value={f} />
            ))}
          </datalist>
          <input
            aria-label="Field holding the prompt"
            placeholder="Field holding the prompt"
            list={listId}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <input
            aria-label="Field holding the answer"
            placeholder="Field holding the answer"
            list={listId}
            value={output}
            onChange={(e) => setOutput(e.target.value)}
          />
          <button
            className="button small-button"
            disabled={busy || !prompt || !output}
          >
            Use these fields
          </button>
        </form>
      )}
      <div className="field-row">
        {connection.status === "failed" && (
          <button
            className="button small-button"
            disabled={busy}
            onClick={() => act(`${base}/restart`, {})}
          >
            <Icon name="refresh" size={14} />
            Try again
          </button>
        )}
        <button
          className="button small-button"
          disabled={busy}
          onClick={() =>
            confirming ? act(`${base}/disconnect`, {}) : setConfirming(true)
          }
          onBlur={() => setConfirming(false)}
        >
          {confirming
            ? "Confirm: stop the agent and delete its credentials"
            : "Disconnect"}
        </button>
      </div>
    </li>
  );
}

export function ConnectionsPanel({ packId }: { packId: string }) {
  const [list, setList] = useState<ConnectionList | null>(null),
    [form, setForm] = useState(EMPTY),
    [open, setOpen] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setList(
        await api<ConnectionList>(`overseer/connections?task_pack=${packId}`),
      );
    } catch {
      // The page's own banner already reports an unavailable lab.
    }
  }, [packId]);
  useEffect(() => {
    const first = setTimeout(load, 0);
    const timer = setInterval(load, 15000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [load]);

  async function run(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
      await load();
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed.");
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    const langfuse = form.kind === "langfuse";
    const done = await run(() =>
      api("overseer/connections", {
        name: form.name,
        task_pack: packId,
        kind: form.kind,
        data_terms: form.data_terms,
        secret: form.secret || null,
        url: form.url || null,
        username: form.username || null,
        auth_type: langfuse ? "basic" : form.auth_type,
        header: form.auth_type === "api-key" ? form.header || null : null,
        cursor_param: form.cursor_param || null,
        start_cursor: form.start_cursor || null,
        records_path: form.records_path || null,
      }),
    );
    if (done) {
      setForm(EMPTY);
      setOpen(false);
    }
  }

  const set =
    (key: keyof typeof EMPTY) =>
    (
      e: React.ChangeEvent<
        HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
      >,
    ) =>
      setForm((f) => ({ ...f, [key]: e.target.value }));
  const vm = list?.providers.superserve;
  const langfuse = form.kind === "langfuse";
  const needsSecret = langfuse || form.auth_type !== "none";
  return (
    <section className="panel connections-panel" aria-label="Connections">
      <header className="overseer-queue-head">
        <div>
          <h3>Connected traffic</h3>
          <p className="muted small">
            Connect a source once. The overseer&apos;s agent reads new traffic
            on its own, so nothing has to be uploaded by hand.
          </p>
        </div>
        <button
          className="button small-button"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Cancel" : "Connect traffic"}
        </button>
      </header>
      {error && <p className="error-banner">{error}</p>}
      {open && (
        <form className="campaign-form" onSubmit={submit}>
          <label>
            Name
            <input
              required
              maxLength={80}
              placeholder="Support bot"
              value={form.name}
              onChange={set("name")}
            />
          </label>
          <label>
            Where your traffic is logged
            <select value={form.kind} onChange={set("kind")}>
              <option value="langfuse">Langfuse</option>
              <option value="http">Any JSON API or log endpoint</option>
            </select>
          </label>
          <label>
            {langfuse ? "Langfuse host (optional)" : "URL"}
            <input
              required={!langfuse}
              placeholder={
                langfuse
                  ? "https://cloud.langfuse.com"
                  : "https://logs.example.com/traces"
              }
              value={form.url}
              onChange={set("url")}
            />
          </label>
          {!langfuse && (
            <>
              <label>
                How it signs in
                <select value={form.auth_type} onChange={set("auth_type")}>
                  <option value="bearer">Bearer token</option>
                  <option value="api-key">API key header</option>
                  <option value="basic">Username and password</option>
                  <option value="none">No sign-in</option>
                </select>
              </label>
              {form.auth_type === "api-key" && (
                <label>
                  Header name
                  <input
                    required
                    placeholder="X-Api-Key"
                    value={form.header}
                    onChange={set("header")}
                  />
                </label>
              )}
              <label>
                Filter for newer records (optional)
                <input
                  placeholder="since"
                  value={form.cursor_param}
                  onChange={set("cursor_param")}
                />
              </label>
              <label>
                Start from (optional)
                <input
                  placeholder="0, or an ISO 8601 time"
                  value={form.start_cursor}
                  onChange={set("start_cursor")}
                />
              </label>
            </>
          )}
          {(langfuse || form.auth_type === "basic") && (
            <label>
              {langfuse ? "Public key" : "Username"}
              <input
                required
                autoComplete="off"
                value={form.username}
                onChange={set("username")}
              />
            </label>
          )}
          {needsSecret && (
            <label>
              {langfuse ? "Secret key" : "Read-only credential"}
              <input
                required
                type="password"
                autoComplete="new-password"
                value={form.secret}
                onChange={set("secret")}
              />
            </label>
          )}
          <label>
            What gives you the right to send this traffic here?
            <textarea
              required
              minLength={20}
              rows={2}
              placeholder="Our own product's traffic; our terms allow us to use it to improve our models."
              value={form.data_terms}
              onChange={set("data_terms")}
            />
          </label>
          <p className="muted small">
            {vm?.ready
              ? "The agent gets its own VM, which can reach only this source and the Forge buffer. It never holds the real credential. The VM is billed by SuperServe while it runs."
              : `The agent will run on this machine and stop reaching new traffic when it sleeps. ${vm?.reason ?? ""}.`}{" "}
            Give it a read-only credential.
          </p>
          <button className="button primary" disabled={busy}>
            {busy ? "Connecting…" : "Connect"}
            <Icon name="arrow" size={16} />
          </button>
        </form>
      )}
      {list && !list.connections.length && !open && (
        <p className="muted">No traffic connected for this task yet.</p>
      )}
      {!!list?.connections.length && (
        <ul className="connection-list">
          {list.connections.map((c) => (
            <ConnectionRow
              key={c.id}
              connection={c}
              busy={busy}
              act={(path, body) => run(() => api(path, body))}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
