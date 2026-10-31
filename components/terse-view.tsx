"use client";
import { useCallback, useEffect, useState } from "react";
import { Campaign, Environment, agentNames, duration } from "@/lib/types";
import { EnvironmentsView, Empty, PanelHeader, Status } from "./views";
import { Icon } from "./icon";
import { CrashTest } from "./crash-test";

type ActorState = {
  read_at: string;
  runtime: string;
  campaign: {
    version: number;
    status: string;
    selection_frozen: boolean;
    final_claimed: boolean;
    consumed_seconds: number;
    config: { objective: string; max_seconds: number };
    reservations: Record<
      string,
      { status: string; seconds: number; actual_seconds?: number }
    >;
    decisions: { version: number; status: string; reason: string }[];
  };
  governance: {
    reviews: Record<string, unknown>;
    promoted: string;
    policy: { reward_version: string };
  };
  registry: Record<string, unknown>;
  experiment: { status: string; metrics: unknown[] } | null;
  experiment_id: string | null;
};

export function TerseView({
  campaign,
  onSelect,
}: {
  campaign: Campaign | null;
  onSelect: (e: Environment) => void;
}) {
  const [state, setState] = useState<ActorState | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const campaignId = campaign?.id;
  const reread = useCallback(() => setAttempt((n) => n + 1), []);
  useEffect(() => {
    let active = true;
    async function read() {
      if (!campaignId) return;
      try {
        const response = await fetch(
          `/api/lab/campaigns/${campaignId}/actors`,
          { cache: "no-store" },
        );
        const data = await response.json();
        if (!response.ok)
          throw new Error(data.detail || "Could not read Terse state.");
        if (active) {
          setState(data);
          setError("");
        }
      } catch (e) {
        if (active)
          setError(
            e instanceof Error ? e.message : "Could not read Terse state.",
          );
      }
    }
    void read();
    const timer = setInterval(() => void read(), 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [campaignId, attempt]); // Read canonical actor state, independently of the UI projection.
  if (!campaign)
    return (
      <Empty
        icon="environments"
        title="Durable state begins with a campaign"
        description="Launch research to inspect its actor state and versioned training worlds."
      />
    );
  if (error && !state)
    return (
      <div className="error-banner" role="alert">
        <p>{error}</p>
        <button
          className="button secondary"
          onClick={() => setAttempt((n) => n + 1)}
        >
          Retry state read
        </button>
      </div>
    );
  if (!state)
    return (
      <section className="panel actor-loading" aria-busy="true">
        <h2>Reading Terse actors</h2>
        <p>
          Fetching campaign, experiment, governance and artifact registry state.
        </p>
        <div />
        <div />
        <div />
      </section>
    );
  const c = state.campaign;
  return (
    <div className="terse-view">
      <section className="terse-intro">
        <div>
          <span className="eyebrow">Terse durable actors</span>
          <h2>The memory behind the research.</h2>
          <p>
            Coordination lives in persisted actors. Optimization runs on the
            worker. These values are read directly from the local native
            runtime.
          </p>
        </div>
        <span className="tag">
          <Icon name="environments" size={15} />
          State version {c.version}
        </span>
      </section>
      <CrashTest campaignId={campaign.id} onRecovered={reread} />
      {error && (
        <p className="actor-stale" role="status">
          Runtime unavailable. Showing the last committed read, from{" "}
          {new Date(state.read_at).toLocaleTimeString("en-GB", {
            timeZone: "UTC",
          })}{" "}
          UTC.
        </p>
      )}
      <div className="actor-grid">
        <section className="actor-card">
          <span className="actor-class mono">ResearchCampaignActor</span>
          <h3>Objective & compute</h3>
          <p>{c.config.objective}</p>
          <div className="actor-fact">
            <span>Settled compute</span>
            <strong>
              {duration(c.consumed_seconds)} / {duration(c.config.max_seconds)}
            </strong>
          </div>
          <div className="actor-fact">
            <span>Checkpoint selection</span>
            <strong>{c.selection_frozen ? "Frozen" : "Open"}</strong>
          </div>
          <div className="actor-fact">
            <span>Final test claim</span>
            <strong>{c.final_claimed ? "Claimed once" : "Unclaimed"}</strong>
          </div>
        </section>
        <section className="actor-card">
          <span className="actor-class mono">ExperimentActor</span>
          <h3>Optimizer history</h3>
          <p className="mono">
            {state.experiment_id || "No experiment created yet"}
          </p>
          <div className="actor-fact">
            <span>Persisted metrics</span>
            <strong>{state.experiment?.metrics.length || 0}</strong>
          </div>
          <div className="actor-fact">
            <span>Experiment state</span>
            <strong>{state.experiment?.status || "Not started"}</strong>
          </div>
        </section>
        <section className="actor-card">
          <span className="actor-class mono">GovernanceActor</span>
          <h3>Review & promotion</h3>
          <p>Evidence-bound decisions gate checkpoint promotion.</p>
          <div className="actor-fact">
            <span>Recorded reviews</span>
            <strong>{Object.keys(state.governance.reviews).length}</strong>
          </div>
          <div className="actor-fact">
            <span>Promoted checkpoint</span>
            <strong>{state.governance.promoted || "None promoted"}</strong>
          </div>
        </section>
        <section className="actor-card">
          <span className="actor-class mono">ArtifactRegistryActor</span>
          <h3>Immutable evidence</h3>
          <p>
            Files are registered with a SHA-256 identity. Conflicting
            registrations are refused.
          </p>
          <div className="actor-fact">
            <span>Registered artifacts</span>
            <strong>{Object.keys(state.registry).length}</strong>
          </div>
          <div className="actor-fact">
            <span>Reward policy</span>
            <strong>{state.governance.policy.reward_version}</strong>
          </div>
        </section>
      </div>
      {!!campaign.workbenches?.length && (
        <section className="panel">
          <PanelHeader
            title="Freestyle VMs"
            icon="code"
            subtitle="Each research role checks evidence with code in its own VM. Every VM is deleted when the campaign ends."
          />
          <div className="table-scroll">
            <table className="workbench-table">
              <thead>
                <tr>
                  <th scope="col">Role</th>
                  <th scope="col">VM</th>
                  <th scope="col" className="numeric">
                    Runs
                  </th>
                  <th scope="col" className="numeric">
                    Run time
                  </th>
                  <th scope="col">After the campaign</th>
                </tr>
              </thead>
              <tbody>
                {campaign.workbenches.map((w) => (
                  <tr key={w.id}>
                    <td>{agentNames[w.role] ?? w.role}</td>
                    <td className="mono">{w.id}</td>
                    <td className="numeric">{w.runs}</td>
                    <td className="numeric">{duration(w.seconds)}</td>
                    <td>
                      {w.deleted ? (
                        "Deleted"
                      ) : (
                        <span className="warning-text">
                          Not deleted yet; it deletes itself at its time limit
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <section className="panel reservation-panel">
        <div className="panel-header">
          <h2>Durable compute reservations</h2>
          <span className="muted small">
            Actor read at{" "}
            {new Date(state.read_at).toLocaleTimeString("en-GB", {
              timeZone: "UTC",
            })}{" "}
            UTC
          </span>
        </div>
        {Object.keys(c.reservations).length ? (
          <div className="reservation-list">
            {Object.entries(c.reservations).map(([id, r]) => (
              <div key={id}>
                <span className="mono">{id}</span>
                <Status value={r.status} />
                <span>
                  {duration(r.actual_seconds ?? r.seconds)}{" "}
                  {r.status === "settled" ? "used" : "reserved"}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted actor-empty">No compute has been reserved yet.</p>
        )}
      </section>
      <EnvironmentsView campaign={campaign} onSelect={onSelect} />
    </div>
  );
}
