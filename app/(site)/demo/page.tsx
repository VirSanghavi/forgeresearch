import type { Metadata } from "next";
import Link from "next/link";
import { SiteHeader } from "@/components/site/header";
import site from "@/components/site/site.module.css";
import styles from "./demo.module.css";

export const metadata: Metadata = {
  title: "Forge · Recorded demo",
  description:
    "A captioned, read-only recording of Forge diagnosing failures, coordinating research agents, and evaluating model changes.",
};

const reasons = [
  {
    n: "01",
    title: "A team keeps the checks independent",
    body: "The director chooses the next experiment, the trainer runs it, and separate evaluator and governance roles check the result. That separation helps stop the team from grading its own homework.",
  },
  {
    n: "02",
    title: "Agent Relay carries accountable handoffs",
    body: "Relay moves work and findings between roles with delivery receipts. Agents can work independently without losing who asked, what came back, or which evidence a decision used.",
  },
  {
    n: "03",
    title: "Terse keeps the campaign resumable",
    body: "Terse stores the campaign, experiments, budgets and reviews as durable actor state. If a worker stops, the research record survives and the run can be reconciled instead of reconstructed from chat.",
  },
];

export default function DemoPage() {
  return (
    <div className={`${site.root} ${styles.page}`}>
      <SiteHeader />
      <main className={styles.main}>
        <section className={styles.intro} aria-labelledby="demo-title">
          <div className={styles.introCopy}>
            <p className={styles.eyebrow}>FORGE / RECORDED WALKTHROUGH</p>
            <h1 id="demo-title">A model should have to earn its promotion.</h1>
            <p className={styles.lede}>
              Two minutes inside Forge: find a real failure, give specialists
              distinct jobs, train a candidate, then let independent evidence
              decide whether it ships.
            </p>
            <div className={styles.status}>
              <span className={styles.statusMark} aria-hidden="true" />
              <strong>Read-only recording</strong>
              <span>·</span>
              <span>2:00</span>
            </div>
          </div>
          <aside className={styles.result} aria-label="Recorded result">
            <p className={styles.resultLabel}>ONE HELD-OUT RUN</p>
            <p className={styles.resultNumber}>
              1 <span>→</span> 10
            </p>
            <p className={styles.resultCaption}>correct answers out of 24</p>
            <div className={styles.resultRule} />
            <p className={styles.resultFoot}>
              A second candidate regressed, so Forge kept the original model.
            </p>
          </aside>
        </section>

        <figure className={styles.film}>
          <video
            controls
            playsInline
            preload="metadata"
            poster="/demo/poster.jpg"
            aria-label="Two-minute captioned Forge product walkthrough"
          >
            <source src="/demo/forge-launch-2min.mp4" type="video/mp4" />
            Your browser does not support the video element. Use the download
            link below.
          </video>
          <figcaption>
            Captioned and silent. The public recording has no audio track.
            <a href="/demo/forge-launch-2min.mp4" download>
              Download the 2-minute film
            </a>
          </figcaption>
        </figure>

        <section className={styles.why} aria-labelledby="why-title">
          <div className={styles.sectionLead}>
            <p className={styles.eyebrow}>WHY THE MACHINERY MATTERS</p>
            <h2 id="why-title">Each part prevents a different failure.</h2>
          </div>
          <ol className={styles.reasonList}>
            {reasons.map((reason) => (
              <li key={reason.n}>
                <span className={styles.reasonNumber}>{reason.n}</span>
                <div>
                  <h3>{reason.title}</h3>
                  <p>{reason.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section className={styles.local} aria-labelledby="local-title">
          <div>
            <p className={styles.eyebrow}>TO RUN IT YOURSELF</p>
            <h2 id="local-title">
              The hosted page is a recording. The lab runs locally.
            </h2>
            <p>
              The public site has no research engine attached, so campaign and
              evidence APIs are unavailable there. A local run uses your own
              machine and optional personal model credentials.
            </p>
          </div>
          <div className={styles.localSteps}>
            <p>Node 24 and Git</p>
            <pre>
              <code>
                {
                  "git clone https://github.com/VirSanghavi/forgeresearch.git\ncd forgeresearch\nnpm run lab"
                }
              </code>
            </pre>
            <p>
              Setup creates an ignored `.env` file. An OpenAI key is optional;
              cloud GPU work uses your own Modal account.
            </p>
            <Link href="https://github.com/VirSanghavi/forgeresearch">
              Get the source <span aria-hidden="true">↗</span>
            </Link>
          </div>
        </section>
      </main>
      <footer className={styles.footer}>
        <span>Forge · autonomous post-training lab</span>
        <Link href="/">Back to the project</Link>
      </footer>
    </div>
  );
}
