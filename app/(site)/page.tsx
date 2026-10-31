/* eslint-disable @next/next/no-img-element -- one static product capture with fixed size */
import Link from "next/link";
import { ForgeMark, Icon } from "@/components/icon";
import { SiteHeader } from "@/components/site/header";
import { ParticleHero } from "@/components/site/particle-hero";
import {
  BuiltWith,
  CtaGrid,
  HeroGrid,
  ProductNav,
  QuoteCarousel,
  type Logo,
  type Quote,
} from "@/components/site/motion";
import {
  AgentFlow,
  FlagList,
  RecoveryTimeline,
  ShipSteps,
  AnswerExample,
} from "@/components/site/live";
import {
  JournalDrawing,
  TeamDrawing,
  TracesDrawing,
} from "@/components/site/drawings";
import { MESSAGES, ROLES } from "@/components/site/data";
import s from "@/components/site/site.module.css";

// Written for a first-time visitor: plain words, and the before/after example is the proof.
// The only number on the page is the run time in the closing line.

const SIDE = [
  { id: "production", label: "Mistakes", icon: "production" },
  { id: "agents", label: "Agents", icon: "chat" },
  { id: "durable", label: "Memory", icon: "terse" },
  { id: "training", label: "Your model", icon: "results" },
];

// Official marks only; every one is a dependency in pyproject.toml or package.json.
const BUILT_WITH: Logo[] = [
  {
    src: "/site/logos/pytorch.svg",
    name: "PyTorch",
    width: 487,
    height: 120,
    h: 26,
  },
  {
    src: "/site/logos/huggingface.svg",
    name: "Hugging Face",
    width: 963,
    height: 256,
    h: 34,
  },
  { src: "/site/logos/qwen.svg", name: "Qwen", width: 24, height: 24, h: 28 },
  {
    src: "/site/logos/eleutherai.svg",
    name: "EleutherAI",
    width: 800,
    height: 149,
    h: 22,
  },
  {
    src: "/site/logos/agent-relay.svg",
    name: "Agent Relay",
    width: 340,
    height: 54,
    h: 24,
  },
  {
    src: "/site/logos/terse-durable-actors.png",
    name: "Durable Actors by Terse",
    width: 1370,
    height: 290,
    h: 32,
  },
];

const QUOTES: Quote[] = MESSAGES.map((m) => ({
  initials: ROLES[m.from].initials,
  from: ROLES[m.from].name,
  to: m.to === "room" ? "the whole team" : ROLES[m.to].name,
  route: m.route,
  text: m.text,
}));

const KEEPS = [
  ["target", "Goal and budget", "What the run is for and what it may spend"],
  ["experiments", "Experiments", "Every setting and every result"],
  ["governance", "Approvals", "A signed decision for each new version"],
  ["artifacts", "Saved files", "Every file, with a fingerprint to check it"],
  ["environments", "Practice problems", "The worlds that score each answer"],
  ["book", "Research notes", "Every idea and decision, in order"],
];

function Eyebrow({
  icon,
  children,
}: {
  icon: string;
  children: React.ReactNode;
}) {
  return (
    <span className={s.eyebrow}>
      <span className={s.eyebrowIcon}>
        <Icon name={icon} size={12} />
      </span>
      {children}
    </span>
  );
}

function Feature({
  icon,
  title,
  children,
}: {
  icon: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className={s.feature}>
      <h4>
        <Icon name={icon} size={16} />
        {title}
      </h4>
      <p className={s.small}>{children}</p>
    </div>
  );
}

function RecordList({
  title,
  icon,
  items,
}: {
  title: string;
  icon: string;
  items: [string, string][];
}) {
  return (
    <div className={s.recordCard}>
      <div className={s.recordHead}>{title}</div>
      <ul className={s.ledger}>
        {items.map(([name, note], i) => (
          <li key={`${i}-${name}`}>
            <span className={s.tick}>
              <Icon name={icon} size={14} />
            </span>
            <span>
              {name}
              <span className={s.note}>{note}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function SitePage() {
  return (
    <div className={s.root} id="top">
      <SiteHeader />
      <main>
        {/* Hero */}
        <section className={s.hero}>
          <HeroGrid className={s.heroGrid} />
          <div className={s.container}>
            <div className={s.heroText}>
              <h1 className={s.h1}>
                <span className={s.soft}>Post-training, </span>
                fully automated
              </h1>
              <p className={`${s.lead} ${s.heroSub}`}>
                Forge trains an AI you own to be the best at your
                business&apos;s work. It learns from your real conversations,
                keeps every model that does the job better, and the model is
                yours to keep.
              </p>
              <div className={s.ctaRow}>
                <Link href="/lab" className={s.btn}>
                  Open the lab
                </Link>
                <a href="#example" className={s.btnGhost}>
                  See the example
                </a>
              </div>
            </div>
            <div className={s.heroShot}>
              <ParticleHero />
            </div>
            <div className={s.strip}>
              <p className={s.stripLabel}>Built with</p>
              <span className={s.stripDivider} aria-hidden="true" />
              <div className={s.stripLoop}>
                <BuiltWith logos={BUILT_WITH} />
              </div>
            </div>
          </div>
        </section>

        {/* The proof: one question, before and after */}
        <section className={s.section} id="example">
          <div className={s.container}>
            <div className={s.sectionHead}>
              <Eyebrow icon="target">See it learn</Eyebrow>
              <h2 className={s.h2}>
                Questions it used to miss, it now gets right
              </h2>
            </div>
            <AnswerExample />
          </div>
        </section>

        {/* Product */}
        <section className={`${s.section} ${s.ruleTop}`} id="product">
          <div className={s.container}>
            <div className={s.sectionHead}>
              <Eyebrow icon="overview">How Forge works</Eyebrow>
              <h2 className={s.h2}>
                From your conversations to the best model for your work
              </h2>
            </div>
            <div className={`${s.grid} ${s.product}`}>
              <ProductNav items={SIDE} />
              <div className={s.blocks}>
                <article className={s.block} id="production">
                  <div className={s.blockText}>
                    <h3 className={s.h3}>It finds the mistakes</h3>
                    <p className={`${s.lead} ${s.blockLead}`}>
                      Upload your agent&apos;s conversations. Forge spots the
                      answers that need fixing, all by itself.
                    </p>
                    <div className={s.features}>
                      <Feature icon="production" title="It checks every answer">
                        Missing answers, replies that were cut off, math slips
                        and corrections from users.
                      </Feature>
                      <Feature icon="target" title="Its flags hold up">
                        Every answer it flagged was confirmed by a person.
                      </Feature>
                      <Feature icon="lock" title="The final test stays private">
                        Anything that matches the final test is set aside first.
                      </Feature>
                    </div>
                  </div>
                  <div className={s.visual}>
                    <FlagList />
                  </div>
                </article>

                <article className={s.block} id="agents">
                  <div className={s.blockText}>
                    <h3 className={s.h3}>Five agents do the work</h3>
                    <p className={`${s.lead} ${s.blockLead}`}>
                      A researcher, a director, a trainer, a tester and a
                      council that approves. They talk to each other, and every
                      message is kept.
                    </p>
                    <div className={s.features}>
                      <Feature icon="message" title="Direct messages">
                        Each handoff goes to one agent, who confirms it before
                        starting.
                      </Feature>
                      <Feature icon="chat" title="A shared room">
                        Plans and results are posted for the whole team.
                      </Feature>
                      <Feature icon="governance" title="Replies in threads">
                        The council answers right under each new version.
                      </Feature>
                      <Feature icon="receipt" title="Read receipts">
                        You can see who read each message.
                      </Feature>
                    </div>
                  </div>
                  <div className={s.visual}>
                    <AgentFlow />
                  </div>
                </article>

                <article className={s.block} id="durable">
                  <div className={s.blockText}>
                    <h3 className={s.h3}>
                      A record you can always resume from
                    </h3>
                    <p className={`${s.lead} ${s.blockLead}`}>
                      Every goal, approval and saved file lives in a permanent
                      record, so a run picks up exactly where it was.
                    </p>
                    <div className={s.features}>
                      <Feature icon="halt" title="Stopped on purpose">
                        We force-stopped Forge in the middle of training.
                      </Feature>
                      <Feature icon="resume" title="Back in seconds">
                        Every record came back exactly as it was.
                      </Feature>
                      <Feature icon="clock" title="Right where it left off">
                        Training carried on from the very same step.
                      </Feature>
                    </div>
                  </div>
                  <div className={s.visual}>
                    <RecoveryTimeline />
                  </div>
                </article>

                <article className={s.block} id="training">
                  <div className={s.blockText}>
                    <h3 className={s.h3}>A model that is yours</h3>
                    <p className={`${s.lead} ${s.blockLead}`}>
                      Forge tests the new model on brand-new questions. Then you
                      download it and run it wherever you like.
                    </p>
                    <div className={s.features}>
                      <Feature
                        icon="production"
                        title="Tested on new questions"
                      >
                        Some real mistakes are kept aside to check the model
                        after training.
                      </Feature>
                      <Feature icon="lock" title="A fair final exam">
                        The final test opens only after the best version is
                        chosen.
                      </Feature>
                      <Feature icon="governance" title="The best version wins">
                        The council promotes the version that did best in
                        practice.
                      </Feature>
                      <Feature icon="download" title="Yours to keep">
                        Download the model, the report and every step of the
                        research.
                      </Feature>
                    </div>
                    <a href="#example" className={`${s.btn} ${s.blockButton}`}>
                      See the example
                    </a>
                  </div>
                  <div className={s.visual}>
                    <ShipSteps />
                  </div>
                </article>
              </div>
            </div>
          </div>
        </section>

        {/* Under the hood */}
        <section className={`${s.section} ${s.band}`} id="platform">
          <div className={s.container}>
            <div className={`${s.sectionHead} ${s.narrow}`}>
              <Eyebrow icon="environments">Under the hood</Eyebrow>
              <h2 className={s.h2}>Every step leaves a trail you can check</h2>
            </div>
            <div className={s.cols3}>
              <div className={s.col}>
                <span className={s.colNum}>1</span>
                <div className={s.drawing}>
                  <TracesDrawing />
                </div>
                <h3 className={s.h3}>It learns from your traffic</h3>
                <p className={s.body}>
                  Your agent&apos;s own answers become the lesson plan.
                </p>
              </div>
              <div className={s.col}>
                <span className={s.colNum}>2</span>
                <div className={s.drawing}>
                  <TeamDrawing />
                </div>
                <h3 className={s.h3}>The team checks its own work</h3>
                <p className={s.body}>
                  The trainer hands each new version to the tester, and the
                  council reviews it before anything is promoted.
                </p>
              </div>
              <div className={s.col}>
                <span className={s.colNum}>3</span>
                <div className={s.drawing}>
                  <JournalDrawing />
                </div>
                <h3 className={s.h3}>Every step is signed and saved</h3>
                <p className={s.body}>
                  Each change is signed and written down, so the whole run can
                  be replayed.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Results, in plain words */}
        <section className={`${s.section} ${s.ruleTop}`} id="results">
          <div className={s.container}>
            <div className={`${s.sectionHead} ${s.narrow}`}>
              <Eyebrow icon="results">Results</Eyebrow>
              <h2 className={s.h2}>
                Better at your work, and it keeps improving
              </h2>
            </div>
            <div className={s.cols4}>
              <div className={s.col4}>
                <h3 className={s.h3sm}>
                  <Icon name="check" size={16} />
                  Fixes what it missed
                </h3>
                <p className={s.body}>
                  After training, it answers questions it used to miss.
                </p>
              </div>
              <div className={s.col4}>
                <h3 className={s.h3sm}>
                  <Icon name="refresh" size={16} />
                  Keeps what it knew
                </h3>
                <p className={s.body}>
                  Everything it already answered well, it still answers well.
                </p>
              </div>
              <div className={s.col4}>
                <h3 className={s.h3sm}>
                  <Icon name="lock" size={16} />
                  Measured fairly
                </h3>
                <p className={s.body}>
                  It is tested on brand-new questions, held back from training.
                </p>
              </div>
              <div className={s.col4}>
                <h3 className={s.h3sm}>
                  <Icon name="download" size={16} />
                  Yours to run
                </h3>
                <p className={s.body}>
                  You download the model and own it, along with what it costs to
                  run.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* The record */}
        <section className={`${s.section} ${s.ruleTop}`} id="record">
          <div className={s.container}>
            <div className={s.splitHead}>
              <div className={s.sectionHead}>
                <Eyebrow icon="record">The record</Eyebrow>
                <h2 className={s.h2}>Every decision is on the record</h2>
              </div>
              <p>
                Each message, review and saved file is kept with the run, so you
                can trace every result back to where it came from.
              </p>
            </div>
            <div className={s.cards3}>
              <div className={s.card}>
                <RecordList
                  title="Versions"
                  icon="experiments"
                  items={[
                    ["First version", "Promoted by the council"],
                    ["Second version", "Kept on the record"],
                  ]}
                />
                <h3 className={s.h3sm}>
                  <Icon name="experiments" size={16} />
                  Every version, with its evidence
                </h3>
                <p className={s.body}>
                  Each version keeps its settings, its results and the
                  council&apos;s decision.
                </p>
              </div>
              <div className={s.card}>
                <RecordList
                  title="Council reviews"
                  icon="check"
                  items={[
                    ["Plan check", "Approved before training"],
                    ["Version review", "Approved after testing"],
                    ["Plan check", "Approved before training"],
                    ["Version review", "Approved after testing"],
                  ]}
                />
                <h3 className={s.h3sm}>
                  <Icon name="governance" size={16} />A signed approval for each
                  step
                </h3>
                <p className={s.body}>
                  Every review is tied to a written policy.
                </p>
              </div>
              <div className={s.card}>
                <RecordList
                  title="Saved files"
                  icon="artifacts"
                  items={[
                    ["Where the data came from", "Saved at the start"],
                    ["The model before training", "Tested and saved"],
                    ["The new model", "Saved after training"],
                    ["Final test results", "Saved at the end"],
                    ["The full report", "Ready to download"],
                  ]}
                />
                <h3 className={s.h3sm}>
                  <Icon name="download" size={16} />
                  Everything is downloadable
                </h3>
                <p className={s.body}>
                  The model, the report and the full research trail.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Inverted band: what the agents said */}
        <section
          className={`${s.sectionTight} ${s.dark}`}
          data-header-theme="dark"
          id="voices"
        >
          <div className={s.container}>
            <QuoteCarousel
              quotes={QUOTES}
              title={
                <div className={s.sectionHead}>
                  <Eyebrow icon="message">Agent Relay</Eyebrow>
                  <h2 className={s.h2}>The agents talk to each other</h2>
                </div>
              }
            />
            <div className={s.tilesHead}>
              <h3 className={s.h3}>What the run remembers</h3>
              <p className={s.body}>
                Six records, each kept by its own durable actor and saved as the
                run goes, so a crash picks up where it left off.
              </p>
            </div>
            <div className={s.tiles}>
              {KEEPS.map(([icon, name, text]) => (
                <div className={s.tile} key={name}>
                  <span className={s.tileIcon}>
                    <Icon name={icon} size={18} />
                  </span>
                  <strong>{name}</strong>
                  <span>{text}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* The product, once */}
        <section className={`${s.sectionTight} ${s.band}`} id="lab">
          <div className={s.container}>
            <div className={s.splitHead}>
              <div className={s.sectionHead}>
                <Eyebrow icon="overview">The lab</Eyebrow>
                <h2 className={s.h2}>Watch every run as it happens</h2>
              </div>
              <p>
                The agents&apos; conversation, every version and the results
                live in one place while the run is going.
              </p>
            </div>
            <figure className={s.productShot}>
              <div className={s.frame}>
                <img
                  src="/site/hero-chat.webp"
                  width={2000}
                  height={1250}
                  alt="The Forge lab: the five agents' conversation in the research room"
                  loading="lazy"
                />
              </div>
            </figure>
          </div>
        </section>

        {/* Final CTA */}
        <section className={s.cta}>
          <CtaGrid className={s.ctaGrid} />
          <div className={s.container}>
            <div className={s.ctaText}>
              <h2 className={s.h1}>
                <span className={s.soft}>Your work.</span>
                <br />
                Your model.
              </h2>
              <div className={s.ctaRow}>
                <Link href="/lab" className={s.btn}>
                  Open the lab
                </Link>
                <a href="#example" className={s.btnGhost}>
                  See the example
                </a>
              </div>
              <p className={s.ctaNote}>
                A full run took about 12 minutes, from one click to a finished
                model.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className={s.footer}>
        <div className={s.container}>
          <div className={s.footerTop}>
            <span className={s.footerBrand}>
              <ForgeMark size={30} />
              forge.
            </span>
            <div className={s.footerCols}>
              <div>
                <h2>Product</h2>
                <ul>
                  <li>
                    <a href="#production">Finding mistakes</a>
                  </li>
                  <li>
                    <a href="#agents">Agents</a>
                  </li>
                  <li>
                    <a href="#durable">Memory</a>
                  </li>
                  <li>
                    <a href="#training">Your model</a>
                  </li>
                </ul>
              </div>
              <div>
                <h2>See it work</h2>
                <ul>
                  <li>
                    <a href="#example">The example</a>
                  </li>
                  <li>
                    <a href="#results">Results</a>
                  </li>
                  <li>
                    <a href="#record">The record</a>
                  </li>
                </ul>
              </div>
              <div>
                <h2>Lab</h2>
                <ul>
                  <li>
                    <Link href="/lab">Open the lab</Link>
                  </li>
                  <li>
                    <a href="#lab">Inside the lab</a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <div className={s.footerBottom}>
            <span>Forge · Your own model, trained on your own traffic</span>
            <span>© 2026 Forge</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
