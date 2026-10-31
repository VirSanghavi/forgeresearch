"use client";
import { useEffect, useState } from "react";
import { Icon } from "@/components/icon";
import DotGrid from "./reactbits/DotGrid";
import LogoLoop from "./reactbits/LogoLoop";
import s from "./site.module.css";
import { useSiteTheme } from "./theme";

const reduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Smooth in-page jump: ~550ms decelerating, landing 96px under the top (sticky header + 32). */
function jumpTo(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const target = el.getBoundingClientRect().top + window.scrollY - 96;
  if (reduced()) {
    window.scrollTo(0, target);
    return;
  }
  const start = window.scrollY;
  const t0 = performance.now();
  const ease = (t: number) => 1 - Math.pow(1 - t, 4); // out-quart
  const step = (now: number) => {
    const t = Math.min(1, (now - t0) / 550);
    window.scrollTo(0, start + (target - start) * ease(t));
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function ProductNav({
  items,
}: {
  items: { id: string; label: string; icon: string }[];
}) {
  const [active, setActive] = useState(items[0].id);
  useEffect(() => {
    // Same band as the reference side nav: rootMargin -20% top, -60% bottom.
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: "-20% 0px -60% 0px" },
    );
    items.forEach((i) => {
      const el = document.getElementById(i.id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, [items]);
  return (
    <nav className={s.sideNav} aria-label="Product sections">
      {items.map((i) => (
        <a
          key={i.id}
          href={`#${i.id}`}
          className={`${s.sideItem} ${active === i.id ? s.sideItemActive : ""}`}
          aria-current={active === i.id ? "true" : undefined}
          onClick={(e) => {
            e.preventDefault();
            jumpTo(i.id);
            history.replaceState(null, "", `#${i.id}`);
          }}
        >
          <Icon name={i.icon} size={18} />
          {i.label}
        </a>
      ))}
    </nav>
  );
}

export function HeroGrid({ className }: { className: string }) {
  const dark = useSiteTheme() === "dark";
  const [on, setOn] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setOn(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div
      className={`${className} ${on ? s.heroGridOn : ""}`}
      aria-hidden="true"
    >
      <DotGrid
        dotSize={3}
        gap={15}
        baseColor={dark ? "#2a2a2a" : "#dedede"}
        activeColor={dark ? "#ededed" : "#111111"}
        proximity={110}
        swell={0.6}
        stretch={0.3}
        glow={0}
        fade={0.55}
        intro={false}
        clickShock={false}
      />
    </div>
  );
}

export function CtaGrid({ className }: { className: string }) {
  const dark = useSiteTheme() === "dark";
  return (
    <div className={className} aria-hidden="true">
      <DotGrid
        dotSize={3}
        gap={15}
        baseColor={dark ? "#2a2a2a" : "#dedede"}
        activeColor={dark ? "#ededed" : "#111111"}
        proximity={110}
        swell={0.6}
        stretch={0.3}
        glow={0}
        fade={0.6}
        intro={false}
        clickShock={false}
      />
    </div>
  );
}

export type Logo = {
  src: string;
  name: string;
  width: number;
  height: number;
  /** Rendered height in px, tuned so every mark reads at the same visual weight. */
  h: number;
};

/** Official marks, recolored to one neutral grey. */
export function BuiltWith({ logos }: { logos: Logo[] }) {
  return (
    <LogoLoop
      logos={logos.map((l) => ({
        node: (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={l.src}
            alt={l.name}
            width={l.width}
            height={l.height}
            className={s.stripLogo}
            style={{ height: l.h, width: "auto" }}
            draggable={false}
          />
        ),
        title: l.name,
      }))}
      speed={13}
      gap={56}
      logoHeight={32}
      fadeOut
      fadeOutWidth={0.09}
      pauseOnHover={false}
      intro={false}
      ariaLabel="Software Forge is built with"
    />
  );
}

export type Quote = {
  initials: string;
  from: string;
  to: string;
  route: string;
  text: string;
};

export function QuoteCarousel({
  quotes,
  title,
}: {
  quotes: Quote[];
  title: React.ReactNode;
}) {
  const [i, setI] = useState(0);
  const [prev, setPrev] = useState<number | null>(null);
  const go = (n: number) => {
    setPrev(i);
    setI((n + quotes.length) % quotes.length);
  };
  return (
    <div>
      <div className={s.carouselTop}>
        {title}
        <div className={s.carouselBtns}>
          <button
            type="button"
            className={s.roundBtn}
            aria-label="Previous message"
            onClick={() => go(i - 1)}
          >
            <span
              style={{ display: "inline-flex", transform: "rotate(180deg)" }}
            >
              <Icon name="chevron" size={18} />
            </span>
          </button>
          <button
            type="button"
            className={s.roundBtn}
            aria-label="Next message"
            onClick={() => go(i + 1)}
          >
            <Icon name="chevron" size={18} />
          </button>
        </div>
      </div>
      <div className={s.quoteCard}>
        <div className={s.quoteStage} aria-live="polite">
          {quotes.map((q, n) => (
            <figure
              key={n}
              className={`${s.quote} ${n === i ? s.quoteIn : ""} ${n === prev && n !== i ? s.quoteOut : ""}`}
              aria-hidden={n !== i}
            >
              <div>
                <span className={s.eyebrow}>
                  <span className={s.eyebrowIcon}>
                    <Icon name="message" size={12} />
                  </span>
                  What the agents said
                </span>
                <blockquote>{q.text}</blockquote>
              </div>
              <figcaption className={s.quoteBy}>
                <span className={s.avatar}>{q.initials}</span>
                <span className={s.quoteMeta}>
                  <span>
                    {q.from} to {q.to}
                  </span>
                  <span>{q.route}</span>
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
        <div className={s.quoteSide}>
          <div className={s.thread} role="tablist" aria-label="Messages">
            {quotes.map((q, n) => (
              <button
                key={n}
                type="button"
                role="tab"
                aria-selected={n === i}
                className={`${s.threadRow} ${n === i ? s.threadRowOn : ""}`}
                onClick={() => n !== i && go(n)}
              >
                <span className={s.avatar}>{q.initials}</span>
                <span>
                  {q.from} to {q.to}
                </span>
              </button>
            ))}
          </div>
          <p className={s.sideNote}>
            Messages from a real Forge run, in plain words.
          </p>
        </div>
      </div>
    </div>
  );
}
