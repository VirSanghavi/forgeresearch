"use client";
/* Light and dark themes for the landing page. The theme lives on
   <html data-site-theme>, set before first paint by THEME_SCRIPT so the page never
   flashes the wrong palette. A stored choice wins; otherwise the system setting is
   followed live. The lab keeps its own theme key and attribute. */
import { useEffect, useSyncExternalStore } from "react";
import { Icon } from "@/components/icon";

export type SiteTheme = "light" | "dark";
const KEY = "forge-site-theme";
const QUERY = "(prefers-color-scheme: dark)";

export const THEME_SCRIPT = `(function(){var t;try{t=localStorage.getItem("${KEY}")}catch(e){}if(t!=="light"&&t!=="dark"){try{t=matchMedia("${QUERY}").matches?"dark":"light"}catch(e){t="light"}}document.documentElement.dataset.siteTheme=t})()`;

function read(): SiteTheme {
  return document.documentElement.dataset.siteTheme === "dark"
    ? "dark"
    : "light";
}

function subscribe(onChange: () => void) {
  const mo = new MutationObserver(onChange);
  mo.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-site-theme"],
  });
  return () => mo.disconnect();
}

/** The current theme; re-renders when it changes. Light during server render. */
export function useSiteTheme(): SiteTheme {
  return useSyncExternalStore(subscribe, read, () => "light");
}

function stored(): SiteTheme | null {
  try {
    const t = localStorage.getItem(KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

function apply(theme: SiteTheme) {
  document.documentElement.dataset.siteTheme = theme;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", theme === "dark" ? "#0e0e0e" : "#ffffff");
}

export function ThemeToggle({ className }: { className: string }) {
  const theme = useSiteTheme();
  // Follow the system setting until the visitor picks a theme.
  useEffect(() => {
    const mq = window.matchMedia(QUERY);
    const sync = () => {
      if (!stored()) apply(mq.matches ? "dark" : "light");
    };
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  const next: SiteTheme = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className={className}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
      onClick={() => {
        apply(next);
        try {
          localStorage.setItem(KEY, next);
        } catch {
          /* Private mode: the choice lasts for this page view. */
        }
      }}
    >
      <Icon name={theme === "dark" ? "sun" : "moon"} size={18} />
    </button>
  );
}
