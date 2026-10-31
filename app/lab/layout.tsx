import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Forge · Autonomous research lab",
  description:
    "Give a model a benchmark and a budget. Follow real post-training experiments, independent governance, and reproducible evidence.",
};

export default function LabLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <>
      {/* The lab's IBM Plex faces; the landing page at / uses its own. */}
      <link
        rel="preload"
        href="/fonts/ibm-plex-sans-latin.woff2"
        as="font"
        type="font/woff2"
        crossOrigin="anonymous"
      />
      <link
        rel="preload"
        href="/fonts/ibm-plex-mono-latin.woff2"
        as="font"
        type="font/woff2"
        crossOrigin="anonymous"
      />
      {children}
    </>
  );
}
