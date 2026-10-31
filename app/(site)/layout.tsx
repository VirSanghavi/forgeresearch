import type { Metadata, Viewport } from "next";
import { DM_Mono, Geist } from "next/font/google";
import { THEME_SCRIPT } from "@/components/site/theme";

// Geist (SIL OFL) supplies the site's sans-serif display and text face.
const geist = Geist({
  subsets: ["latin"],
  variable: "--site-font",
  display: "swap",
});

// DM Mono (SIL OFL) is the code face the reference ships.
const mono = DM_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--site-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Forge · Post-training that runs itself",
  description:
    "Forge trains an AI you own to be the best at your business's work, from your real conversations. Fully automated, and the weights are yours.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0e0e0e" },
  ],
};

export default function SiteLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className={`${geist.variable} ${mono.variable}`}>
      {/* Sets the theme before first paint, so the page never flashes the wrong one. */}
      <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      {children}
    </div>
  );
}
