import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
export default defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    ".next/**",
    "node_modules/**",
    "services/**",
    "packages/**",
    "apps/**",
    "data/**",
    "artifacts/**",
    ".venv/**",
    "supabase/**",
    "next-env.d.ts",
  ]),
]);
