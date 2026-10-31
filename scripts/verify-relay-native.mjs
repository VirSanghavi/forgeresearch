import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
const require=createRequire(import.meta.url);
try { const Database=require('better-sqlite3'); new Database(':memory:').close(); }
catch { const result=spawnSync('npm',['rebuild','better-sqlite3'],{stdio:'inherit'}); if(result.status!==0)process.exit(result.status||1); }
