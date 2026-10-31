-- Production traces buffered for the overseer. A deployment posts through the `ingest` edge
-- function with its own token; the lab pulls with the service role. No public policies.
create table public.forge_ingest_tokens (
  id text primary key,
  token_sha256 text not null unique,
  source text not null check (source ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$'),
  pack_id text not null check (pack_id ~ '^[a-z0-9][a-z0-9-]{1,63}$'),
  daily_cap integer not null default 10000 check (daily_cap > 0),
  revoked boolean not null default false,
  created_at timestamptz not null default now()
);
alter table public.forge_ingest_tokens enable row level security;
comment on table public.forge_ingest_tokens is 'Overseer ingest tokens, stored as SHA-256 only. Service role only.';

create table public.forge_traces (
  id bigint generated always as identity primary key,
  source text not null,
  trace_id text not null,
  pack_id text not null,
  payload jsonb not null,
  received_at timestamptz not null default now(),
  pulled_at timestamptz,
  unique (source, trace_id)
);
create index forge_traces_unpulled on public.forge_traces (id) where pulled_at is null;
create index forge_traces_source_day on public.forge_traces (source, received_at);
alter table public.forge_traces enable row level security;
comment on table public.forge_traces is 'Production traces awaiting the overseer. Service role only.';
