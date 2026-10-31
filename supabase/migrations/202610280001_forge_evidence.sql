create table public.forge_campaigns (
  id text primary key,
  snapshot jsonb not null,
  archived_at timestamptz not null default now()
);
alter table public.forge_campaigns enable row level security;
comment on table public.forge_campaigns is 'Private research evidence mirror. Service role only. Terse remains authoritative.';

insert into storage.buckets (id, name, public, file_size_limit)
values ('forge-artifacts', 'forge-artifacts', false, 104857600)
on conflict (id) do nothing;

-- No anonymous or authenticated read/write policies. Evidence is archived by the
-- trusted server and shared only through checksum-verified local downloads.
