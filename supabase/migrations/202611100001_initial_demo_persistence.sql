-- Synthetic-demo persistence only. The server uses the Supabase service-role
-- key; no browser-facing policies are created in this initial migration.
create extension if not exists pgcrypto;

create table if not exists public.calls (
    session_id text primary key,
    caller text,
    source text not null default 'BROWSER_SIM',
    is_simulated boolean not null default true,
    started_at timestamptz not null default now(),
    ended_at timestamptz,
    end_reason text,
    created_at timestamptz not null default now()
);

create table if not exists public.visits (
    id uuid primary key default gen_random_uuid(),
    session_id text not null unique references public.calls(session_id) on delete cascade,
    last_utterance_seq integer not null default 0,
    patient jsonb not null default '{}'::jsonb,
    extracted_turn jsonb not null default '{}'::jsonb,
    visit_values jsonb not null default '{}'::jsonb,
    validation_issues jsonb not null default '[]'::jsonb,
    triage jsonb not null default '{"level":"ROUTINE","rules_fired":[]}'::jsonb,
    verification text not null default 'UNVERIFIED'
        check (verification in ('UNVERIFIED', 'VERIFIED', 'INCOMPLETE')),
    next_action text,
    updated_at timestamptz not null default now()
);

create table if not exists public.call_events (
    session_id text not null references public.calls(session_id) on delete cascade,
    seq integer not null,
    event_type text not null check (event_type in ('call_started', 'turn_completed', 'call_ended')),
    occurred_at timestamptz not null default now(),
    is_simulated boolean not null default true,
    payload jsonb not null default '{}'::jsonb,
    primary key (session_id, seq)
);

create index if not exists visits_updated_at_idx on public.visits (updated_at desc);
create index if not exists call_events_occurred_at_idx on public.call_events (occurred_at);

alter table public.calls enable row level security;
alter table public.visits enable row level security;
alter table public.call_events enable row level security;

comment on table public.calls is 'Synthetic ASHA-Vani demo call sessions.';
comment on table public.visits is 'Latest accumulated and verified-or-pending visit snapshot per call.';
comment on table public.call_events is 'Append-oriented turn-level events used to reconstruct the demo console.';
