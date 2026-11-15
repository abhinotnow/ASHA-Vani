-- ASHA worker registry, multi-phone mapping, pending phone links, and call language metadata.

create table if not exists public.asha_workers (
    asha_id      text primary key,
    full_name    text not null,
    language     text not null default 'hi'
        check (language in ('hi','ta','te','kn','ml','mr','bn','gu','pa','en')),
    state        text,
    district     text,
    phc          text,
    active       boolean not null default true,
    is_simulated boolean not null default true,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

create table if not exists public.asha_worker_phones (
    phone_e164 text not null check (phone_e164 ~ '^\+91[6-9][0-9]{9}$'),
    asha_id    text not null references public.asha_workers(asha_id) on delete cascade,
    label      text not null default 'own',
    verified   boolean not null default true,
    created_at timestamptz not null default now(),
    primary key (phone_e164, asha_id)
);
create index if not exists asha_worker_phones_phone_idx on public.asha_worker_phones (phone_e164);

create table if not exists public.pending_phone_links (
    id              uuid primary key default gen_random_uuid(),
    phone_e164      text not null,
    claimed_asha_id text not null references public.asha_workers(asha_id),
    session_id      text references public.calls(session_id) on delete set null,
    status          text not null default 'PENDING'
        check (status in ('PENDING','APPROVED','REJECTED')),
    created_at      timestamptz not null default now(),
    resolved_at     timestamptz,
    unique (phone_e164, claimed_asha_id, status)
);

alter table public.calls
    add column if not exists asha_id text references public.asha_workers(asha_id),
    add column if not exists language text,
    add column if not exists language_source text
        check (language_source in ('REGISTERED','AUTO_DETECTED','SWITCHED','DEFAULT'));

alter table public.call_events drop constraint if exists call_events_event_type_check;
alter table public.call_events add constraint call_events_event_type_check
    check (event_type in ('call_started','turn_completed','call_ended','language_changed'));

alter table public.asha_workers enable row level security;
alter table public.asha_worker_phones enable row level security;
alter table public.pending_phone_links enable row level security;

comment on table public.asha_workers is 'Registered ASHA workers with preferred language.';
comment on table public.asha_worker_phones is 'E.164 phone-to-worker links (supports shared family phones).';
comment on table public.pending_phone_links is 'Unverified phone claims queued for PHC staff approval.';

