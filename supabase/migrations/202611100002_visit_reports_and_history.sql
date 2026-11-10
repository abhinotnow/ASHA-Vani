-- Split patient identity from visits and allow multiple visits in one call.
-- The existing one-row-per-call visit becomes visit number 1 and is preserved.
create extension if not exists pgcrypto;

create table if not exists public.patients (
    id uuid primary key default gen_random_uuid(),
    source_visit_key text unique,
    full_name text,
    spouse_or_guardian_name text,
    age_years integer,
    village text,
    tola text,
    created_at timestamptz not null default now()
);

alter table public.visits
    drop constraint if exists visits_session_id_key;

alter table public.visits
    add column if not exists visit_number integer not null default 1,
    add column if not exists patient_id uuid references public.patients(id) on delete set null,
    add column if not exists final_report jsonb not null default '{}'::jsonb,
    add column if not exists finalized_at timestamptz;

-- Preserve legacy identities as separate profiles. This intentionally does not
-- merge records that happen to have similar names.
insert into public.patients (
    source_visit_key, full_name, spouse_or_guardian_name, age_years, village, tola
)
select
    'legacy:' || v.id::text,
    nullif(v.patient->>'full_name', ''),
    nullif(v.patient->>'spouse_or_guardian_name', ''),
    nullif(v.patient->>'age_years', '')::integer,
    nullif(v.patient->>'village', ''),
    nullif(v.patient->>'tola', '')
from public.visits v
where nullif(v.patient->>'full_name', '') is not null
on conflict (source_visit_key) do nothing;

update public.visits v
set patient_id = p.id
from public.patients p
where p.source_visit_key = 'legacy:' || v.id::text
  and v.patient_id is null;

update public.visits v
set finalized_at = coalesce(v.finalized_at, v.updated_at),
    final_report = jsonb_build_object(
        'session_id', v.session_id,
        'visit_number', v.visit_number,
        'patient_id', v.patient_id,
        'patient', v.patient,
        'visit_values', v.visit_values,
        'triage', v.triage,
        'validation_issues', v.validation_issues,
        'verification', v.verification,
        'recorded_at', coalesce(v.finalized_at, v.updated_at),
        'is_simulated', c.is_simulated
    )
from public.calls c
where c.session_id = v.session_id
  and v.verification = 'VERIFIED'
  and v.final_report = '{}'::jsonb;

create unique index if not exists visits_session_visit_number_uidx
    on public.visits (session_id, visit_number);

create index if not exists visits_patient_updated_at_idx
    on public.visits (patient_id, updated_at desc)
    where patient_id is not null;

create index if not exists visits_patient_name_idx
    on public.visits (lower(patient->>'full_name'));

alter table public.patients enable row level security;

comment on table public.patients is
    'Patient identities are linked only after explicit ASHA confirmation.';
comment on column public.visits.visit_number is
    'One call can collect several independently verified visits.';
comment on column public.visits.final_report is
    'Sanitized visit snapshot for PHC register and patient history.';
