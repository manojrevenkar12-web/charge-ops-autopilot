-- The central system's record. Append-only: what a charger said, and what we told it, is never edited.
create table if not exists ocpp_message (
  id          bigserial primary key,
  ts          bigint not null,        -- when it happened at the charger (epoch seconds)
  received    bigint not null,        -- when the CSMS received it (features use this)
  charger_id  text   not null,
  direction   text   not null check (direction in ('cp', 'cs', 'res')),
  action      text   not null,
  message_id  text,
  payload     jsonb  not null,        -- the OCPP 2.0.1 payload exactly as on the wire
  flat        jsonb  not null         -- the same message as one ocpp_flat row
);
create index if not exists ocpp_message_received on ocpp_message (received);
create index if not exists ocpp_message_charger on ocpp_message (charger_id, received);

create table if not exists command (
  id          bigserial primary key,
  ts          bigint not null,
  charger_id  text   not null,
  action      text   not null,
  evse        int    not null default 0,
  params      jsonb  not null default '{}',
  decided_by  text   not null,
  incident    text,
  reason      text,
  shield      jsonb  not null,        -- the verdict, with every rule that blocked
  result      text
);

create or replace function cpa_append_only() returns trigger language plpgsql as $$
begin
  raise exception '% is append-only', tg_table_name;
end $$;
drop trigger if exists ocpp_message_append_only on ocpp_message;
create trigger ocpp_message_append_only before update or delete on ocpp_message
  for each row execute function cpa_append_only();
