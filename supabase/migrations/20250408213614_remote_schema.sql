drop trigger if exists "update_connections_updated_at" on "public"."connections";

drop trigger if exists "update_events_updated_at" on "public"."events";

drop trigger if exists "update_messages_updated_at" on "public"."messages";

drop trigger if exists "update_profiles_updated_at" on "public"."profiles";

revoke delete on table "public"."connections" from "anon";

revoke insert on table "public"."connections" from "anon";

revoke references on table "public"."connections" from "anon";

revoke select on table "public"."connections" from "anon";

revoke trigger on table "public"."connections" from "anon";

revoke truncate on table "public"."connections" from "anon";

revoke update on table "public"."connections" from "anon";

revoke delete on table "public"."connections" from "authenticated";

revoke insert on table "public"."connections" from "authenticated";

revoke references on table "public"."connections" from "authenticated";

revoke select on table "public"."connections" from "authenticated";

revoke trigger on table "public"."connections" from "authenticated";

revoke truncate on table "public"."connections" from "authenticated";

revoke update on table "public"."connections" from "authenticated";

revoke delete on table "public"."connections" from "service_role";

revoke insert on table "public"."connections" from "service_role";

revoke references on table "public"."connections" from "service_role";

revoke select on table "public"."connections" from "service_role";

revoke trigger on table "public"."connections" from "service_role";

revoke truncate on table "public"."connections" from "service_role";

revoke update on table "public"."connections" from "service_role";

revoke delete on table "public"."event_attendees" from "anon";

revoke insert on table "public"."event_attendees" from "anon";

revoke references on table "public"."event_attendees" from "anon";

revoke select on table "public"."event_attendees" from "anon";

revoke trigger on table "public"."event_attendees" from "anon";

revoke truncate on table "public"."event_attendees" from "anon";

revoke update on table "public"."event_attendees" from "anon";

revoke delete on table "public"."event_attendees" from "authenticated";

revoke insert on table "public"."event_attendees" from "authenticated";

revoke references on table "public"."event_attendees" from "authenticated";

revoke select on table "public"."event_attendees" from "authenticated";

revoke trigger on table "public"."event_attendees" from "authenticated";

revoke truncate on table "public"."event_attendees" from "authenticated";

revoke update on table "public"."event_attendees" from "authenticated";

revoke delete on table "public"."event_attendees" from "service_role";

revoke insert on table "public"."event_attendees" from "service_role";

revoke references on table "public"."event_attendees" from "service_role";

revoke select on table "public"."event_attendees" from "service_role";

revoke trigger on table "public"."event_attendees" from "service_role";

revoke truncate on table "public"."event_attendees" from "service_role";

revoke update on table "public"."event_attendees" from "service_role";

revoke delete on table "public"."event_skills" from "anon";

revoke insert on table "public"."event_skills" from "anon";

revoke references on table "public"."event_skills" from "anon";

revoke select on table "public"."event_skills" from "anon";

revoke trigger on table "public"."event_skills" from "anon";

revoke truncate on table "public"."event_skills" from "anon";

revoke update on table "public"."event_skills" from "anon";

revoke delete on table "public"."event_skills" from "authenticated";

revoke insert on table "public"."event_skills" from "authenticated";

revoke references on table "public"."event_skills" from "authenticated";

revoke select on table "public"."event_skills" from "authenticated";

revoke trigger on table "public"."event_skills" from "authenticated";

revoke truncate on table "public"."event_skills" from "authenticated";

revoke update on table "public"."event_skills" from "authenticated";

revoke delete on table "public"."event_skills" from "service_role";

revoke insert on table "public"."event_skills" from "service_role";

revoke references on table "public"."event_skills" from "service_role";

revoke select on table "public"."event_skills" from "service_role";

revoke trigger on table "public"."event_skills" from "service_role";

revoke truncate on table "public"."event_skills" from "service_role";

revoke update on table "public"."event_skills" from "service_role";

revoke delete on table "public"."event_topics" from "anon";

revoke insert on table "public"."event_topics" from "anon";

revoke references on table "public"."event_topics" from "anon";

revoke select on table "public"."event_topics" from "anon";

revoke trigger on table "public"."event_topics" from "anon";

revoke truncate on table "public"."event_topics" from "anon";

revoke update on table "public"."event_topics" from "anon";

revoke delete on table "public"."event_topics" from "authenticated";

revoke insert on table "public"."event_topics" from "authenticated";

revoke references on table "public"."event_topics" from "authenticated";

revoke select on table "public"."event_topics" from "authenticated";

revoke trigger on table "public"."event_topics" from "authenticated";

revoke truncate on table "public"."event_topics" from "authenticated";

revoke update on table "public"."event_topics" from "authenticated";

revoke delete on table "public"."event_topics" from "service_role";

revoke insert on table "public"."event_topics" from "service_role";

revoke references on table "public"."event_topics" from "service_role";

revoke select on table "public"."event_topics" from "service_role";

revoke trigger on table "public"."event_topics" from "service_role";

revoke truncate on table "public"."event_topics" from "service_role";

revoke update on table "public"."event_topics" from "service_role";

revoke delete on table "public"."events" from "anon";

revoke insert on table "public"."events" from "anon";

revoke references on table "public"."events" from "anon";

revoke select on table "public"."events" from "anon";

revoke trigger on table "public"."events" from "anon";

revoke truncate on table "public"."events" from "anon";

revoke update on table "public"."events" from "anon";

revoke delete on table "public"."events" from "authenticated";

revoke insert on table "public"."events" from "authenticated";

revoke references on table "public"."events" from "authenticated";

revoke select on table "public"."events" from "authenticated";

revoke trigger on table "public"."events" from "authenticated";

revoke truncate on table "public"."events" from "authenticated";

revoke update on table "public"."events" from "authenticated";

revoke delete on table "public"."events" from "service_role";

revoke insert on table "public"."events" from "service_role";

revoke references on table "public"."events" from "service_role";

revoke select on table "public"."events" from "service_role";

revoke trigger on table "public"."events" from "service_role";

revoke truncate on table "public"."events" from "service_role";

revoke update on table "public"."events" from "service_role";

revoke delete on table "public"."message_attachments" from "anon";

revoke insert on table "public"."message_attachments" from "anon";

revoke references on table "public"."message_attachments" from "anon";

revoke select on table "public"."message_attachments" from "anon";

revoke trigger on table "public"."message_attachments" from "anon";

revoke truncate on table "public"."message_attachments" from "anon";

revoke update on table "public"."message_attachments" from "anon";

revoke delete on table "public"."message_attachments" from "authenticated";

revoke insert on table "public"."message_attachments" from "authenticated";

revoke references on table "public"."message_attachments" from "authenticated";

revoke select on table "public"."message_attachments" from "authenticated";

revoke trigger on table "public"."message_attachments" from "authenticated";

revoke truncate on table "public"."message_attachments" from "authenticated";

revoke update on table "public"."message_attachments" from "authenticated";

revoke delete on table "public"."message_attachments" from "service_role";

revoke insert on table "public"."message_attachments" from "service_role";

revoke references on table "public"."message_attachments" from "service_role";

revoke select on table "public"."message_attachments" from "service_role";

revoke trigger on table "public"."message_attachments" from "service_role";

revoke truncate on table "public"."message_attachments" from "service_role";

revoke update on table "public"."message_attachments" from "service_role";

revoke delete on table "public"."messages" from "anon";

revoke insert on table "public"."messages" from "anon";

revoke references on table "public"."messages" from "anon";

revoke select on table "public"."messages" from "anon";

revoke trigger on table "public"."messages" from "anon";

revoke truncate on table "public"."messages" from "anon";

revoke update on table "public"."messages" from "anon";

revoke delete on table "public"."messages" from "authenticated";

revoke insert on table "public"."messages" from "authenticated";

revoke references on table "public"."messages" from "authenticated";

revoke select on table "public"."messages" from "authenticated";

revoke trigger on table "public"."messages" from "authenticated";

revoke truncate on table "public"."messages" from "authenticated";

revoke update on table "public"."messages" from "authenticated";

revoke delete on table "public"."messages" from "service_role";

revoke insert on table "public"."messages" from "service_role";

revoke references on table "public"."messages" from "service_role";

revoke select on table "public"."messages" from "service_role";

revoke trigger on table "public"."messages" from "service_role";

revoke truncate on table "public"."messages" from "service_role";

revoke update on table "public"."messages" from "service_role";

revoke delete on table "public"."profiles" from "anon";

revoke insert on table "public"."profiles" from "anon";

revoke references on table "public"."profiles" from "anon";

revoke select on table "public"."profiles" from "anon";

revoke trigger on table "public"."profiles" from "anon";

revoke truncate on table "public"."profiles" from "anon";

revoke update on table "public"."profiles" from "anon";

revoke delete on table "public"."profiles" from "authenticated";

revoke insert on table "public"."profiles" from "authenticated";

revoke references on table "public"."profiles" from "authenticated";

revoke select on table "public"."profiles" from "authenticated";

revoke trigger on table "public"."profiles" from "authenticated";

revoke truncate on table "public"."profiles" from "authenticated";

revoke update on table "public"."profiles" from "authenticated";

revoke delete on table "public"."profiles" from "service_role";

revoke insert on table "public"."profiles" from "service_role";

revoke references on table "public"."profiles" from "service_role";

revoke select on table "public"."profiles" from "service_role";

revoke trigger on table "public"."profiles" from "service_role";

revoke truncate on table "public"."profiles" from "service_role";

revoke update on table "public"."profiles" from "service_role";

revoke delete on table "public"."skills" from "anon";

revoke insert on table "public"."skills" from "anon";

revoke references on table "public"."skills" from "anon";

revoke select on table "public"."skills" from "anon";

revoke trigger on table "public"."skills" from "anon";

revoke truncate on table "public"."skills" from "anon";

revoke update on table "public"."skills" from "anon";

revoke delete on table "public"."skills" from "authenticated";

revoke insert on table "public"."skills" from "authenticated";

revoke references on table "public"."skills" from "authenticated";

revoke select on table "public"."skills" from "authenticated";

revoke trigger on table "public"."skills" from "authenticated";

revoke truncate on table "public"."skills" from "authenticated";

revoke update on table "public"."skills" from "authenticated";

revoke delete on table "public"."skills" from "service_role";

revoke insert on table "public"."skills" from "service_role";

revoke references on table "public"."skills" from "service_role";

revoke select on table "public"."skills" from "service_role";

revoke trigger on table "public"."skills" from "service_role";

revoke truncate on table "public"."skills" from "service_role";

revoke update on table "public"."skills" from "service_role";

revoke delete on table "public"."topics" from "anon";

revoke insert on table "public"."topics" from "anon";

revoke references on table "public"."topics" from "anon";

revoke select on table "public"."topics" from "anon";

revoke trigger on table "public"."topics" from "anon";

revoke truncate on table "public"."topics" from "anon";

revoke update on table "public"."topics" from "anon";

revoke delete on table "public"."topics" from "authenticated";

revoke insert on table "public"."topics" from "authenticated";

revoke references on table "public"."topics" from "authenticated";

revoke select on table "public"."topics" from "authenticated";

revoke trigger on table "public"."topics" from "authenticated";

revoke truncate on table "public"."topics" from "authenticated";

revoke update on table "public"."topics" from "authenticated";

revoke delete on table "public"."topics" from "service_role";

revoke insert on table "public"."topics" from "service_role";

revoke references on table "public"."topics" from "service_role";

revoke select on table "public"."topics" from "service_role";

revoke trigger on table "public"."topics" from "service_role";

revoke truncate on table "public"."topics" from "service_role";

revoke update on table "public"."topics" from "service_role";

revoke delete on table "public"."user_skills" from "anon";

revoke insert on table "public"."user_skills" from "anon";

revoke references on table "public"."user_skills" from "anon";

revoke select on table "public"."user_skills" from "anon";

revoke trigger on table "public"."user_skills" from "anon";

revoke truncate on table "public"."user_skills" from "anon";

revoke update on table "public"."user_skills" from "anon";

revoke delete on table "public"."user_skills" from "authenticated";

revoke insert on table "public"."user_skills" from "authenticated";

revoke references on table "public"."user_skills" from "authenticated";

revoke select on table "public"."user_skills" from "authenticated";

revoke trigger on table "public"."user_skills" from "authenticated";

revoke truncate on table "public"."user_skills" from "authenticated";

revoke update on table "public"."user_skills" from "authenticated";

revoke delete on table "public"."user_skills" from "service_role";

revoke insert on table "public"."user_skills" from "service_role";

revoke references on table "public"."user_skills" from "service_role";

revoke select on table "public"."user_skills" from "service_role";

revoke trigger on table "public"."user_skills" from "service_role";

revoke truncate on table "public"."user_skills" from "service_role";

revoke update on table "public"."user_skills" from "service_role";

alter table "public"."connections" drop constraint "connections_receiver_id_fkey";

alter table "public"."connections" drop constraint "connections_requester_id_fkey";

alter table "public"."connections" drop constraint "connections_requester_id_receiver_id_key";

alter table "public"."event_attendees" drop constraint "event_attendees_event_id_fkey";

alter table "public"."event_attendees" drop constraint "event_attendees_profile_id_fkey";

alter table "public"."event_skills" drop constraint "event_skills_event_id_fkey";

alter table "public"."event_skills" drop constraint "event_skills_skill_id_fkey";

alter table "public"."event_topics" drop constraint "event_topics_event_id_fkey";

alter table "public"."event_topics" drop constraint "event_topics_topic_id_fkey";

alter table "public"."events" drop constraint "events_organizer_id_fkey";

alter table "public"."message_attachments" drop constraint "message_attachments_message_id_fkey";

alter table "public"."messages" drop constraint "messages_receiver_id_fkey";

alter table "public"."messages" drop constraint "messages_sender_id_fkey";

alter table "public"."profiles" drop constraint "profiles_email_key";

alter table "public"."skills" drop constraint "skills_name_key";

alter table "public"."topics" drop constraint "topics_name_key";

alter table "public"."user_skills" drop constraint "user_skills_profile_id_fkey";

alter table "public"."user_skills" drop constraint "user_skills_skill_id_fkey";

drop function if exists "public"."calculate_match_score"(user1_id uuid, user2_id uuid);

drop function if exists "public"."get_recommended_connections"(p_user_id uuid, p_limit integer);

drop function if exists "public"."get_recommended_events"(p_user_id uuid, p_limit integer);

drop function if exists "public"."get_user_features"(user_id uuid);

alter table "public"."connections" drop constraint "connections_pkey";

alter table "public"."event_attendees" drop constraint "event_attendees_pkey";

alter table "public"."event_skills" drop constraint "event_skills_pkey";

alter table "public"."event_topics" drop constraint "event_topics_pkey";

alter table "public"."events" drop constraint "events_pkey";

alter table "public"."message_attachments" drop constraint "message_attachments_pkey";

alter table "public"."messages" drop constraint "messages_pkey";

alter table "public"."profiles" drop constraint "profiles_pkey";

alter table "public"."skills" drop constraint "skills_pkey";

alter table "public"."topics" drop constraint "topics_pkey";

alter table "public"."user_skills" drop constraint "user_skills_pkey";

drop index if exists "public"."connections_pkey";

drop index if exists "public"."connections_requester_id_receiver_id_key";

drop index if exists "public"."event_attendees_pkey";

drop index if exists "public"."event_skills_pkey";

drop index if exists "public"."event_topics_pkey";

drop index if exists "public"."events_pkey";

drop index if exists "public"."message_attachments_pkey";

drop index if exists "public"."messages_pkey";

drop index if exists "public"."profiles_email_key";

drop index if exists "public"."profiles_pkey";

drop index if exists "public"."skills_name_key";

drop index if exists "public"."skills_pkey";

drop index if exists "public"."topics_name_key";

drop index if exists "public"."topics_pkey";

drop index if exists "public"."user_skills_pkey";

drop table "public"."connections";

drop table "public"."event_attendees";

drop table "public"."event_skills";

drop table "public"."event_topics";

drop table "public"."events";

drop table "public"."message_attachments";

drop table "public"."messages";

drop table "public"."profiles";

drop table "public"."skills";

drop table "public"."topics";

drop table "public"."user_skills";

create table "public"."event_participants" (
    "id" uuid not null default uuid_generate_v4(),
    "event_id" uuid not null,
    "user_id" uuid not null,
    "status" text not null,
    "created_at" timestamp with time zone not null default timezone('utc'::text, now()),
    "updated_at" timestamp with time zone not null default timezone('utc'::text, now())
);


alter table "public"."event_participants" enable row level security;

create table "public"."event_registrations" (
    "id" uuid not null default uuid_generate_v4(),
    "event_id" uuid,
    "user_id" uuid,
    "created_at" timestamp with time zone default timezone('utc'::text, now())
);


alter table "public"."event_registrations" enable row level security;

create table "public"."group_members" (
    "id" uuid not null default uuid_generate_v4(),
    "group_id" uuid not null,
    "user_id" uuid not null,
    "role" text not null,
    "created_at" timestamp with time zone not null default timezone('utc'::text, now()),
    "updated_at" timestamp with time zone not null default timezone('utc'::text, now())
);


alter table "public"."group_members" enable row level security;

create table "public"."group_topics" (
    "id" uuid not null default uuid_generate_v4(),
    "group_id" uuid,
    "topic" text not null,
    "created_at" timestamp with time zone not null default timezone('utc'::text, now())
);


alter table "public"."group_topics" enable row level security;

create table "public"."groups" (
    "id" uuid not null default uuid_generate_v4(),
    "name" text not null,
    "description" text,
    "cover_image" text,
    "is_private" boolean default false,
    "created_by" uuid not null,
    "created_at" timestamp with time zone not null default timezone('utc'::text, now()),
    "updated_at" timestamp with time zone not null default timezone('utc'::text, now())
);


alter table "public"."groups" enable row level security;

create table "public"."matches" (
    "id" uuid not null default uuid_generate_v4(),
    "user_id" uuid not null,
    "matched_user_id" uuid not null,
    "score" double precision not null,
    "status" text not null,
    "created_at" timestamp with time zone not null default timezone('utc'::text, now()),
    "updated_at" timestamp with time zone not null default timezone('utc'::text, now())
);


alter table "public"."matches" enable row level security;

create table "public"."user_profiles" (
    "id" uuid not null,
    "name" text,
    "title" text,
    "company" text,
    "interests" text[] default '{}'::text[],
    "skills" text[] default '{}'::text[],
    "experience" text[] default '{}'::text[],
    "created_at" timestamp with time zone not null default timezone('utc'::text, now()),
    "updated_at" timestamp with time zone not null default timezone('utc'::text, now())
);


alter table "public"."user_profiles" enable row level security;

create table "public"."users" (
    "id" uuid not null,
    "email" text not null,
    "first_name" text not null,
    "last_name" text not null,
    "created_at" timestamp with time zone default timezone('utc'::text, now()),
    "updated_at" timestamp with time zone default timezone('utc'::text, now()),
    "avatar_url" text
);


drop type "public"."connection_status";

drop type "public"."event_format";

drop type "public"."skill_level";

drop type "public"."user_role";

CREATE UNIQUE INDEX event_participants_event_id_user_id_key ON public.event_participants USING btree (event_id, user_id);

CREATE UNIQUE INDEX event_participants_pkey ON public.event_participants USING btree (id);

CREATE UNIQUE INDEX event_registrations_event_id_user_id_key ON public.event_registrations USING btree (event_id, user_id);

CREATE UNIQUE INDEX event_registrations_pkey ON public.event_registrations USING btree (id);

CREATE UNIQUE INDEX group_members_group_id_user_id_key ON public.group_members USING btree (group_id, user_id);

CREATE UNIQUE INDEX group_members_pkey ON public.group_members USING btree (id);

CREATE UNIQUE INDEX group_topics_pkey ON public.group_topics USING btree (id);

CREATE UNIQUE INDEX groups_pkey ON public.groups USING btree (id);

CREATE INDEX idx_event_participants_event_id ON public.event_participants USING btree (event_id);

CREATE INDEX idx_event_participants_user_id ON public.event_participants USING btree (user_id);

CREATE INDEX idx_group_members_group_id ON public.group_members USING btree (group_id);

CREATE INDEX idx_group_members_user_id ON public.group_members USING btree (user_id);

CREATE INDEX idx_groups_created_by ON public.groups USING btree (created_by);

CREATE INDEX idx_matches_matched_user_id ON public.matches USING btree (matched_user_id);

CREATE INDEX idx_matches_user_id ON public.matches USING btree (user_id);

CREATE UNIQUE INDEX matches_pkey ON public.matches USING btree (id);

CREATE UNIQUE INDEX matches_user_id_matched_user_id_key ON public.matches USING btree (user_id, matched_user_id);

CREATE UNIQUE INDEX user_profiles_pkey ON public.user_profiles USING btree (id);

CREATE UNIQUE INDEX users_email_key ON public.users USING btree (email);

CREATE UNIQUE INDEX users_pkey ON public.users USING btree (id);

alter table "public"."event_participants" add constraint "event_participants_pkey" PRIMARY KEY using index "event_participants_pkey";

alter table "public"."event_registrations" add constraint "event_registrations_pkey" PRIMARY KEY using index "event_registrations_pkey";

alter table "public"."group_members" add constraint "group_members_pkey" PRIMARY KEY using index "group_members_pkey";

alter table "public"."group_topics" add constraint "group_topics_pkey" PRIMARY KEY using index "group_topics_pkey";

alter table "public"."groups" add constraint "groups_pkey" PRIMARY KEY using index "groups_pkey";

alter table "public"."matches" add constraint "matches_pkey" PRIMARY KEY using index "matches_pkey";

alter table "public"."user_profiles" add constraint "user_profiles_pkey" PRIMARY KEY using index "user_profiles_pkey";

alter table "public"."users" add constraint "users_pkey" PRIMARY KEY using index "users_pkey";

alter table "public"."event_participants" add constraint "event_participants_event_id_user_id_key" UNIQUE using index "event_participants_event_id_user_id_key";

alter table "public"."event_participants" add constraint "event_participants_status_check" CHECK ((status = ANY (ARRAY['going'::text, 'maybe'::text, 'declined'::text]))) not valid;

alter table "public"."event_participants" validate constraint "event_participants_status_check";

alter table "public"."event_registrations" add constraint "event_registrations_event_id_user_id_key" UNIQUE using index "event_registrations_event_id_user_id_key";

alter table "public"."event_registrations" add constraint "event_registrations_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."event_registrations" validate constraint "event_registrations_user_id_fkey";

alter table "public"."group_members" add constraint "group_members_group_id_fkey" FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE not valid;

alter table "public"."group_members" validate constraint "group_members_group_id_fkey";

alter table "public"."group_members" add constraint "group_members_group_id_user_id_key" UNIQUE using index "group_members_group_id_user_id_key";

alter table "public"."group_members" add constraint "group_members_role_check" CHECK ((role = ANY (ARRAY['admin'::text, 'member'::text]))) not valid;

alter table "public"."group_members" validate constraint "group_members_role_check";

alter table "public"."group_topics" add constraint "group_topics_group_id_fkey" FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE not valid;

alter table "public"."group_topics" validate constraint "group_topics_group_id_fkey";

alter table "public"."matches" add constraint "matches_status_check" CHECK ((status = ANY (ARRAY['pending'::text, 'accepted'::text, 'rejected'::text]))) not valid;

alter table "public"."matches" validate constraint "matches_status_check";

alter table "public"."matches" add constraint "matches_user_id_matched_user_id_key" UNIQUE using index "matches_user_id_matched_user_id_key";

alter table "public"."user_profiles" add constraint "user_profiles_id_fkey" FOREIGN KEY (id) REFERENCES auth.users(id) not valid;

alter table "public"."user_profiles" validate constraint "user_profiles_id_fkey";

alter table "public"."users" add constraint "users_email_key" UNIQUE using index "users_email_key";

alter table "public"."users" add constraint "users_id_fkey" FOREIGN KEY (id) REFERENCES auth.users(id) not valid;

alter table "public"."users" validate constraint "users_id_fkey";

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.calculate_match_score(user1_interests text[], user1_skills text[], user1_experience text[], user2_interests text[], user2_skills text[], user2_experience text[])
 RETURNS double precision
 LANGUAGE plpgsql
AS $function$
DECLARE
  interest_score float;
  skill_score float;
  experience_score float;
BEGIN
  -- Calculate interest similarity (40% weight)
  interest_score := (
    SELECT COUNT(*)::float / NULLIF(GREATEST(array_length(user1_interests, 1), array_length(user2_interests, 1)), 0)
    FROM unnest(user1_interests) i1
    WHERE i1 = ANY(user2_interests)
  ) * 0.4;

  -- Calculate skill similarity (40% weight)
  skill_score := (
    SELECT COUNT(*)::float / NULLIF(GREATEST(array_length(user1_skills, 1), array_length(user2_skills, 1)), 0)
    FROM unnest(user1_skills) s1
    WHERE s1 = ANY(user2_skills)
  ) * 0.4;

  -- Calculate experience similarity (20% weight)
  experience_score := (
    SELECT COUNT(*)::float / NULLIF(GREATEST(array_length(user1_experience, 1), array_length(user2_experience, 1)), 0)
    FROM unnest(user1_experience) e1
    WHERE e1 = ANY(user2_experience)
  ) * 0.2;

  RETURN COALESCE(interest_score, 0) + COALESCE(skill_score, 0) + COALESCE(experience_score, 0);
END;
$function$
;

CREATE OR REPLACE FUNCTION public.get_matches(user_id uuid)
 RETURNS TABLE(id uuid, name text, title text, company text, interests text[], skills text[], experience text[], match_score double precision)
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  user_interests text[];
  user_skills text[];
  user_experience text[];
BEGIN
  -- Get the user's profile data
  SELECT interests, skills, experience
  INTO user_interests, user_skills, user_experience
  FROM user_profiles
  WHERE id = user_id;

  -- Return matches
  RETURN QUERY
  SELECT 
    up.id,
    up.name,
    up.title,
    up.company,
    up.interests,
    up.skills,
    up.experience,
    CASE 
      WHEN up.interests IS NULL OR up.skills IS NULL OR up.experience IS NULL THEN 0
      ELSE (
        COALESCE(
          (
            SELECT COUNT(*)::float / NULLIF(GREATEST(array_length(up.interests, 1), array_length(user_interests, 1)), 0)
            FROM unnest(up.interests) i
            WHERE i = ANY(user_interests)
          ) * 0.4,
          0
        ) +
        COALESCE(
          (
            SELECT COUNT(*)::float / NULLIF(GREATEST(array_length(up.skills, 1), array_length(user_skills, 1)), 0)
            FROM unnest(up.skills) s
            WHERE s = ANY(user_skills)
          ) * 0.4,
          0
        ) +
        COALESCE(
          (
            SELECT COUNT(*)::float / NULLIF(GREATEST(array_length(up.experience, 1), array_length(user_experience, 1)), 0)
            FROM unnest(up.experience) e
            WHERE e = ANY(user_experience)
          ) * 0.2,
          0
        )
      )
    END as match_score
  FROM user_profiles up
  WHERE up.id != user_id
  ORDER BY match_score DESC
  LIMIT 10;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::text, NOW());
    RETURN NEW;
END;
$function$
;

grant delete on table "public"."event_participants" to "anon";

grant insert on table "public"."event_participants" to "anon";

grant references on table "public"."event_participants" to "anon";

grant select on table "public"."event_participants" to "anon";

grant trigger on table "public"."event_participants" to "anon";

grant truncate on table "public"."event_participants" to "anon";

grant update on table "public"."event_participants" to "anon";

grant delete on table "public"."event_participants" to "authenticated";

grant insert on table "public"."event_participants" to "authenticated";

grant references on table "public"."event_participants" to "authenticated";

grant select on table "public"."event_participants" to "authenticated";

grant trigger on table "public"."event_participants" to "authenticated";

grant truncate on table "public"."event_participants" to "authenticated";

grant update on table "public"."event_participants" to "authenticated";

grant delete on table "public"."event_participants" to "service_role";

grant insert on table "public"."event_participants" to "service_role";

grant references on table "public"."event_participants" to "service_role";

grant select on table "public"."event_participants" to "service_role";

grant trigger on table "public"."event_participants" to "service_role";

grant truncate on table "public"."event_participants" to "service_role";

grant update on table "public"."event_participants" to "service_role";

grant delete on table "public"."event_registrations" to "anon";

grant insert on table "public"."event_registrations" to "anon";

grant references on table "public"."event_registrations" to "anon";

grant select on table "public"."event_registrations" to "anon";

grant trigger on table "public"."event_registrations" to "anon";

grant truncate on table "public"."event_registrations" to "anon";

grant update on table "public"."event_registrations" to "anon";

grant delete on table "public"."event_registrations" to "authenticated";

grant insert on table "public"."event_registrations" to "authenticated";

grant references on table "public"."event_registrations" to "authenticated";

grant select on table "public"."event_registrations" to "authenticated";

grant trigger on table "public"."event_registrations" to "authenticated";

grant truncate on table "public"."event_registrations" to "authenticated";

grant update on table "public"."event_registrations" to "authenticated";

grant delete on table "public"."event_registrations" to "service_role";

grant insert on table "public"."event_registrations" to "service_role";

grant references on table "public"."event_registrations" to "service_role";

grant select on table "public"."event_registrations" to "service_role";

grant trigger on table "public"."event_registrations" to "service_role";

grant truncate on table "public"."event_registrations" to "service_role";

grant update on table "public"."event_registrations" to "service_role";

grant delete on table "public"."group_members" to "anon";

grant insert on table "public"."group_members" to "anon";

grant references on table "public"."group_members" to "anon";

grant select on table "public"."group_members" to "anon";

grant trigger on table "public"."group_members" to "anon";

grant truncate on table "public"."group_members" to "anon";

grant update on table "public"."group_members" to "anon";

grant delete on table "public"."group_members" to "authenticated";

grant insert on table "public"."group_members" to "authenticated";

grant references on table "public"."group_members" to "authenticated";

grant select on table "public"."group_members" to "authenticated";

grant trigger on table "public"."group_members" to "authenticated";

grant truncate on table "public"."group_members" to "authenticated";

grant update on table "public"."group_members" to "authenticated";

grant delete on table "public"."group_members" to "service_role";

grant insert on table "public"."group_members" to "service_role";

grant references on table "public"."group_members" to "service_role";

grant select on table "public"."group_members" to "service_role";

grant trigger on table "public"."group_members" to "service_role";

grant truncate on table "public"."group_members" to "service_role";

grant update on table "public"."group_members" to "service_role";

grant delete on table "public"."group_topics" to "anon";

grant insert on table "public"."group_topics" to "anon";

grant references on table "public"."group_topics" to "anon";

grant select on table "public"."group_topics" to "anon";

grant trigger on table "public"."group_topics" to "anon";

grant truncate on table "public"."group_topics" to "anon";

grant update on table "public"."group_topics" to "anon";

grant delete on table "public"."group_topics" to "authenticated";

grant insert on table "public"."group_topics" to "authenticated";

grant references on table "public"."group_topics" to "authenticated";

grant select on table "public"."group_topics" to "authenticated";

grant trigger on table "public"."group_topics" to "authenticated";

grant truncate on table "public"."group_topics" to "authenticated";

grant update on table "public"."group_topics" to "authenticated";

grant delete on table "public"."group_topics" to "service_role";

grant insert on table "public"."group_topics" to "service_role";

grant references on table "public"."group_topics" to "service_role";

grant select on table "public"."group_topics" to "service_role";

grant trigger on table "public"."group_topics" to "service_role";

grant truncate on table "public"."group_topics" to "service_role";

grant update on table "public"."group_topics" to "service_role";

grant delete on table "public"."groups" to "anon";

grant insert on table "public"."groups" to "anon";

grant references on table "public"."groups" to "anon";

grant select on table "public"."groups" to "anon";

grant trigger on table "public"."groups" to "anon";

grant truncate on table "public"."groups" to "anon";

grant update on table "public"."groups" to "anon";

grant delete on table "public"."groups" to "authenticated";

grant insert on table "public"."groups" to "authenticated";

grant references on table "public"."groups" to "authenticated";

grant select on table "public"."groups" to "authenticated";

grant trigger on table "public"."groups" to "authenticated";

grant truncate on table "public"."groups" to "authenticated";

grant update on table "public"."groups" to "authenticated";

grant delete on table "public"."groups" to "service_role";

grant insert on table "public"."groups" to "service_role";

grant references on table "public"."groups" to "service_role";

grant select on table "public"."groups" to "service_role";

grant trigger on table "public"."groups" to "service_role";

grant truncate on table "public"."groups" to "service_role";

grant update on table "public"."groups" to "service_role";

grant delete on table "public"."matches" to "anon";

grant insert on table "public"."matches" to "anon";

grant references on table "public"."matches" to "anon";

grant select on table "public"."matches" to "anon";

grant trigger on table "public"."matches" to "anon";

grant truncate on table "public"."matches" to "anon";

grant update on table "public"."matches" to "anon";

grant delete on table "public"."matches" to "authenticated";

grant insert on table "public"."matches" to "authenticated";

grant references on table "public"."matches" to "authenticated";

grant select on table "public"."matches" to "authenticated";

grant trigger on table "public"."matches" to "authenticated";

grant truncate on table "public"."matches" to "authenticated";

grant update on table "public"."matches" to "authenticated";

grant delete on table "public"."matches" to "service_role";

grant insert on table "public"."matches" to "service_role";

grant references on table "public"."matches" to "service_role";

grant select on table "public"."matches" to "service_role";

grant trigger on table "public"."matches" to "service_role";

grant truncate on table "public"."matches" to "service_role";

grant update on table "public"."matches" to "service_role";

grant delete on table "public"."user_profiles" to "anon";

grant insert on table "public"."user_profiles" to "anon";

grant references on table "public"."user_profiles" to "anon";

grant select on table "public"."user_profiles" to "anon";

grant trigger on table "public"."user_profiles" to "anon";

grant truncate on table "public"."user_profiles" to "anon";

grant update on table "public"."user_profiles" to "anon";

grant delete on table "public"."user_profiles" to "authenticated";

grant insert on table "public"."user_profiles" to "authenticated";

grant references on table "public"."user_profiles" to "authenticated";

grant select on table "public"."user_profiles" to "authenticated";

grant trigger on table "public"."user_profiles" to "authenticated";

grant truncate on table "public"."user_profiles" to "authenticated";

grant update on table "public"."user_profiles" to "authenticated";

grant delete on table "public"."user_profiles" to "service_role";

grant insert on table "public"."user_profiles" to "service_role";

grant references on table "public"."user_profiles" to "service_role";

grant select on table "public"."user_profiles" to "service_role";

grant trigger on table "public"."user_profiles" to "service_role";

grant truncate on table "public"."user_profiles" to "service_role";

grant update on table "public"."user_profiles" to "service_role";

grant delete on table "public"."users" to "anon";

grant insert on table "public"."users" to "anon";

grant references on table "public"."users" to "anon";

grant select on table "public"."users" to "anon";

grant trigger on table "public"."users" to "anon";

grant truncate on table "public"."users" to "anon";

grant update on table "public"."users" to "anon";

grant delete on table "public"."users" to "authenticated";

grant insert on table "public"."users" to "authenticated";

grant references on table "public"."users" to "authenticated";

grant select on table "public"."users" to "authenticated";

grant trigger on table "public"."users" to "authenticated";

grant truncate on table "public"."users" to "authenticated";

grant update on table "public"."users" to "authenticated";

grant delete on table "public"."users" to "service_role";

grant insert on table "public"."users" to "service_role";

grant references on table "public"."users" to "service_role";

grant select on table "public"."users" to "service_role";

grant trigger on table "public"."users" to "service_role";

grant truncate on table "public"."users" to "service_role";

grant update on table "public"."users" to "service_role";

create policy "Anyone can view event participants"
on "public"."event_participants"
as permissive
for select
to authenticated
using (true);


create policy "Authenticated users can join events"
on "public"."event_participants"
as permissive
for insert
to authenticated
with check (true);


create policy "Users can join events"
on "public"."event_participants"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can leave events"
on "public"."event_participants"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can view event participants"
on "public"."event_participants"
as permissive
for select
to public
using (true);


create policy "Enable all operations for authenticated users"
on "public"."event_registrations"
as permissive
for all
to authenticated
using (true)
with check (true);


create policy "Register for events"
on "public"."event_registrations"
as permissive
for insert
to authenticated
with check ((auth.uid() = user_id));


create policy "Unregister from events"
on "public"."event_registrations"
as permissive
for delete
to authenticated
using ((auth.uid() = user_id));


create policy "View event registrations"
on "public"."event_registrations"
as permissive
for select
to authenticated
using (true);


create policy "Join groups"
on "public"."group_members"
as permissive
for insert
to authenticated
with check (true);


create policy "Manage group members"
on "public"."group_members"
as permissive
for all
to authenticated
using ((EXISTS ( SELECT 1
   FROM group_members gm
  WHERE ((gm.group_id = group_members.group_id) AND (gm.user_id = auth.uid()) AND (gm.role = 'admin'::text)))));


create policy "Users can view group members"
on "public"."group_members"
as permissive
for select
to public
using ((EXISTS ( SELECT 1
   FROM groups
  WHERE ((groups.id = group_members.group_id) AND ((NOT groups.is_private) OR (EXISTS ( SELECT 1
           FROM group_members gm
          WHERE ((gm.group_id = groups.id) AND (gm.user_id = auth.uid())))))))));


create policy "View group members"
on "public"."group_members"
as permissive
for select
to authenticated
using (true);


create policy "Anyone can view group topics"
on "public"."group_topics"
as permissive
for select
to authenticated
using (true);


create policy "Group organizers can add topics"
on "public"."group_topics"
as permissive
for insert
to authenticated
with check ((EXISTS ( SELECT 1
   FROM group_members
  WHERE ((group_members.group_id = group_topics.group_id) AND (group_members.user_id = auth.uid()) AND (group_members.role = 'admin'::text)))));


create policy "Anyone can view groups"
on "public"."groups"
as permissive
for select
to authenticated
using (true);


create policy "Authenticated users can create groups"
on "public"."groups"
as permissive
for insert
to authenticated
with check (true);


create policy "Users can create groups"
on "public"."groups"
as permissive
for insert
to public
with check ((auth.uid() = created_by));


create policy "Users can view private groups they are members of"
on "public"."groups"
as permissive
for select
to public
using ((EXISTS ( SELECT 1
   FROM group_members
  WHERE ((group_members.group_id = groups.id) AND (group_members.user_id = auth.uid())))));


create policy "Users can view public groups"
on "public"."groups"
as permissive
for select
to public
using ((NOT is_private));


create policy "Users can update their own matches"
on "public"."matches"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view their own matches"
on "public"."matches"
as permissive
for select
to public
using (((auth.uid() = user_id) OR (auth.uid() = matched_user_id)));


create policy "Users can update their own profile"
on "public"."user_profiles"
as permissive
for update
to authenticated
using ((auth.uid() = id));


create policy "Users can view all profiles"
on "public"."user_profiles"
as permissive
for select
to authenticated
using (true);


CREATE TRIGGER update_event_participants_updated_at BEFORE UPDATE ON public.event_participants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_group_members_updated_at BEFORE UPDATE ON public.group_members FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_groups_updated_at BEFORE UPDATE ON public.groups FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_matches_updated_at BEFORE UPDATE ON public.matches FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


