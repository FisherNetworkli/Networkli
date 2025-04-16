-- Enable necessary extensions
create extension if not exists "uuid-ossp";
create extension if not exists "pgcrypto";

-- Profiles table (extends Supabase auth.users)
create table public.profiles (
    id uuid references auth.users primary key,
    email text unique not null,
    name text,
    title text,
    company text,
    industry text,
    bio text,
    avatar_url text,
    preferred_event_format text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Skills table
create table public.skills (
    id uuid default uuid_generate_v4() primary key,
    name text unique not null
);

-- User Skills (junction table)
create table public.user_skills (
    user_id uuid references public.profiles(id) on delete cascade,
    skill_id uuid references public.skills(id) on delete cascade,
    primary key (user_id, skill_id)
);

-- Interests table
create table public.interests (
    id uuid default uuid_generate_v4() primary key,
    name text unique not null
);

-- User Interests (junction table)
create table public.user_interests (
    user_id uuid references public.profiles(id) on delete cascade,
    interest_id uuid references public.interests(id) on delete cascade,
    primary key (user_id, interest_id)
);

-- Connections table
create table public.connections (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references public.profiles(id) on delete cascade,
    connected_user_id uuid references public.profiles(id) on delete cascade,
    status text check (status in ('pending', 'accepted', 'rejected')) not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(user_id, connected_user_id)
);

-- Events table
create table public.events (
    id uuid default uuid_generate_v4() primary key,
    title text not null,
    description text,
    date timestamp with time zone not null,
    end_date timestamp with time zone,
    location text,
    format text check (format in ('in_person', 'virtual', 'hybrid')) not null,
    organizer_id uuid references public.profiles(id) on delete set null,
    max_attendees integer,
    image_url text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Event Topics (junction table)
create table public.event_topics (
    event_id uuid references public.events(id) on delete cascade,
    topic_id uuid references public.interests(id) on delete cascade,
    primary key (event_id, topic_id)
);

-- Event Required Skills (junction table)
create table public.event_required_skills (
    event_id uuid references public.events(id) on delete cascade,
    skill_id uuid references public.skills(id) on delete cascade,
    primary key (event_id, skill_id)
);

-- Event Attendees
create table public.event_attendees (
    event_id uuid references public.events(id) on delete cascade,
    user_id uuid references public.profiles(id) on delete cascade,
    status text check (status in ('registered', 'waitlisted', 'attended', 'cancelled')) not null,
    registered_at timestamp with time zone default timezone('utc'::text, now()) not null,
    primary key (event_id, user_id)
);

-- Groups table
create table public.groups (
    id uuid default uuid_generate_v4() primary key,
    name text not null,
    description text,
    industry text,
    image_url text,
    created_by uuid references public.profiles(id) on delete set null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Group Focus Areas (junction table)
create table public.group_focus_areas (
    group_id uuid references public.groups(id) on delete cascade,
    interest_id uuid references public.interests(id) on delete cascade,
    primary key (group_id, interest_id)
);

-- Group Relevant Skills (junction table)
create table public.group_relevant_skills (
    group_id uuid references public.groups(id) on delete cascade,
    skill_id uuid references public.skills(id) on delete cascade,
    primary key (group_id, skill_id)
);

-- Group Members
create table public.group_members (
    group_id uuid references public.groups(id) on delete cascade,
    user_id uuid references public.profiles(id) on delete cascade,
    role text check (role in ('admin', 'moderator', 'member')) not null default 'member',
    joined_at timestamp with time zone default timezone('utc'::text, now()) not null,
    primary key (group_id, user_id)
);

-- Messages table
create table public.messages (
    id uuid default uuid_generate_v4() primary key,
    sender_id uuid references public.profiles(id) on delete set null,
    receiver_id uuid references public.profiles(id) on delete set null,
    content text not null,
    read boolean default false not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Message Attachments
create table public.message_attachments (
    id uuid default uuid_generate_v4() primary key,
    message_id uuid references public.messages(id) on delete cascade,
    type text check (type in ('image', 'file')) not null,
    url text not null,
    name text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes for better query performance
create index idx_connections_user_id on public.connections(user_id);
create index idx_connections_connected_user_id on public.connections(connected_user_id);
create index idx_events_date on public.events(date);
create index idx_messages_sender_receiver on public.messages(sender_id, receiver_id);
create index idx_messages_created_at on public.messages(created_at);
create index idx_group_members_user_id on public.group_members(user_id);

-- Create functions for real-time features
create or replace function public.handle_new_user()
returns trigger as $$
begin
    insert into public.profiles (id, email)
    values (new.id, new.email);
    return new;
end;
$$ language plpgsql security definer;

-- Trigger for new user creation
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();

-- Function to update updated_at timestamp
create or replace function public.update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

-- Add updated_at triggers to relevant tables
create trigger update_profiles_updated_at
    before update on public.profiles
    for each row execute procedure public.update_updated_at_column();

create trigger update_events_updated_at
    before update on public.events
    for each row execute procedure public.update_updated_at_column();

create trigger update_groups_updated_at
    before update on public.groups
    for each row execute procedure public.update_updated_at_column();

-- RLS Policies
alter table public.profiles enable row level security;
alter table public.connections enable row level security;
alter table public.events enable row level security;
alter table public.groups enable row level security;
alter table public.messages enable row level security;

-- Profiles policy
create policy "Public profiles are viewable by everyone"
    on public.profiles for select
    using (true);

create policy "Users can update own profile"
    on public.profiles for update
    using (auth.uid() = id);

-- Connections policy
create policy "Users can view their own connections"
    on public.connections for select
    using (auth.uid() = user_id or auth.uid() = connected_user_id);

create policy "Users can create their own connections"
    on public.connections for insert
    with check (auth.uid() = user_id);

-- Events policy
create policy "Events are viewable by everyone"
    on public.events for select
    using (true);

create policy "Organizers can update their events"
    on public.events for update
    using (auth.uid() = organizer_id);

-- Groups policy
create policy "Groups are viewable by everyone"
    on public.groups for select
    using (true);

create policy "Group admins can update their groups"
    on public.groups for update
    using (
        exists (
            select 1 from public.group_members
            where group_id = id
            and user_id = auth.uid()
            and role = 'admin'
        )
    );

-- Messages policy
create policy "Users can view their own messages"
    on public.messages for select
    using (auth.uid() = sender_id or auth.uid() = receiver_id);

create policy "Users can send messages"
    on public.messages for insert
    with check (auth.uid() = sender_id); 