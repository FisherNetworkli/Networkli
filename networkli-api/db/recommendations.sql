-- Function to get user's skills and interests
create or replace function public.get_user_features(user_id uuid)
returns table (
    skill_names text[],
    interest_names text[]
) as $$
begin
    return query
    select
        array_agg(distinct s.name) as skill_names,
        array_agg(distinct i.name) as interest_names
    from public.profiles p
    left join public.user_skills us on us.user_id = p.id
    left join public.skills s on s.id = us.skill_id
    left join public.user_interests ui on ui.user_id = p.id
    left join public.interests i on i.id = ui.interest_id
    where p.id = user_id
    group by p.id;
end;
$$ language plpgsql security definer;

-- Function to calculate match score between users
create or replace function public.calculate_match_score(
    user1_id uuid,
    user2_id uuid
)
returns float as $$
declare
    user1_skills text[];
    user1_interests text[];
    user2_skills text[];
    user2_interests text[];
    skill_match float;
    interest_match float;
    mutual_connections int;
begin
    -- Get user features
    select * from public.get_user_features(user1_id)
    into user1_skills, user1_interests;
    
    select * from public.get_user_features(user2_id)
    into user2_skills, user2_interests;
    
    -- Calculate skill match
    select count(*)::float / nullif(array_length(user1_skills, 1) + array_length(user2_skills, 1), 0)
    from unnest(user1_skills) skill
    where skill = any(user2_skills)
    into skill_match;
    
    -- Calculate interest match
    select count(*)::float / nullif(array_length(user1_interests, 1) + array_length(user2_interests, 1), 0)
    from unnest(user1_interests) interest
    where interest = any(user2_interests)
    into interest_match;
    
    -- Get mutual connections
    select count(*)
    from public.connections c1
    join public.connections c2 on c2.connected_user_id = c1.user_id
    where c1.user_id = user1_id
    and c2.user_id = user2_id
    and c1.status = 'accepted'
    and c2.status = 'accepted'
    into mutual_connections;
    
    -- Calculate final score (weighted average)
    return (
        coalesce(skill_match, 0) * 0.4 +
        coalesce(interest_match, 0) * 0.4 +
        least(mutual_connections::float / 10, 1) * 0.2
    );
end;
$$ language plpgsql security definer;

-- Function to get recommended connections
create or replace function public.get_recommended_connections(
    p_user_id uuid,
    p_limit int default 10
)
returns table (
    id uuid,
    name text,
    title text,
    company text,
    match_score float,
    mutual_connections int,
    skills text[],
    interests text[]
) as $$
begin
    return query
    with potential_matches as (
        select
            p.id,
            p.name,
            p.title,
            p.company,
            public.calculate_match_score(p_user_id, p.id) as match_score,
            (
                select count(*)
                from public.connections c1
                join public.connections c2 on c2.connected_user_id = c1.user_id
                where c1.user_id = p_user_id
                and c2.user_id = p.id
                and c1.status = 'accepted'
                and c2.status = 'accepted'
            ) as mutual_connections,
            array_agg(distinct s.name) as skills,
            array_agg(distinct i.name) as interests
        from public.profiles p
        left join public.user_skills us on us.user_id = p.id
        left join public.skills s on s.id = us.skill_id
        left join public.user_interests ui on ui.user_id = p.id
        left join public.interests i on i.id = ui.interest_id
        where p.id != p_user_id
        and not exists (
            select 1 from public.connections c
            where (c.user_id = p_user_id and c.connected_user_id = p.id)
            or (c.user_id = p.id and c.connected_user_id = p_user_id)
        )
        group by p.id, p.name, p.title, p.company
    )
    select *
    from potential_matches
    where match_score > 0
    order by match_score desc
    limit p_limit;
end;
$$ language plpgsql security definer;

-- Function to get recommended events
create or replace function public.get_recommended_events(
    p_user_id uuid,
    p_limit int default 5
)
returns table (
    id uuid,
    title text,
    description text,
    date timestamp with time zone,
    format text,
    location text,
    match_score float,
    topics text[],
    required_skills text[]
) as $$
declare
    user_skills text[];
    user_interests text[];
begin
    -- Get user features
    select * from public.get_user_features(p_user_id)
    into user_skills, user_interests;
    
    return query
    select
        e.id,
        e.title,
        e.description,
        e.date,
        e.format,
        e.location,
        (
            -- Calculate match score based on topics and skills
            (
                array_length(
                    array(
                        select unnest(array_agg(i.name))
                        intersect
                        select unnest(user_interests)
                    ),
                    1
                )::float /
                nullif(array_length(array_agg(i.name), 1) + array_length(user_interests, 1), 0)
            ) * 0.6 +
            (
                array_length(
                    array(
                        select unnest(array_agg(s.name))
                        intersect
                        select unnest(user_skills)
                    ),
                    1
                )::float /
                nullif(array_length(array_agg(s.name), 1) + array_length(user_skills, 1), 0)
            ) * 0.4
        ) as match_score,
        array_agg(distinct i.name) as topics,
        array_agg(distinct s.name) as required_skills
    from public.events e
    left join public.event_topics et on et.event_id = e.id
    left join public.interests i on i.id = et.topic_id
    left join public.event_required_skills ers on ers.event_id = e.id
    left join public.skills s on s.id = ers.skill_id
    where e.date >= current_timestamp
    group by e.id, e.title, e.description, e.date, e.format, e.location
    having (
        array_length(
            array(
                select unnest(array_agg(i.name))
                intersect
                select unnest(user_interests)
            ),
            1
        ) > 0
        or
        array_length(
            array(
                select unnest(array_agg(s.name))
                intersect
                select unnest(user_skills)
            ),
            1
        ) > 0
    )
    order by match_score desc, e.date asc
    limit p_limit;
end;
$$ language plpgsql security definer;

-- Function to get recommended groups
create or replace function public.get_recommended_groups(
    p_user_id uuid,
    p_limit int default 5
)
returns table (
    id uuid,
    name text,
    description text,
    industry text,
    member_count bigint,
    match_score float,
    focus_areas text[],
    relevant_skills text[]
) as $$
declare
    user_skills text[];
    user_interests text[];
    user_industry text;
begin
    -- Get user features
    select * from public.get_user_features(p_user_id)
    into user_skills, user_interests;
    
    -- Get user industry
    select industry from public.profiles where id = p_user_id into user_industry;
    
    return query
    select
        g.id,
        g.name,
        g.description,
        g.industry,
        count(distinct gm.user_id) as member_count,
        (
            -- Calculate match score based on focus areas, skills, and industry
            (
                array_length(
                    array(
                        select unnest(array_agg(distinct i.name))
                        intersect
                        select unnest(user_interests)
                    ),
                    1
                )::float /
                nullif(array_length(array_agg(distinct i.name), 1) + array_length(user_interests, 1), 0)
            ) * 0.4 +
            (
                array_length(
                    array(
                        select unnest(array_agg(distinct s.name))
                        intersect
                        select unnest(user_skills)
                    ),
                    1
                )::float /
                nullif(array_length(array_agg(distinct s.name), 1) + array_length(user_skills, 1), 0)
            ) * 0.3 +
            case when g.industry = user_industry then 0.3 else 0 end
        ) as match_score,
        array_agg(distinct i.name) as focus_areas,
        array_agg(distinct s.name) as relevant_skills
    from public.groups g
    left join public.group_focus_areas gfa on gfa.group_id = g.id
    left join public.interests i on i.id = gfa.interest_id
    left join public.group_relevant_skills grs on grs.group_id = g.id
    left join public.skills s on s.id = grs.skill_id
    left join public.group_members gm on gm.group_id = g.id
    where not exists (
        select 1 from public.group_members
        where group_id = g.id and user_id = p_user_id
    )
    group by g.id, g.name, g.description, g.industry
    having (
        array_length(
            array(
                select unnest(array_agg(distinct i.name))
                intersect
                select unnest(user_interests)
            ),
            1
        ) > 0
        or
        array_length(
            array(
                select unnest(array_agg(distinct s.name))
                intersect
                select unnest(user_skills)
            ),
            1
        ) > 0
        or
        g.industry = user_industry
    )
    order by match_score desc
    limit p_limit;
end;
$$ language plpgsql security definer; 