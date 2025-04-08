-- Create enum types for various status fields
CREATE TYPE user_role AS ENUM ('user', 'admin');
CREATE TYPE event_format AS ENUM ('in_person', 'virtual', 'hybrid');
CREATE TYPE skill_level AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
CREATE TYPE connection_status AS ENUM ('pending', 'accepted', 'rejected');

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create profiles table
CREATE TABLE profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL UNIQUE,
    full_name TEXT,
    avatar_url TEXT,
    title TEXT,
    company TEXT,
    industry TEXT,
    bio TEXT,
    location TEXT,
    website TEXT,
    role user_role DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create skills table
CREATE TABLE skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create user_skills table
CREATE TABLE user_skills (
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES skills(id) ON DELETE CASCADE,
    level skill_level NOT NULL,
    years_of_experience INTEGER NOT NULL,
    PRIMARY KEY (profile_id, skill_id)
);

-- Create topics table
CREATE TABLE topics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create events table
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    description TEXT,
    format event_format NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    location TEXT,
    virtual_link TEXT,
    max_attendees INTEGER,
    organizer_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create event_skills table
CREATE TABLE event_skills (
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES skills(id) ON DELETE CASCADE,
    required_level skill_level NOT NULL,
    PRIMARY KEY (event_id, skill_id)
);

-- Create event_topics table
CREATE TABLE event_topics (
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES topics(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, topic_id)
);

-- Create event_attendees table
CREATE TABLE event_attendees (
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id, profile_id)
);

-- Create connections table
CREATE TABLE connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    requester_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    receiver_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    status connection_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (requester_id, receiver_id)
);

-- Create messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sender_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    receiver_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create message_attachments table
CREATE TABLE message_attachments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    url TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create functions
CREATE OR REPLACE FUNCTION get_user_features(user_id UUID)
RETURNS TABLE (
    skill_names TEXT[],
    interest_names TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ARRAY_AGG(DISTINCT s.name) AS skill_names,
        ARRAY_AGG(DISTINCT t.name) AS interest_names
    FROM profiles p
    LEFT JOIN user_skills us ON p.id = us.profile_id
    LEFT JOIN skills s ON us.skill_id = s.id
    LEFT JOIN event_topics et ON et.topic_id IN (
        SELECT topic_id 
        FROM event_attendees ea 
        WHERE ea.profile_id = p.id
    )
    LEFT JOIN topics t ON et.topic_id = t.id
    WHERE p.id = user_id
    GROUP BY p.id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION calculate_match_score(user1_id UUID, user2_id UUID)
RETURNS FLOAT AS $$
DECLARE
    common_skills INTEGER;
    common_interests INTEGER;
    total_skills INTEGER;
    total_interests INTEGER;
    match_score FLOAT;
BEGIN
    -- Count common skills
    SELECT COUNT(*) INTO common_skills
    FROM (
        SELECT skill_id FROM user_skills WHERE profile_id = user1_id
        INTERSECT
        SELECT skill_id FROM user_skills WHERE profile_id = user2_id
    ) common;

    -- Count total unique skills
    SELECT COUNT(*) INTO total_skills
    FROM (
        SELECT skill_id FROM user_skills WHERE profile_id = user1_id
        UNION
        SELECT skill_id FROM user_skills WHERE profile_id = user2_id
    ) total;

    -- Count common interests (based on event topics they've attended)
    SELECT COUNT(*) INTO common_interests
    FROM (
        SELECT DISTINCT et.topic_id
        FROM event_attendees ea
        JOIN event_topics et ON ea.event_id = et.event_id
        WHERE ea.profile_id = user1_id
        INTERSECT
        SELECT DISTINCT et.topic_id
        FROM event_attendees ea
        JOIN event_topics et ON ea.event_id = et.event_id
        WHERE ea.profile_id = user2_id
    ) common;

    -- Count total unique interests
    SELECT COUNT(*) INTO total_interests
    FROM (
        SELECT DISTINCT et.topic_id
        FROM event_attendees ea
        JOIN event_topics et ON ea.event_id = et.event_id
        WHERE ea.profile_id IN (user1_id, user2_id)
    ) total;

    -- Calculate match score (50% skills, 50% interests)
    match_score := (
        CASE 
            WHEN total_skills = 0 THEN 0 
            ELSE (common_skills::FLOAT / total_skills) * 0.5 
        END +
        CASE 
            WHEN total_interests = 0 THEN 0 
            ELSE (common_interests::FLOAT / total_interests) * 0.5 
        END
    ) * 100;

    RETURN match_score;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_recommended_connections(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    name TEXT,
    title TEXT,
    company TEXT,
    match_score FLOAT,
    mutual_connections INTEGER,
    skills TEXT[],
    interests TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH potential_connections AS (
        SELECT 
            p.id,
            p.full_name AS name,
            p.title,
            p.company,
            calculate_match_score(p_user_id, p.id) AS match_score,
            COUNT(DISTINCT c2.receiver_id) AS mutual_connections,
            ARRAY_AGG(DISTINCT s.name) AS skills,
            ARRAY_AGG(DISTINCT t.name) AS interests
        FROM profiles p
        LEFT JOIN connections c1 ON (c1.requester_id = p_user_id AND c1.receiver_id = p.id)
            OR (c1.receiver_id = p_user_id AND c1.requester_id = p.id)
        LEFT JOIN connections c2 ON (c2.requester_id = p.id OR c2.receiver_id = p.id)
            AND c2.status = 'accepted'
        LEFT JOIN user_skills us ON p.id = us.profile_id
        LEFT JOIN skills s ON us.skill_id = s.id
        LEFT JOIN event_attendees ea ON p.id = ea.profile_id
        LEFT JOIN event_topics et ON ea.event_id = et.event_id
        LEFT JOIN topics t ON et.topic_id = t.id
        WHERE p.id != p_user_id
            AND c1.id IS NULL -- Not already connected
        GROUP BY p.id, p.full_name, p.title, p.company
    )
    SELECT *
    FROM potential_connections
    ORDER BY match_score DESC, mutual_connections DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_recommended_events(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    description TEXT,
    date TIMESTAMP WITH TIME ZONE,
    format event_format,
    location TEXT,
    match_score FLOAT,
    topics TEXT[],
    required_skills TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH user_skills AS (
        SELECT ARRAY_AGG(s.name) AS skills
        FROM user_skills us
        JOIN skills s ON us.skill_id = s.id
        WHERE us.profile_id = p_user_id
    ),
    user_topics AS (
        SELECT ARRAY_AGG(DISTINCT t.name) AS topics
        FROM event_attendees ea
        JOIN event_topics et ON ea.event_id = et.event_id
        JOIN topics t ON et.topic_id = t.id
        WHERE ea.profile_id = p_user_id
    ),
    event_matches AS (
        SELECT 
            e.id,
            e.title,
            e.description,
            e.date,
            e.format,
            e.location,
            (
                CASE 
                    WHEN us.skills IS NULL OR es_agg.skills IS NULL THEN 0
                    ELSE ARRAY_LENGTH(ARRAY(
                        SELECT UNNEST(us.skills)
                        INTERSECT
                        SELECT UNNEST(es_agg.skills)
                    ), 1)::FLOAT / GREATEST(ARRAY_LENGTH(us.skills, 1), ARRAY_LENGTH(es_agg.skills, 1))
                END * 0.6 +
                CASE 
                    WHEN ut.topics IS NULL OR et_agg.topics IS NULL THEN 0
                    ELSE ARRAY_LENGTH(ARRAY(
                        SELECT UNNEST(ut.topics)
                        INTERSECT
                        SELECT UNNEST(et_agg.topics)
                    ), 1)::FLOAT / GREATEST(ARRAY_LENGTH(ut.topics, 1), ARRAY_LENGTH(et_agg.topics, 1))
                END * 0.4
            ) * 100 AS match_score,
            et_agg.topics,
            es_agg.skills AS required_skills
        FROM events e
        CROSS JOIN user_skills us
        CROSS JOIN user_topics ut
        LEFT JOIN LATERAL (
            SELECT ARRAY_AGG(t.name) AS topics
            FROM event_topics et
            JOIN topics t ON et.topic_id = t.id
            WHERE et.event_id = e.id
        ) et_agg ON true
        LEFT JOIN LATERAL (
            SELECT ARRAY_AGG(s.name) AS skills
            FROM event_skills es
            JOIN skills s ON es.skill_id = s.id
            WHERE es.event_id = e.id
        ) es_agg ON true
        WHERE e.date >= CURRENT_TIMESTAMP
            AND NOT EXISTS (
                SELECT 1
                FROM event_attendees ea
                WHERE ea.event_id = e.id
                AND ea.profile_id = p_user_id
            )
    )
    SELECT *
    FROM event_matches
    ORDER BY match_score DESC, date ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_events_updated_at
    BEFORE UPDATE ON events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_connections_updated_at
    BEFORE UPDATE ON connections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at
    BEFORE UPDATE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 