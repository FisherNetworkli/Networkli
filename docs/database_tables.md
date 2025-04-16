# Networkli Database Schema Documentation

This document outlines the structure of the database tables used in the Networkli platform, with a focus on tables required for the recommendation system and interaction tracking.

## Core Tables

### `profiles`

Stores user profile information and is the central table for user data.

```sql
CREATE TABLE public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id),
    email TEXT,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT,
    avatar_url TEXT,
    title TEXT,
    company TEXT,
    industry TEXT,
    bio TEXT,
    location TEXT,
    website TEXT,
    role TEXT DEFAULT 'user',
    linkedin_url TEXT,
    github_url TEXT,
    portfolio_url TEXT,
    twitter_url TEXT,
    expertise TEXT,
    needs TEXT,
    meaningful_goals TEXT,
    skills TEXT[],
    interests TEXT[],
    experience_level TEXT,
    domain_id TEXT,
    profile_visibility TEXT DEFAULT 'public',
    is_premium BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    bio_embedding VECTOR(384),
    expertise_embedding VECTOR(384),
    needs_embedding VECTOR(384),
    goals_embedding VECTOR(384)
);
```

### `connections`

Records the connections between users, including pending connection requests.

```sql
CREATE TABLE public.connections (
    id SERIAL PRIMARY KEY,
    requester_id UUID REFERENCES public.profiles(id),
    receiver_id UUID REFERENCES public.profiles(id),
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(requester_id, receiver_id)
);
```

### `profile_views`

Records instances when one user views another user's profile.

```sql
CREATE TABLE public.profile_views (
    id SERIAL PRIMARY KEY,
    profile_id UUID REFERENCES public.profiles(id),
    visitor_id UUID REFERENCES public.profiles(id),
    view_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source TEXT,
    referrer TEXT,
    metadata JSONB
);
```

## Interaction Tracking

### `interaction_history`

Central table for tracking all user interactions on the platform.

```sql
CREATE TABLE public.interaction_history (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id),
    interaction_type TEXT NOT NULL,
    target_entity_type TEXT,
    target_entity_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create index for faster querying by user and type
CREATE INDEX idx_interaction_history_user_type 
ON public.interaction_history(user_id, interaction_type);
```

The `interaction_type` field can have values like:
- `PROFILE_VIEW` - Viewing another user's profile
- `RECOMMENDATION_CLICK` - Clicking on a recommended user
- `RECOMMENDATION_LIKE` - Liking/accepting a recommendation (swiping right)
- `RECOMMENDATION_PASS` - Passing on a recommendation (swiping left)
- `SEARCH` - Performing a search
- `EVENT_INTERACTION` - Interacting with an event
- `GROUP_INTERACTION` - Interacting with a group

### `search_history`

Records user search queries for analysis and recommendation improvement.

```sql
CREATE TABLE public.search_history (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id),
    query TEXT,
    filters JSONB,
    result_count INTEGER,
    search_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Recommendation System

### `recommendations`

Stores precalculated recommendations for quick retrieval.

```sql
CREATE TABLE public.recommendations (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id),
    recommended_id UUID REFERENCES public.profiles(id),
    score FLOAT,
    reasons JSONB,
    algorithm_version TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, recommended_id)
);
```

### `recommendation_clicks`

Records when users click on or interact with recommendations.

```sql
CREATE TABLE public.recommendation_clicks (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id),
    recommendation_id INTEGER REFERENCES public.recommendations(id),
    click_type TEXT,
    result TEXT,
    source_page TEXT,
    rank INTEGER,
    clicked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
```

## Organization Tables

### `groups`

Stores information about professional networking groups.

```sql
CREATE TABLE public.groups (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    organizer_id UUID REFERENCES public.profiles(id),
    category TEXT,
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### `group_members`

Tracks group membership and roles within groups.

```sql
CREATE TABLE public.group_members (
    id SERIAL PRIMARY KEY,
    group_id INTEGER REFERENCES public.groups(id),
    member_id UUID REFERENCES public.profiles(id),
    role TEXT DEFAULT 'member',
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(group_id, member_id)
);
```

### `events`

Stores information about networking events.

```sql
CREATE TABLE public.events (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    organizer_id UUID REFERENCES public.profiles(id),
    group_id INTEGER REFERENCES public.groups(id),
    location TEXT,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### `event_attendees`

Tracks event attendance and RSVPs.

```sql
CREATE TABLE public.event_attendees (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES public.events(id),
    attendee_id UUID REFERENCES public.profiles(id),
    status TEXT DEFAULT 'going',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, attendee_id)
);
```

## Feature Development Tables

### `user_preferences`

Stores user preferences used for personalization and recommendations.

```sql
CREATE TABLE public.user_preferences (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) UNIQUE,
    interests TEXT[],
    networking_style TEXT,
    notification_preferences JSONB,
    discovery_settings JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### `skill_taxonomy`

Formalizes the skill structure for better matching.

```sql
CREATE TABLE public.skill_taxonomy (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE,
    category TEXT,
    parent_id INTEGER REFERENCES public.skill_taxonomy(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### `interest_categories`

Formalizes interest categories for better matching.

```sql
CREATE TABLE public.interest_categories (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE,
    parent_id INTEGER REFERENCES public.interest_categories(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Database Extensions

For advanced features like vector similarity search, we use the following extensions:

```sql
-- Enable vector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Enable pgcrypto for UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

## Indexing Strategy

To ensure good performance, we implement the following indexes:

```sql
-- Indexes for profiles
CREATE INDEX idx_profiles_skills ON public.profiles USING GIN (skills);
CREATE INDEX idx_profiles_interests ON public.profiles USING GIN (interests);
CREATE INDEX idx_profiles_domain ON public.profiles(domain_id);
CREATE INDEX idx_profiles_industry ON public.profiles(industry);
CREATE INDEX idx_profiles_location ON public.profiles(location);

-- Indexes for vector embeddings
CREATE INDEX idx_profiles_bio_embedding ON public.profiles USING ivfflat (bio_embedding vector_cosine_ops);
CREATE INDEX idx_profiles_expertise_embedding ON public.profiles USING ivfflat (expertise_embedding vector_cosine_ops);

-- Indexes for connections
CREATE INDEX idx_connections_requester ON public.connections(requester_id);
CREATE INDEX idx_connections_receiver ON public.connections(receiver_id);
CREATE INDEX idx_connections_status ON public.connections(status);

-- Indexes for recommendations
CREATE INDEX idx_recommendations_user ON public.recommendations(user_id);
CREATE INDEX idx_recommendations_recommended ON public.recommendations(recommended_id);
CREATE INDEX idx_recommendations_score ON public.recommendations(score DESC);

-- Indexes for interaction_history
CREATE INDEX idx_interaction_history_created_at ON public.interaction_history(created_at);
CREATE INDEX idx_interaction_history_target ON public.interaction_history(target_entity_type, target_entity_id);
```

## Implementation Status

- **Fully Implemented**:
  - `profiles`
  - `connections`
  - `profile_views`
  - `interaction_history`
  - `groups`
  
- **Partially Implemented**:
  - `recommendations`
  - `events`
  - `event_attendees`
  
- **Planned**:
  - `skill_taxonomy`
  - `interest_categories`
  - Vector embeddings for profile fields

## Next Steps

1. **Add Vector Embeddings**: Implement BERT embeddings storage in the `profiles` table.
2. **Formalize Skills**: Implement the `skill_taxonomy` table and migrate existing skills.
3. **Enhance Interaction Logging**: Complete the implementation of all interaction types.
4. **Recommendation Storage**: Fully implement the recommendation caching system. 