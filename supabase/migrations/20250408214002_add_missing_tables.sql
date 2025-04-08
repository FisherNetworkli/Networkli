-- Create events table
CREATE TABLE IF NOT EXISTS public.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    format TEXT NOT NULL CHECK (format IN ('in_person', 'virtual', 'hybrid')),
    date TIMESTAMPTZ NOT NULL,
    location TEXT,
    virtual_link TEXT,
    max_attendees INTEGER,
    organizer_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create event_skills table
CREATE TABLE IF NOT EXISTS public.event_skills (
    event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES public.skills(id) ON DELETE CASCADE,
    required_level INTEGER NOT NULL DEFAULT 1 CHECK (required_level BETWEEN 1 AND 5),
    PRIMARY KEY (event_id, skill_id)
);

-- Create event_topics table
CREATE TABLE IF NOT EXISTS public.event_topics (
    event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
    topic_id UUID NOT NULL REFERENCES public.topics(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, topic_id)
);

-- Create event_attendees table
CREATE TABLE IF NOT EXISTS public.event_attendees (
    event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
    profile_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('registered', 'waitlisted', 'cancelled')),
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_id, profile_id)
);

-- Create connections table
CREATE TABLE IF NOT EXISTS public.connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requester_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    receiver_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(requester_id, receiver_id)
);

-- Create messages table
CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sender_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    receiver_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create message_attachments table
CREATE TABLE IF NOT EXISTS public.message_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES public.messages(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    url TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_events_updated_at
    BEFORE UPDATE ON public.events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_connections_updated_at
    BEFORE UPDATE ON public.connections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at
    BEFORE UPDATE ON public.messages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add RLS policies
ALTER TABLE public.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.event_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.event_topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.event_attendees ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.message_attachments ENABLE ROW LEVEL SECURITY;

-- Events are viewable by all authenticated users
CREATE POLICY "Events are viewable by all authenticated users"
ON public.events FOR SELECT
TO authenticated
USING (true);

-- Event organizers can manage their events
CREATE POLICY "Event organizers can manage their events"
ON public.events
FOR ALL TO authenticated
USING (organizer_id = auth.uid())
WITH CHECK (organizer_id = auth.uid());

-- Event skills and topics are viewable by all authenticated users
CREATE POLICY "Event skills are viewable by all authenticated users"
ON public.event_skills FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "Event topics are viewable by all authenticated users"
ON public.event_topics FOR SELECT
TO authenticated
USING (true);

-- Event organizers can manage their event skills and topics
CREATE POLICY "Event organizers can manage their event skills"
ON public.event_skills
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.events e WHERE e.id = event_id AND e.organizer_id = auth.uid()))
WITH CHECK (EXISTS (SELECT 1 FROM public.events e WHERE e.id = event_id AND e.organizer_id = auth.uid()));

CREATE POLICY "Event organizers can manage their event topics"
ON public.event_topics
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.events e WHERE e.id = event_id AND e.organizer_id = auth.uid()))
WITH CHECK (EXISTS (SELECT 1 FROM public.events e WHERE e.id = event_id AND e.organizer_id = auth.uid()));

-- Event attendees can be viewed by all authenticated users
CREATE POLICY "Event attendees are viewable by all authenticated users"
ON public.event_attendees FOR SELECT
TO authenticated
USING (true);

-- Users can manage their own event attendance
CREATE POLICY "Users can manage their own event attendance"
ON public.event_attendees
FOR ALL TO authenticated
USING (profile_id = auth.uid())
WITH CHECK (profile_id = auth.uid());

-- Connections are viewable by the involved users
CREATE POLICY "Users can view their own connections"
ON public.connections FOR SELECT
TO authenticated
USING (requester_id = auth.uid() OR receiver_id = auth.uid());

-- Users can manage their own connections
CREATE POLICY "Users can manage their own connections"
ON public.connections
FOR ALL TO authenticated
USING (requester_id = auth.uid() OR receiver_id = auth.uid())
WITH CHECK (requester_id = auth.uid() OR receiver_id = auth.uid());

-- Messages are viewable by the involved users
CREATE POLICY "Users can view their own messages"
ON public.messages FOR SELECT
TO authenticated
USING (sender_id = auth.uid() OR receiver_id = auth.uid());

-- Users can manage their own messages
CREATE POLICY "Users can manage their own messages"
ON public.messages
FOR ALL TO authenticated
USING (sender_id = auth.uid())
WITH CHECK (sender_id = auth.uid());

-- Message attachments are viewable by the involved users
CREATE POLICY "Users can view their own message attachments"
ON public.message_attachments FOR SELECT
TO authenticated
USING (EXISTS (SELECT 1 FROM public.messages m WHERE m.id = message_id AND (m.sender_id = auth.uid() OR m.receiver_id = auth.uid())));

-- Users can manage their own message attachments
CREATE POLICY "Users can manage their own message attachments"
ON public.message_attachments
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.messages m WHERE m.id = message_id AND m.sender_id = auth.uid()))
WITH CHECK (EXISTS (SELECT 1 FROM public.messages m WHERE m.id = message_id AND m.sender_id = auth.uid())); 