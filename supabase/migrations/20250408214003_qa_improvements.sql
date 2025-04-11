-- Add constraint to prevent self-connections
ALTER TABLE public.connections 
ADD CONSTRAINT no_self_connections 
CHECK (requester_id != receiver_id);

-- Add content size limit for messages
ALTER TABLE public.messages 
ADD CONSTRAINT content_length 
CHECK (length(content) <= 10000);

-- Add index for message querying
CREATE INDEX IF NOT EXISTS idx_messages_read_status 
ON public.messages(receiver_id, read);

-- Add trigger to enforce max_attendees limit
CREATE OR REPLACE FUNCTION check_max_attendees()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.events e
        WHERE e.id = NEW.event_id
        AND e.max_attendees IS NOT NULL
        AND (
            SELECT COUNT(*) 
            FROM public.event_attendees ea 
            WHERE ea.event_id = NEW.event_id 
            AND ea.status = 'registered'
        ) >= e.max_attendees
        AND NEW.status = 'registered'
    ) THEN
        -- Automatically set to waitlisted if max_attendees reached
        NEW.status := 'waitlisted';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_max_attendees
    BEFORE INSERT OR UPDATE ON public.event_attendees
    FOR EACH ROW
    EXECUTE FUNCTION check_max_attendees();

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_events_date 
ON public.events(date);

CREATE INDEX IF NOT EXISTS idx_connections_status 
ON public.connections(status);

CREATE INDEX IF NOT EXISTS idx_event_attendees_status 
ON public.event_attendees(status);

-- Add composite indexes for relationship lookups
CREATE INDEX IF NOT EXISTS idx_user_skills_lookup 
ON public.user_skills(profile_id, skill_id);

-- Add text search index for event search
CREATE INDEX IF NOT EXISTS idx_events_text_search 
ON public.events USING GIN (to_tsvector('english', title || ' ' || COALESCE(description, ''))); 