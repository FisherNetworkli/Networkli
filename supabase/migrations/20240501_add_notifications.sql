-- Create notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT,
    type TEXT, -- Can be 'message', 'connection', 'event', etc.
    related_id UUID, -- ID of the related entity (message, connection, event, etc.)
    read BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Add index for faster lookup
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(user_id, read);

-- Create trigger for updated_at
CREATE TRIGGER update_notifications_updated_at
    BEFORE UPDATE ON notifications
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add RLS policies
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Users can view their own notifications
CREATE POLICY "Users can view their own notifications"
    ON notifications FOR SELECT
    USING (auth.uid() = user_id);

-- Only service role and authenticated users can insert their own notifications
CREATE POLICY "Users can insert their own notifications"
    ON notifications FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Users can update their own notifications (e.g., mark as read)
CREATE POLICY "Users can update their own notifications"
    ON notifications FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Trigger function to create notifications for new messages
CREATE OR REPLACE FUNCTION create_message_notification()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO notifications (user_id, title, content, type, related_id)
    VALUES (
        NEW.receiver_id,
        'New Message',
        'You have received a new message',
        'message',
        NEW.id
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create notifications for new messages
CREATE TRIGGER on_message_created
    AFTER INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION create_message_notification();

-- Trigger function to create notifications for new connection requests
CREATE OR REPLACE FUNCTION create_connection_notification()
RETURNS TRIGGER AS $$
DECLARE
    requester_name TEXT;
    requester_title TEXT;
    requester_company TEXT;
    notification_content TEXT;
BEGIN
    -- Get requester profile information
    SELECT full_name, title, company 
    INTO requester_name, requester_title, requester_company
    FROM profiles WHERE id = NEW.requester_id;
    
    -- Create personalized notification message with professional context
    IF requester_title IS NOT NULL AND requester_company IS NOT NULL THEN
        notification_content := requester_name || ', ' || requester_title || ' at ' || requester_company || ', would like to connect with you';
    ELSIF requester_title IS NOT NULL THEN
        notification_content := requester_name || ', ' || requester_title || ', would like to connect with you';
    ELSE
        notification_content := requester_name || ' would like to connect with you';
    END IF;
    
    IF NEW.status = 'pending' THEN
        INSERT INTO notifications (user_id, title, content, type, related_id)
        VALUES (
            NEW.receiver_id,
            'New Professional Connection Request',
            notification_content,
            'connection',
            NEW.id
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create notifications for new connection requests
CREATE TRIGGER on_connection_created
    AFTER INSERT ON connections
    FOR EACH ROW
    EXECUTE FUNCTION create_connection_notification(); 