-- Add new fields to profiles table
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS first_name TEXT,
ADD COLUMN IF NOT EXISTS last_name TEXT,
ADD COLUMN IF NOT EXISTS experience_level TEXT,
ADD COLUMN IF NOT EXISTS linkedin_url TEXT,
ADD COLUMN IF NOT EXISTS github_url TEXT,
ADD COLUMN IF NOT EXISTS portfolio_url TEXT,
ADD COLUMN IF NOT EXISTS twitter_url TEXT,
ADD COLUMN IF NOT EXISTS profile_visibility TEXT DEFAULT 'public' CHECK (profile_visibility IN ('public', 'private', 'connections')),
ADD COLUMN IF NOT EXISTS email_notifications BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS marketing_emails BOOLEAN DEFAULT false;

-- Create user preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    interests TEXT[] DEFAULT '{}',
    looking_for TEXT[] DEFAULT '{}',
    preferred_industries TEXT[] DEFAULT '{}',
    preferred_roles TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add RLS policies for user_preferences
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own preferences"
    ON user_preferences FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own preferences"
    ON user_preferences FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own preferences"
    ON user_preferences FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Create trigger for updated_at
CREATE TRIGGER set_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Update profiles trigger to handle new fields
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (
        id,
        email,
        first_name,
        last_name,
        profile_visibility,
        email_notifications,
        marketing_emails
    )
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data->>'firstName',
        NEW.raw_user_meta_data->>'lastName',
        COALESCE(NEW.raw_user_meta_data->>'profileVisibility', 'public'),
        COALESCE((NEW.raw_user_meta_data->>'emailNotifications')::boolean, true),
        COALESCE((NEW.raw_user_meta_data->>'marketingEmails')::boolean, false)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Create new trigger
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION handle_new_user(); 