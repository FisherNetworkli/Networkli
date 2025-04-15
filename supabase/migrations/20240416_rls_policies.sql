-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Create policies for profiles table
CREATE POLICY "Public profiles are viewable by everyone" ON profiles
    FOR SELECT USING (true);

CREATE POLICY "Service role can insert profiles" ON profiles
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Service role can manage profiles" ON profiles
    TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Users can update own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id) WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can view public profiles" ON profiles
    FOR SELECT USING (profile_visibility = 'public');

-- Profiles policies
CREATE POLICY "Users can view profiles of their connections"
  ON profiles FOR SELECT
  USING (
    profile_visibility = 'connections' AND
    EXISTS (
      SELECT 1 FROM connections
      WHERE (requester_id = auth.uid() AND receiver_id = profiles.id AND status = 'accepted')
      OR (receiver_id = auth.uid() AND requester_id = profiles.id AND status = 'accepted')
    )
  );

CREATE POLICY "Users can view their own profile"
  ON profiles FOR SELECT
  USING (id = auth.uid());

CREATE POLICY "Users can update their own profile"
  ON profiles FOR UPDATE
  USING (id = auth.uid())
  WITH CHECK (id = auth.uid());

-- User preferences policies
CREATE POLICY "Users can view their own preferences"
  ON user_preferences FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can update their own preferences"
  ON user_preferences FOR UPDATE
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can insert their own preferences"
  ON user_preferences FOR INSERT
  WITH CHECK (user_id = auth.uid());

-- Activity tracking policies
CREATE POLICY "Users can view their own activity"
  ON activity_tracking FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can update their own activity"
  ON activity_tracking FOR UPDATE
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can insert their own activity"
  ON activity_tracking FOR INSERT
  WITH CHECK (user_id = auth.uid());

-- User skills policies
CREATE POLICY "Anyone can view user skills"
  ON user_skills FOR SELECT
  USING (true);

CREATE POLICY "Users can manage their own skills"
  ON user_skills FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

-- Connections policies
CREATE POLICY "Users can view their own connections"
  ON connections FOR SELECT
  USING (requester_id = auth.uid() OR receiver_id = auth.uid());

CREATE POLICY "Users can create connection requests"
  ON connections FOR INSERT
  WITH CHECK (requester_id = auth.uid());

CREATE POLICY "Users can update their connection status"
  ON connections FOR UPDATE
  USING (receiver_id = auth.uid())
  WITH CHECK (receiver_id = auth.uid());

-- Messages policies
CREATE POLICY "Users can view their own messages"
  ON messages FOR SELECT
  USING (sender_id = auth.uid() OR receiver_id = auth.uid());

CREATE POLICY "Users can send messages"
  ON messages FOR INSERT
  WITH CHECK (sender_id = auth.uid());

CREATE POLICY "Users can update their own sent messages"
  ON messages FOR UPDATE
  USING (sender_id = auth.uid())
  WITH CHECK (sender_id = auth.uid()); 