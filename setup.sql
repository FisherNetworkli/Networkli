-- Drop existing policies and triggers
DROP POLICY IF EXISTS "Users can insert their own profile" ON profiles;
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Create the profile creation function
CREATE OR REPLACE FUNCTION create_profile_with_role()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (
    id,
    email,
    first_name,
    last_name,
    full_name,
    profile_visibility,
    email_notifications,
    marketing_emails,
    role
  )
  VALUES (
    NEW.id,
    NEW.email,
    NEW.raw_user_meta_data->>'firstName',
    NEW.raw_user_meta_data->>'lastName',
    CONCAT(NEW.raw_user_meta_data->>'firstName', ' ', NEW.raw_user_meta_data->>'lastName'),
    COALESCE(NEW.raw_user_meta_data->>'profileVisibility', 'public'),
    COALESCE((NEW.raw_user_meta_data->>'emailNotifications')::boolean, true),
    COALESCE((NEW.raw_user_meta_data->>'marketingEmails')::boolean, false),
    'user'
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create the trigger
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION create_profile_with_role();

-- Create the insert policy
CREATE POLICY "Service role can insert profiles"
  ON profiles FOR INSERT
  WITH CHECK (true); 