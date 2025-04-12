-- Create application status enum
CREATE TYPE application_status AS ENUM ('PENDING', 'REVIEWING', 'ACCEPTED', 'REJECTED');

-- Create applications table
CREATE TABLE application_submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    linkedin TEXT,
    github TEXT,
    portfolio TEXT,
    experience TEXT NOT NULL,
    availability TEXT NOT NULL,
    salary TEXT,
    referral TEXT,
    video_url TEXT NOT NULL,
    status application_status NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- Add RLS policies
ALTER TABLE application_submissions ENABLE ROW LEVEL SECURITY;

-- Allow admins to view all applications
CREATE POLICY "Allow admins to view all applications" ON application_submissions
    FOR SELECT
    TO authenticated
    USING (auth.jwt() ->> 'email' IN (SELECT email FROM profiles WHERE role = 'admin'::user_role));

-- Allow admins to update applications
CREATE POLICY "Allow admins to update applications" ON application_submissions
    FOR UPDATE
    TO authenticated
    USING (auth.jwt() ->> 'email' IN (SELECT email FROM profiles WHERE role = 'admin'::user_role))
    WITH CHECK (auth.jwt() ->> 'email' IN (SELECT email FROM profiles WHERE role = 'admin'::user_role));

-- Allow anyone to create applications
CREATE POLICY "Allow anyone to create applications" ON application_submissions
    FOR INSERT
    TO public
    WITH CHECK (true);

-- Create updated_at trigger
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON application_submissions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 