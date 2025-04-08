-- Create topics table
CREATE TABLE IF NOT EXISTS public.topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create skills table
CREATE TABLE IF NOT EXISTS public.skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create user_skills table for many-to-many relationship
CREATE TABLE IF NOT EXISTS public.user_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES public.skills(id) ON DELETE CASCADE,
    level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 5),
    years_experience NUMERIC(4,1) NOT NULL CHECK (years_experience >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, skill_id)
);

-- Create user_interests table for many-to-many relationship
CREATE TABLE IF NOT EXISTS public.user_interests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    topic_id UUID NOT NULL REFERENCES public.topics(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, topic_id)
);

-- Add updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_topics_updated_at
    BEFORE UPDATE ON public.topics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_skills_updated_at
    BEFORE UPDATE ON public.skills
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_skills_updated_at
    BEFORE UPDATE ON public.user_skills
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add RLS policies
ALTER TABLE public.topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_interests ENABLE ROW LEVEL SECURITY;

-- Topics and skills are readable by all authenticated users
CREATE POLICY "Topics are viewable by all authenticated users"
ON public.topics FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "Skills are viewable by all authenticated users"
ON public.skills FOR SELECT
TO authenticated
USING (true);

-- User skills and interests are viewable by all authenticated users
CREATE POLICY "User skills are viewable by all authenticated users"
ON public.user_skills FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "User interests are viewable by all authenticated users"
ON public.user_interests FOR SELECT
TO authenticated
USING (true);

-- Users can manage their own skills and interests
CREATE POLICY "Users can manage their own skills"
ON public.user_skills
FOR ALL TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage their own interests"
ON public.user_interests
FOR ALL TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- Insert some initial topics
INSERT INTO public.topics (name) VALUES
    ('Artificial Intelligence'),
    ('Blockchain'),
    ('Cloud Computing'),
    ('Data Science'),
    ('DevOps'),
    ('Digital Marketing'),
    ('Entrepreneurship'),
    ('Finance'),
    ('Healthcare'),
    ('IoT'),
    ('Machine Learning'),
    ('Mobile Development'),
    ('Product Management'),
    ('Software Engineering'),
    ('UX/UI Design'),
    ('Web Development')
ON CONFLICT (name) DO NOTHING;

-- Insert some initial skills
INSERT INTO public.skills (name, category) VALUES
    ('JavaScript', 'Programming'),
    ('Python', 'Programming'),
    ('Java', 'Programming'),
    ('SQL', 'Database'),
    ('React', 'Frontend'),
    ('Node.js', 'Backend'),
    ('AWS', 'Cloud'),
    ('Docker', 'DevOps'),
    ('Kubernetes', 'DevOps'),
    ('Git', 'Version Control'),
    ('TypeScript', 'Programming'),
    ('HTML/CSS', 'Frontend'),
    ('MongoDB', 'Database'),
    ('PostgreSQL', 'Database'),
    ('Vue.js', 'Frontend'),
    ('Angular', 'Frontend'),
    ('Swift', 'Mobile'),
    ('Kotlin', 'Mobile'),
    ('Flutter', 'Mobile'),
    ('React Native', 'Mobile')
ON CONFLICT (name) DO NOTHING;
