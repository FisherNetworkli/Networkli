-- Seed data for testing

-- Insert sample skills
INSERT INTO skills (name, category) VALUES
    ('JavaScript', 'Programming'),
    ('Python', 'Programming'),
    ('React', 'Frontend'),
    ('Node.js', 'Backend'),
    ('SQL', 'Database'),
    ('AWS', 'Cloud'),
    ('Docker', 'DevOps'),
    ('UI/UX Design', 'Design'),
    ('Product Management', 'Business'),
    ('Digital Marketing', 'Marketing')
ON CONFLICT (name) DO UPDATE SET category = EXCLUDED.category;

-- Insert sample topics
INSERT INTO topics (name) VALUES
    ('Web Development'),
    ('Data Science'),
    ('Mobile Development'),
    ('Cloud Computing'),
    ('Artificial Intelligence'),
    ('Product Management'),
    ('UX/UI Design')
ON CONFLICT (name) DO NOTHING;

-- Insert sample profiles (passwords would be handled by Supabase Auth)
INSERT INTO profiles (id, email, full_name, title, company, industry, bio, location, website, role) VALUES
    ('123e4567-e89b-12d3-a456-426614174000', 'john.doe@example.com', 'John Doe', 'Senior Developer', 'Tech Corp', 'Technology', 'Passionate about web development and cloud computing', 'San Francisco, CA', 'https://johndoe.dev', 'user'),
    ('223e4567-e89b-12d3-a456-426614174001', 'jane.smith@example.com', 'Jane Smith', 'Product Manager', 'Innovation Inc', 'Software', 'Experienced in leading tech products', 'New York, NY', 'https://janesmith.com', 'user'),
    ('323e4567-e89b-12d3-a456-426614174002', 'bob.wilson@example.com', 'Bob Wilson', 'UX Designer', 'Design Studio', 'Design', 'Creating user-centered digital experiences', 'Austin, TX', 'https://bobwilson.design', 'user'),
    ('423e4567-e89b-12d3-a456-426614174003', 'alice.johnson@example.com', 'Alice Johnson', 'Data Scientist', 'Data Analytics Co', 'Data Science', 'Machine learning enthusiast', 'Seattle, WA', 'https://alicejohnson.ai', 'user'),
    ('523e4567-e89b-12d3-a456-426614174004', 'admin@networkli.com', 'Admin User', 'Platform Admin', 'Networkli', 'Technology', 'Platform administrator', 'Remote', 'https://networkli.com', 'admin');

-- Insert sample user skills
INSERT INTO user_skills (profile_id, skill_id, level, years_of_experience) VALUES
    ('123e4567-e89b-12d3-a456-426614174000', (SELECT id FROM skills WHERE name = 'JavaScript'), 'expert'::skill_level, 8),
    ('123e4567-e89b-12d3-a456-426614174000', (SELECT id FROM skills WHERE name = 'React'), 'advanced'::skill_level, 5),
    ('223e4567-e89b-12d3-a456-426614174001', (SELECT id FROM skills WHERE name = 'Product Management'), 'expert'::skill_level, 6),
    ('323e4567-e89b-12d3-a456-426614174002', (SELECT id FROM skills WHERE name = 'UI/UX Design'), 'expert'::skill_level, 7),
    ('423e4567-e89b-12d3-a456-426614174003', (SELECT id FROM skills WHERE name = 'Python'), 'expert'::skill_level, 5);

-- Insert sample events
INSERT INTO events (id, title, description, format, date, location, virtual_link, max_attendees, organizer_id, image_url) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', 'Web Development Workshop', 'Learn modern web development techniques', 'hybrid', NOW() + INTERVAL '7 days', 'San Francisco, CA', 'https://zoom.us/j/123456789', 50, '123e4567-e89b-12d3-a456-426614174000', 'https://example.com/event1.jpg'),
    ('723e4567-e89b-12d3-a456-426614174001', 'Product Management Fundamentals', 'Essential skills for product managers', 'virtual', NOW() + INTERVAL '14 days', NULL, 'https://zoom.us/j/987654321', 100, '223e4567-e89b-12d3-a456-426614174001', 'https://example.com/event2.jpg'),
    ('823e4567-e89b-12d3-a456-426614174002', 'UX Design Meetup', 'Networking event for UX designers', 'in_person', NOW() + INTERVAL '21 days', 'Austin, TX', NULL, 30, '323e4567-e89b-12d3-a456-426614174002', 'https://example.com/event3.jpg');

-- Insert sample event skills
INSERT INTO event_skills (event_id, skill_id, required_level) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', (SELECT id FROM skills WHERE name = 'JavaScript'), 'intermediate'::skill_level),
    ('623e4567-e89b-12d3-a456-426614174000', (SELECT id FROM skills WHERE name = 'React'), 'beginner'::skill_level),
    ('723e4567-e89b-12d3-a456-426614174001', (SELECT id FROM skills WHERE name = 'Product Management'), 'intermediate'::skill_level),
    ('823e4567-e89b-12d3-a456-426614174002', (SELECT id FROM skills WHERE name = 'UI/UX Design'), 'advanced'::skill_level);

-- Insert sample event topics
INSERT INTO event_topics (event_id, topic_id) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', (SELECT id FROM topics WHERE name = 'Web Development')),
    ('723e4567-e89b-12d3-a456-426614174001', (SELECT id FROM topics WHERE name = 'Product Management')),
    ('823e4567-e89b-12d3-a456-426614174002', (SELECT id FROM topics WHERE name = 'UX/UI Design'));

-- Insert sample connections
INSERT INTO connections (id, requester_id, receiver_id, status) VALUES
    ('923e4567-e89b-12d3-a456-426614174000', '123e4567-e89b-12d3-a456-426614174000', '223e4567-e89b-12d3-a456-426614174001', 'accepted'),
    ('a23e4567-e89b-12d3-a456-426614174001', '323e4567-e89b-12d3-a456-426614174002', '123e4567-e89b-12d3-a456-426614174000', 'pending'),
    ('b23e4567-e89b-12d3-a456-426614174002', '423e4567-e89b-12d3-a456-426614174003', '223e4567-e89b-12d3-a456-426614174001', 'accepted');

-- Insert sample messages
INSERT INTO messages (id, sender_id, receiver_id, content, read) VALUES
    ('c23e4567-e89b-12d3-a456-426614174000', '123e4567-e89b-12d3-a456-426614174000', '223e4567-e89b-12d3-a456-426614174001', 'Hey, would love to collaborate on a project!', true),
    ('d23e4567-e89b-12d3-a456-426614174001', '223e4567-e89b-12d3-a456-426614174001', '123e4567-e89b-12d3-a456-426614174000', 'Sounds great! What do you have in mind?', false),
    ('e23e4567-e89b-12d3-a456-426614174002', '423e4567-e89b-12d3-a456-426614174003', '223e4567-e89b-12d3-a456-426614174001', 'Thanks for connecting!', true);

-- Insert sample message attachments
INSERT INTO message_attachments (id, message_id, type, url, name) VALUES
    ('f23e4567-e89b-12d3-a456-426614174000', 'c23e4567-e89b-12d3-a456-426614174000', 'file', 'https://example.com/files/proposal.pdf', 'Project Proposal'),
    ('023e4567-e89b-12d3-a456-426614174001', 'd23e4567-e89b-12d3-a456-426614174001', 'image', 'https://example.com/images/mockup.png', 'Project Mockup');

-- Insert sample event attendees
INSERT INTO event_attendees (event_id, profile_id, status) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', '223e4567-e89b-12d3-a456-426614174001', 'registered'),
    ('723e4567-e89b-12d3-a456-426614174001', '323e4567-e89b-12d3-a456-426614174002', 'registered'),
    ('823e4567-e89b-12d3-a456-426614174002', '423e4567-e89b-12d3-a456-426614174003', 'registered'); 