-- Seed data for testing

-- Insert sample skills
INSERT INTO skills (id, name, category) VALUES
    ('c22e33d6-1e98-4f0a-a290-4c0523171f12', 'JavaScript', 'Programming'),
    ('d8b4b7a2-b5d1-4c91-8f11-5b4e3a4e8f9d', 'Python', 'Programming'),
    ('f6a7b234-8c91-4f6a-9e56-2e5f9c8d1a3b', 'React', 'Frontend'),
    ('a1b2c3d4-e5f6-4a5b-8c9d-1e2f3a4b5c6d', 'Node.js', 'Backend'),
    ('b2c3d4e5-f6a7-5b6c-9d0e-2f3a4b5c6d7e', 'SQL', 'Database'),
    ('c3d4e5f6-a7b8-6c9d-0e1f-3a4b5c6d7e8f', 'AWS', 'Cloud'),
    ('d4e5f6a7-b8c9-7d0e-1f2a-4b5c6d7e8f9a', 'Docker', 'DevOps'),
    ('e5f6a7b8-c9d0-8e1f-2a3b-5c6d7e8f9a0b', 'UI/UX Design', 'Design'),
    ('f6a7b8c9-d0e1-9f2a-3b4c-6d7e8f9a0b1c', 'Product Management', 'Business'),
    ('a7b8c9d0-e1f2-0a3b-4c5d-7e8f9a0b1c2d', 'Digital Marketing', 'Marketing');

-- Insert sample topics
INSERT INTO topics (id, name) VALUES
    ('b8c9d0e1-f2a3-1b4c-5d6e-8f9a0b1c2d3e', 'Web Development'),
    ('c9d0e1f2-a3b4-2c5d-6e7f-9a0b1c2d3e4f', 'Data Science'),
    ('d0e1f2a3-b4c5-3d6e-7f8a-0b1c2d3e4f5a', 'Mobile Development'),
    ('e1f2a3b4-c5d6-4e7f-8a9b-1c2d3e4f5a6b', 'Cloud Computing'),
    ('f2a3b4c5-d6e7-5f8a-9b0c-2d3e4f5a6b7c', 'Artificial Intelligence');

-- Insert sample profiles (passwords would be handled by Supabase Auth)
INSERT INTO profiles (id, email, full_name, title, company, industry, bio, location, website, role) VALUES
    ('123e4567-e89b-12d3-a456-426614174000', 'john.doe@example.com', 'John Doe', 'Senior Developer', 'Tech Corp', 'Technology', 'Passionate about web development and cloud computing', 'San Francisco, CA', 'https://johndoe.dev', 'user'),
    ('223e4567-e89b-12d3-a456-426614174001', 'jane.smith@example.com', 'Jane Smith', 'Product Manager', 'Innovation Inc', 'Software', 'Experienced in leading tech products', 'New York, NY', 'https://janesmith.com', 'user'),
    ('323e4567-e89b-12d3-a456-426614174002', 'bob.wilson@example.com', 'Bob Wilson', 'UX Designer', 'Design Studio', 'Design', 'Creating user-centered digital experiences', 'Austin, TX', 'https://bobwilson.design', 'user'),
    ('423e4567-e89b-12d3-a456-426614174003', 'alice.johnson@example.com', 'Alice Johnson', 'Data Scientist', 'Data Analytics Co', 'Data Science', 'Machine learning enthusiast', 'Seattle, WA', 'https://alicejohnson.ai', 'user'),
    ('523e4567-e89b-12d3-a456-426614174004', 'admin@networkli.com', 'Admin User', 'Platform Admin', 'Networkli', 'Technology', 'Platform administrator', 'Remote', 'https://networkli.com', 'admin');

-- Insert sample user skills
INSERT INTO user_skills (profile_id, skill_id, level, years_of_experience) VALUES
    ('123e4567-e89b-12d3-a456-426614174000', 'c22e33d6-1e98-4f0a-a290-4c0523171f12', 'expert', 8),
    ('123e4567-e89b-12d3-a456-426614174000', 'f6a7b234-8c91-4f6a-9e56-2e5f9c8d1a3b', 'advanced', 5),
    ('223e4567-e89b-12d3-a456-426614174001', 'f6a7b8c9-d0e1-9f2a-3b4c-6d7e8f9a0b1c', 'expert', 6),
    ('323e4567-e89b-12d3-a456-426614174002', 'e5f6a7b8-c9d0-8e1f-2a3b-5c6d7e8f9a0b', 'expert', 7),
    ('423e4567-e89b-12d3-a456-426614174003', 'd8b4b7a2-b5d1-4c91-8f11-5b4e3a4e8f9d', 'expert', 5);

-- Insert sample events
INSERT INTO events (id, title, description, format, date, location, virtual_link, max_attendees, organizer_id, image_url) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', 'Web Development Workshop', 'Learn modern web development techniques', 'hybrid', NOW() + INTERVAL '7 days', 'San Francisco, CA', 'https://zoom.us/j/123456789', 50, '123e4567-e89b-12d3-a456-426614174000', 'https://example.com/event1.jpg'),
    ('723e4567-e89b-12d3-a456-426614174001', 'Product Management Fundamentals', 'Essential skills for product managers', 'virtual', NOW() + INTERVAL '14 days', NULL, 'https://zoom.us/j/987654321', 100, '223e4567-e89b-12d3-a456-426614174001', 'https://example.com/event2.jpg'),
    ('823e4567-e89b-12d3-a456-426614174002', 'UX Design Meetup', 'Networking event for UX designers', 'in_person', NOW() + INTERVAL '21 days', 'Austin, TX', NULL, 30, '323e4567-e89b-12d3-a456-426614174002', 'https://example.com/event3.jpg');

-- Insert sample event skills
INSERT INTO event_skills (event_id, skill_id, required_level) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', 'c22e33d6-1e98-4f0a-a290-4c0523171f12', 'intermediate'),
    ('623e4567-e89b-12d3-a456-426614174000', 'f6a7b234-8c91-4f6a-9e56-2e5f9c8d1a3b', 'beginner'),
    ('723e4567-e89b-12d3-a456-426614174001', 'f6a7b8c9-d0e1-9f2a-3b4c-6d7e8f9a0b1c', 'intermediate'),
    ('823e4567-e89b-12d3-a456-426614174002', 'e5f6a7b8-c9d0-8e1f-2a3b-5c6d7e8f9a0b', 'advanced');

-- Insert sample event topics
INSERT INTO event_topics (event_id, topic_id) VALUES
    ('623e4567-e89b-12d3-a456-426614174000', 'b8c9d0e1-f2a3-1b4c-5d6e-8f9a0b1c2d3e'),
    ('723e4567-e89b-12d3-a456-426614174001', 'f2a3b4c5-d6e7-5f8a-9b0c-2d3e4f5a6b7c'),
    ('823e4567-e89b-12d3-a456-426614174002', 'b8c9d0e1-f2a3-1b4c-5d6e-8f9a0b1c2d3e');

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