export interface Option {
  value: string;
  label: string;
  category: string;
}

export const SKILL_OPTIONS: Option[] = [
  // Programming Languages
  { value: 'javascript', label: 'JavaScript', category: 'Programming Languages' },
  { value: 'typescript', label: 'TypeScript', category: 'Programming Languages' },
  { value: 'python', label: 'Python', category: 'Programming Languages' },
  { value: 'java', label: 'Java', category: 'Programming Languages' },
  { value: 'cpp', label: 'C++', category: 'Programming Languages' },
  { value: 'csharp', label: 'C#', category: 'Programming Languages' },
  { value: 'ruby', label: 'Ruby', category: 'Programming Languages' },
  { value: 'swift', label: 'Swift', category: 'Programming Languages' },
  { value: 'kotlin', label: 'Kotlin', category: 'Programming Languages' },
  { value: 'go', label: 'Go', category: 'Programming Languages' },

  // Web Development
  { value: 'react', label: 'React', category: 'Web Development' },
  { value: 'angular', label: 'Angular', category: 'Web Development' },
  { value: 'vue', label: 'Vue.js', category: 'Web Development' },
  { value: 'node', label: 'Node.js', category: 'Web Development' },
  { value: 'html_css', label: 'HTML/CSS', category: 'Web Development' },
  { value: 'nextjs', label: 'Next.js', category: 'Web Development' },
  { value: 'graphql', label: 'GraphQL', category: 'Web Development' },
  { value: 'webpack', label: 'Webpack', category: 'Web Development' },

  // Data & AI
  { value: 'machine_learning', label: 'Machine Learning', category: 'Data & AI' },
  { value: 'data_science', label: 'Data Science', category: 'Data & AI' },
  { value: 'deep_learning', label: 'Deep Learning', category: 'Data & AI' },
  { value: 'nlp', label: 'Natural Language Processing', category: 'Data & AI' },
  { value: 'computer_vision', label: 'Computer Vision', category: 'Data & AI' },
  { value: 'data_analysis', label: 'Data Analysis', category: 'Data & AI' },
  { value: 'sql', label: 'SQL', category: 'Data & AI' },
  { value: 'big_data', label: 'Big Data', category: 'Data & AI' },

  // Cloud & DevOps
  { value: 'aws', label: 'AWS', category: 'Cloud & DevOps' },
  { value: 'azure', label: 'Azure', category: 'Cloud & DevOps' },
  { value: 'gcp', label: 'Google Cloud', category: 'Cloud & DevOps' },
  { value: 'docker', label: 'Docker', category: 'Cloud & DevOps' },
  { value: 'kubernetes', label: 'Kubernetes', category: 'Cloud & DevOps' },
  { value: 'ci_cd', label: 'CI/CD', category: 'Cloud & DevOps' },
  { value: 'terraform', label: 'Terraform', category: 'Cloud & DevOps' },
  { value: 'linux', label: 'Linux', category: 'Cloud & DevOps' },

  // Design & UX
  { value: 'ui_design', label: 'UI Design', category: 'Design & UX' },
  { value: 'ux_design', label: 'UX Design', category: 'Design & UX' },
  { value: 'figma', label: 'Figma', category: 'Design & UX' },
  { value: 'sketch', label: 'Sketch', category: 'Design & UX' },
  { value: 'adobe_xd', label: 'Adobe XD', category: 'Design & UX' },
  { value: 'prototyping', label: 'Prototyping', category: 'Design & UX' },
  { value: 'wireframing', label: 'Wireframing', category: 'Design & UX' },
  { value: 'user_research', label: 'User Research', category: 'Design & UX' },

  // Project Management
  { value: 'agile', label: 'Agile', category: 'Project Management' },
  { value: 'scrum', label: 'Scrum', category: 'Project Management' },
  { value: 'kanban', label: 'Kanban', category: 'Project Management' },
  { value: 'jira', label: 'Jira', category: 'Project Management' },
  { value: 'trello', label: 'Trello', category: 'Project Management' },
  { value: 'project_planning', label: 'Project Planning', category: 'Project Management' },
  { value: 'risk_management', label: 'Risk Management', category: 'Project Management' },
  { value: 'stakeholder_management', label: 'Stakeholder Management', category: 'Project Management' },
];

export const INTEREST_OPTIONS: Option[] = [
  // Technology
  { value: 'artificial_intelligence', label: 'Artificial Intelligence', category: 'Technology' },
  { value: 'blockchain', label: 'Blockchain', category: 'Technology' },
  { value: 'cloud_computing', label: 'Cloud Computing', category: 'Technology' },
  { value: 'cybersecurity', label: 'Cybersecurity', category: 'Technology' },
  { value: 'data_science', label: 'Data Science', category: 'Technology' },
  { value: 'machine_learning', label: 'Machine Learning', category: 'Technology' },
  { value: 'mobile_development', label: 'Mobile Development', category: 'Technology' },
  { value: 'web_development', label: 'Web Development', category: 'Technology' },
  { value: 'devops', label: 'DevOps', category: 'Technology' },
  { value: 'iot', label: 'Internet of Things', category: 'Technology' },
  { value: 'ar_vr', label: 'AR/VR', category: 'Technology' },
  { value: 'quantum_computing', label: 'Quantum Computing', category: 'Technology' },

  // Business & Industry
  { value: 'business_strategy', label: 'Business Strategy', category: 'Business & Industry' },
  { value: 'entrepreneurship', label: 'Entrepreneurship', category: 'Business & Industry' },
  { value: 'finance', label: 'Finance & Investment', category: 'Business & Industry' },
  { value: 'marketing', label: 'Marketing & Growth', category: 'Business & Industry' },
  { value: 'product_management', label: 'Product Management', category: 'Business & Industry' },
  { value: 'project_management', label: 'Project Management', category: 'Business & Industry' },
  { value: 'sales', label: 'Sales & Business Development', category: 'Business & Industry' },
  { value: 'consulting', label: 'Consulting', category: 'Business & Industry' },
  { value: 'ecommerce', label: 'E-commerce', category: 'Business & Industry' },
  { value: 'digital_transformation', label: 'Digital Transformation', category: 'Business & Industry' },

  // Creative & Design
  { value: 'ui_ux_design', label: 'UI/UX Design', category: 'Creative & Design' },
  { value: 'graphic_design', label: 'Graphic Design', category: 'Creative & Design' },
  { value: 'content_creation', label: 'Content Creation', category: 'Creative & Design' },
  { value: 'digital_media', label: 'Digital Media', category: 'Creative & Design' },
  { value: 'branding', label: 'Branding', category: 'Creative & Design' },
  { value: 'animation', label: 'Animation', category: 'Creative & Design' },
  { value: 'video_production', label: 'Video Production', category: 'Creative & Design' },
  { value: 'game_design', label: 'Game Design', category: 'Creative & Design' },

  // Innovation & Research
  { value: 'research_development', label: 'Research & Development', category: 'Innovation & Research' },
  { value: 'biotech', label: 'Biotechnology', category: 'Innovation & Research' },
  { value: 'renewable_energy', label: 'Renewable Energy', category: 'Innovation & Research' },
  { value: 'space_technology', label: 'Space Technology', category: 'Innovation & Research' },
  { value: 'nanotechnology', label: 'Nanotechnology', category: 'Innovation & Research' },
  { value: 'robotics', label: 'Robotics', category: 'Innovation & Research' },
  { value: 'materials_science', label: 'Materials Science', category: 'Innovation & Research' },

  // Social Impact
  { value: 'sustainability', label: 'Sustainability', category: 'Social Impact' },
  { value: 'social_entrepreneurship', label: 'Social Entrepreneurship', category: 'Social Impact' },
  { value: 'education_tech', label: 'Education Technology', category: 'Social Impact' },
  { value: 'healthcare_innovation', label: 'Healthcare Innovation', category: 'Social Impact' },
  { value: 'climate_tech', label: 'Climate Technology', category: 'Social Impact' },
  { value: 'smart_cities', label: 'Smart Cities', category: 'Social Impact' },
  { value: 'digital_inclusion', label: 'Digital Inclusion', category: 'Social Impact' },
];

export const GOAL_OPTIONS: Option[] = [
  // Career Development
  { value: 'career_growth', label: 'Career Growth & Advancement', category: 'Career Development' },
  { value: 'skill_development', label: 'New Skills & Expertise', category: 'Career Development' },
  { value: 'leadership', label: 'Leadership Development', category: 'Career Development' },
  { value: 'entrepreneurship', label: 'Start a Business', category: 'Career Development' },
  { value: 'job_opportunities', label: 'Find Job Opportunities', category: 'Career Development' },
  { value: 'career_transition', label: 'Career Transition', category: 'Career Development' },
  { value: 'industry_expertise', label: 'Develop Industry Expertise', category: 'Career Development' },
  { value: 'executive_position', label: 'Reach Executive Position', category: 'Career Development' },

  // Professional Network
  { value: 'find_mentor', label: 'Find a Mentor', category: 'Professional Network' },
  { value: 'become_mentor', label: 'Become a Mentor', category: 'Professional Network' },
  { value: 'collaboration', label: 'Find Collaborators', category: 'Professional Network' },
  { value: 'networking', label: 'Expand Professional Network', category: 'Professional Network' },
  { value: 'partnership', label: 'Find Business Partners', category: 'Professional Network' },
  { value: 'industry_connections', label: 'Build Industry Connections', category: 'Professional Network' },
  { value: 'global_network', label: 'Build Global Network', category: 'Professional Network' },
  { value: 'community_building', label: 'Build Professional Community', category: 'Professional Network' },

  // Business Growth
  { value: 'start_business', label: 'Start a Business', category: 'Business Growth' },
  { value: 'scale_business', label: 'Scale Business', category: 'Business Growth' },
  { value: 'funding', label: 'Secure Funding', category: 'Business Growth' },
  { value: 'market_expansion', label: 'Market Expansion', category: 'Business Growth' },
  { value: 'innovation', label: 'Drive Innovation', category: 'Business Growth' },
  { value: 'digital_transformation', label: 'Lead Digital Transformation', category: 'Business Growth' },
  { value: 'acquisition', label: 'Mergers & Acquisitions', category: 'Business Growth' },

  // Personal Growth
  { value: 'work_life_balance', label: 'Better Work-Life Balance', category: 'Personal Growth' },
  { value: 'continuous_learning', label: 'Continuous Learning', category: 'Personal Growth' },
  { value: 'public_speaking', label: 'Public Speaking', category: 'Personal Growth' },
  { value: 'personal_brand', label: 'Build Personal Brand', category: 'Personal Growth' },
  { value: 'thought_leadership', label: 'Become Thought Leader', category: 'Personal Growth' },
  { value: 'industry_influence', label: 'Increase Industry Influence', category: 'Personal Growth' },
  { value: 'writing_publishing', label: 'Writing & Publishing', category: 'Personal Growth' },
];

export const VALUE_OPTIONS: Option[] = [
  // Professional Values
  { value: 'innovation', label: 'Innovation', category: 'Professional Values' },
  { value: 'integrity', label: 'Integrity', category: 'Professional Values' },
  { value: 'excellence', label: 'Excellence', category: 'Professional Values' },
  { value: 'accountability', label: 'Accountability', category: 'Professional Values' },
  { value: 'professionalism', label: 'Professionalism', category: 'Professional Values' },
  { value: 'quality', label: 'Quality', category: 'Professional Values' },
  { value: 'results_driven', label: 'Results-Driven', category: 'Professional Values' },
  { value: 'customer_focus', label: 'Customer Focus', category: 'Professional Values' },

  // Leadership Values
  { value: 'vision', label: 'Vision', category: 'Leadership Values' },
  { value: 'inspiration', label: 'Inspiration', category: 'Leadership Values' },
  { value: 'empowerment', label: 'Empowerment', category: 'Leadership Values' },
  { value: 'decisiveness', label: 'Decisiveness', category: 'Leadership Values' },
  { value: 'strategic_thinking', label: 'Strategic Thinking', category: 'Leadership Values' },
  { value: 'mentorship', label: 'Mentorship', category: 'Leadership Values' },
  { value: 'influence', label: 'Positive Influence', category: 'Leadership Values' },

  // Collaboration Values
  { value: 'teamwork', label: 'Teamwork', category: 'Collaboration Values' },
  { value: 'communication', label: 'Open Communication', category: 'Collaboration Values' },
  { value: 'respect', label: 'Mutual Respect', category: 'Collaboration Values' },
  { value: 'trust', label: 'Trust', category: 'Collaboration Values' },
  { value: 'empathy', label: 'Empathy', category: 'Collaboration Values' },
  { value: 'diversity', label: 'Diversity & Inclusion', category: 'Collaboration Values' },
  { value: 'cultural_awareness', label: 'Cultural Awareness', category: 'Collaboration Values' },

  // Growth Values
  { value: 'continuous_learning', label: 'Continuous Learning', category: 'Growth Values' },
  { value: 'adaptability', label: 'Adaptability', category: 'Growth Values' },
  { value: 'creativity', label: 'Creativity', category: 'Growth Values' },
  { value: 'resilience', label: 'Resilience', category: 'Growth Values' },
  { value: 'curiosity', label: 'Curiosity', category: 'Growth Values' },
  { value: 'initiative', label: 'Initiative', category: 'Growth Values' },
  { value: 'growth_mindset', label: 'Growth Mindset', category: 'Growth Values' },

  // Social Impact Values
  { value: 'sustainability', label: 'Sustainability', category: 'Social Impact Values' },
  { value: 'social_responsibility', label: 'Social Responsibility', category: 'Social Impact Values' },
  { value: 'environmental_care', label: 'Environmental Care', category: 'Social Impact Values' },
  { value: 'community_impact', label: 'Community Impact', category: 'Social Impact Values' },
  { value: 'ethical_practice', label: 'Ethical Practice', category: 'Social Impact Values' },
  { value: 'global_perspective', label: 'Global Perspective', category: 'Social Impact Values' },
  { value: 'future_focus', label: 'Future-Focused', category: 'Social Impact Values' },
]; 