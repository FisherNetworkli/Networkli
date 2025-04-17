'use client';

import React, { useState, useEffect } from 'react';
import { createClientComponentClient, Session } from '@supabase/auth-helpers-nextjs';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  BeakerIcon, 
  ArrowPathIcon, 
  UserGroupIcon, 
  UserCircleIcon, 
  CalendarIcon, 
  HomeIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  PlusCircleIcon,
  UsersIcon,
  ChatBubbleOvalLeftEllipsisIcon,
  InformationCircleIcon,
  ArrowDownOnSquareIcon,
  TrashIcon,
  SparklesIcon,
  LightBulbIcon,
  PresentationChartLineIcon
} from '@heroicons/react/24/outline';
import { PlayCircleIcon, DocumentDuplicateIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';
import toast from 'react-hot-toast';
import { KeyIcon } from '@heroicons/react/24/solid'; // Added KeyIcon

// *** Updated Celebrity Profile Structure ***
interface CelebrityProfile {
  id?: string;
  first_name: string; // Changed from full_name
  last_name: string;  // Changed from full_name
  email: string;
  role: string;
  title: string; // Renamed from headline
  bio: string;
  location: string;
  industry: string;
  company: string;
  avatar_url?: string;
  skills: string[];
  interests: string[];
  professional_goals?: string[]; // Added (snake_case for potential DB mapping)
  values?: string[]; // Added
  website?: string | null; // Added
  linkedin_url?: string | null; // Added (snake_case)
  github_url?: string | null; // Added (snake_case)
  is_demo?: boolean;
  is_celebrity?: boolean;
}

// Sample celebrity data (Updated structure & added fields)
const SAMPLE_CELEBRITIES: CelebrityProfile[] = [
  {
    first_name: "Elon", last_name: "Musk", // Updated
    email: "elon.demo@networkli.com",
    role: "user",
    title: "Founder & CEO", // Updated (was headline)
    bio: "Building rockets, electric cars, and changing the world.",
    location: "Austin, TX",
    industry: "Technology",
    company: "Tesla, SpaceX, X",
    avatar_url: "https://randomuser.me/api/portraits/men/1.jpg",
    skills: ["Entrepreneurship", "Engineering", "Innovation", "Leadership"],
    interests: ["Space Exploration", "Sustainable Energy", "AI"],
    professional_goals: ['startup', 'learn'], // Added
    values: ['innovation', 'excellence'], // Added
    website: 'https://www.tesla.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Taylor", last_name: "Swift", // Updated
    email: "taylor.demo@networkli.com",
    role: "premium",
    title: "Singer-Songwriter & Entrepreneur", // Updated
    bio: "Creating music that connects with millions worldwide.",
    location: "Nashville, TN",
    industry: "Music",
    company: "Taylor Swift Productions",
    avatar_url: "https://randomuser.me/api/portraits/women/1.jpg",
    skills: ["Songwriting", "Performance", "Business Strategy", "Branding"],
    interests: ["Music Production", "Philanthropy", "Creative Writing", "Fashion"],
    professional_goals: ['personal-brand'], // Added
    values: ['collaboration', 'integrity'], // Added
    website: 'https://www.taylorswift.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Bill", last_name: "Gates", // Updated
    email: "bill.demo@networkli.com",
    role: "user",
    title: "Co-founder & Philanthropist", // Updated
    bio: "Working on global health, development, education, and climate change through the Bill & Melinda Gates Foundation.",
    location: "Seattle, WA",
    industry: "Technology",
    company: "Bill & Melinda Gates Foundation",
    avatar_url: "https://randomuser.me/api/portraits/men/2.jpg",
    skills: ["Strategy", "Leadership", "Philanthropy", "Software Development"],
    interests: ["Global Health", "Climate Change", "Education", "Reading"],
    professional_goals: ['open-source'], // Added
    values: ['sustainability'], // Added
    website: 'https://www.gatesfoundation.org', // Added
    linkedin_url: 'https://www.linkedin.com/in/williamhgates', // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Oprah", last_name: "Winfrey", // Updated
    email: "oprah.demo@networkli.com",
    role: "premium",
    title: "Media Executive & Philanthropist", // Updated
    bio: "Building media empires and connecting with audiences through authentic storytelling.",
    location: "Chicago, IL",
    industry: "Media",
    company: "Harpo Productions, OWN",
    avatar_url: "https://randomuser.me/api/portraits/women/2.jpg",
    skills: ["Media Production", "Public Speaking", "Interviewing", "Leadership", "Branding"],
    interests: ["Literature", "Wellness", "Education", "Personal Development"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['integrity', 'diversity'], // Added
    website: 'https://www.oprahdaily.com', // Added
    linkedin_url: 'https://www.linkedin.com/company/oprah-winfrey-network', // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Mark", last_name: "Zuckerberg", // Updated
    email: "mark.demo@networkli.com",
    role: "user",
    title: "Founder & CEO", // Updated
    bio: "Connecting the world through technology and building the metaverse.",
    location: "Palo Alto, CA",
    industry: "Technology",
    company: "Meta Platforms",
    avatar_url: "https://randomuser.me/api/portraits/men/3.jpg",
    skills: ["Product Development", "Leadership", "Social Media", "VR/AR"],
    interests: ["Metaverse", "AI Research", "Community Building"],
    professional_goals: ['startup', 'learn'], // Added
    values: ['innovation'], // Added
    website: 'https://about.meta.com/', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Satya", last_name: "Nadella", // Updated
    email: "satya.demo@networkli.com",
    role: "user",
    title: "CEO of Microsoft", // Updated
    bio: "Empowering every person and every organization on the planet to achieve more.",
    location: "Bellevue, WA",
    industry: "Technology",
    company: "Microsoft",
    avatar_url: "https://randomuser.me/api/portraits/men/4.jpg",
    skills: ["Cloud Computing", "Artificial Intelligence", "Corporate Strategy", "Leadership"],
    interests: ["Cricket", "Poetry", "Lifelong Learning"],
    professional_goals: ['tech-lead', 'learn'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.microsoft.com', // Added
    linkedin_url: 'https://www.linkedin.com/in/satyanadella', // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Jeff", last_name: "Bezos", // Updated
    email: "jeff.demo@networkli.com",
    role: "user",
    title: "Founder of Amazon, Blue Origin",
    bio: "Focused on space exploration, climate change initiatives, and media.",
    location: "Seattle, WA",
    industry: "Technology",
    company: "Amazon, Blue Origin",
    avatar_url: "https://randomuser.me/api/portraits/men/5.jpg",
    skills: ["E-commerce", "Logistics", "Entrepreneurship", "Innovation"],
    interests: ["Space Travel", "Climate Pledge", "Washington Post"],
    professional_goals: ['startup', 'learn'], // Added
    values: ['innovation'], // Added
    website: 'https://www.amazon.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Tim", last_name: "Cook", // Updated
    email: "tim.demo@networkli.com",
    role: "user",
    title: "CEO of Apple",
    bio: "Leading Apple to create innovative products and services that enrich people's lives.",
    location: "Cupertino, CA",
    industry: "Technology",
    company: "Apple",
    avatar_url: "https://randomuser.me/api/portraits/men/6.jpg",
    skills: ["Operations Management", "Supply Chain", "Leadership", "Finance"],
    interests: ["Fitness", "Cycling", "Human Rights", "Environment"],
    professional_goals: ['leadership', 'innovation'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.apple.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Sundar", last_name: "Pichai", // Updated
    email: "sundar.demo@networkli.com",
    role: "user",
    title: "CEO of Alphabet Inc. and Google",
    bio: "Making information universally accessible and useful through technology.",
    location: "Mountain View, CA",
    industry: "Technology",
    company: "Google / Alphabet",
    avatar_url: "https://randomuser.me/api/portraits/men/7.jpg",
    skills: ["Product Management", "AI", "Search Technology", "Leadership"],
    interests: ["Cricket", "Soccer", "Artificial Intelligence"],
    professional_goals: ['tech-lead', 'learn'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://about.google/', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Beyoncé", last_name: "Knowles-Carter", // Updated
    email: "beyonce.demo@networkli.com",
    role: "premium",
    title: "Global music icon, businesswoman, and visual artist",
    bio: "Creating groundbreaking music and visual experiences, empowering audiences worldwide.",
    location: "Los Angeles, CA",
    industry: "Music",
    company: "Parkwood Entertainment",
    avatar_url: "https://randomuser.me/api/portraits/women/3.jpg",
    skills: ["Vocal Performance", "Songwriting", "Visual Arts", "Branding", "Entrepreneurship"],
    interests: ["Fashion", "Film", "Art History", "Social Justice"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.parkwoodentertainment.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Leonardo", last_name: "DiCaprio", // Updated
    email: "leo.demo@networkli.com",
    role: "premium",
    title: "Academy Award-winning actor and environmental activist",
    bio: "Using storytelling to entertain and raise awareness about environmental issues.",
    location: "Los Angeles, CA",
    industry: "Film",
    company: "Appian Way Productions",
    avatar_url: "https://randomuser.me/api/portraits/men/8.jpg",
    skills: ["Acting", "Film Production", "Environmental Advocacy"],
    interests: ["Climate Change", "Wildlife Conservation", "Art Collecting"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.appianway.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Shonda", last_name: "Rhimes", // Updated
    email: "shonda.demo@networkli.com",
    role: "premium",
    title: "Award-winning television producer, screenwriter, and author",
    bio: "Creating compelling television narratives and empowering diverse voices.",
    location: "Los Angeles, CA",
    industry: "Media",
    company: "Shondaland",
    avatar_url: "https://randomuser.me/api/portraits/women/4.jpg",
    skills: ["Screenwriting", "Television Production", "Showrunning", "Storytelling"],
    interests: ["Reading", "Mentorship", "Diversity in Media"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.shondaland.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Dwayne", last_name: "\"The Rock\"", // Updated
    email: "dwayne.demo@networkli.com",
    role: "premium",
    title: "Actor, producer, and entrepreneur",
    bio: "Building global brands and motivating millions through hard work and positivity.",
    location: "Miami, FL",
    industry: "Film",
    company: "Seven Bucks Productions",
    avatar_url: "https://randomuser.me/api/portraits/men/9.jpg",
    skills: ["Acting", "Producing", "Marketing", "Brand Building"],
    interests: ["Fitness", "Wrestling", "Motivation"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.sevenbucks.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Adele", last_name: "Adkins", // Updated
    email: "adele.demo@networkli.com",
    role: "premium",
    title: "Multi-Grammy Award-winning singer-songwriter",
    bio: "Connecting with listeners through powerful vocals and emotionally resonant lyrics.",
    location: "London, UK",
    industry: "Music",
    company: "Melted Stone",
    avatar_url: "https://randomuser.me/api/portraits/women/5.jpg",
    skills: ["Singing", "Songwriting", "Vocal Technique"],
    interests: ["Motherhood", "Privacy", "British Culture"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.adele.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
  {
    first_name: "Tom", last_name: "Hanks", // Updated
    email: "tom.demo@networkli.com",
    role: "premium",
    title: "Two-time Academy Award-winning actor and filmmaker",
    bio: "Bringing relatable and iconic characters to life on screen for decades.",
    location: "Los Angeles, CA",
    industry: "Film",
    company: "Playtone",
    avatar_url: "https://randomuser.me/api/portraits/men/10.jpg",
    skills: ["Acting", "Producing", "Directing", "Writing"],
    interests: ["History", "Typewriters", "Space Exploration"],
    professional_goals: ['personal-brand', 'network'], // Added
    values: ['excellence', 'collaboration'], // Added
    website: 'https://www.playtone.com', // Added
    linkedin_url: null, // Added
    github_url: null, // Added
    is_demo: true,
    is_celebrity: true
  },
];

// ADDED: Sample Demo Groups
const SAMPLE_GROUPS = [
  { name: 'AI Innovators Circle', description: 'Discussing the latest in AI/ML research and applications.', category: 'Technology', industry: 'Artificial Intelligence', location: 'Global / Virtual', is_demo: true },
  { name: 'NYC Marketing Professionals', description: 'Networking and knowledge sharing for marketers in NYC.', category: 'Marketing', industry: 'Marketing & Advertising', location: 'New York, NY', is_demo: true },
  { name: 'Remote Workers Network', description: 'Tips, challenges, and community for remote professionals.', category: 'Career Development', industry: 'Various', location: 'Global / Virtual', is_demo: true },
  { name: 'Sustainable Tech Leaders', description: 'Exploring green technology and sustainable business practices.', category: 'Technology', industry: 'Renewable Energy', location: 'Global / Virtual', is_demo: true },
  { name: 'Frontend Developers Hub', description: 'Sharing best practices, tools, and frameworks for frontend dev.', category: 'Technology', industry: 'Software Development', location: 'Global / Virtual', is_demo: true },
  { name: 'Healthcare Tech Advances', description: 'Focusing on technology adoption in the healthcare industry.', category: 'Healthcare', industry: 'Healthcare Technology', location: 'Boston, MA', is_demo: true },
  { name: 'Financial Analysts Forum', description: 'Discussions on market trends, analysis techniques, and fintech.', category: 'Finance', industry: 'Financial Services', location: 'London, UK', is_demo: true },
  { name: 'Creative Arts Collaborators', description: 'A space for artists, writers, and designers to connect.', category: 'Arts', industry: 'Arts & Entertainment', location: 'Los Angeles, CA', is_demo: true },
  { name: 'Startup Founders Connect', description: 'Peer support and resources for early-stage startup founders.', category: 'Entrepreneurship', industry: 'Venture Capital & Startups', location: 'Global / Virtual', is_demo: true },
  { name: 'Data Science Practitioners', description: 'Sharing knowledge on data analysis, visualization, and ML models.', category: 'Technology', industry: 'Data Science', location: 'Global / Virtual', is_demo: true },
];

// ADDED: Sample Demo Events
const SAMPLE_EVENTS = [
  { title: 'AI Ethics & Society Conference', description: 'Exploring the ethical implications of advanced AI.', category: 'Technology', industry: 'Artificial Intelligence', date: new Date(Date.now() + 15 * 24 * 60 * 60 * 1000).toISOString(), location: 'Virtual', format: 'Conference', is_demo: true },
  { title: 'Future of Marketing Summit', description: 'Trends and strategies shaping modern marketing.', category: 'Marketing', industry: 'Marketing & Advertising', date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), location: 'New York, NY', format: 'In-Person', is_demo: true },
  { title: 'Remote Work Productivity Hacks', description: 'Workshop on maximizing efficiency for remote teams.', category: 'Career Development', industry: 'Various', date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), location: 'Virtual', format: 'Workshop', is_demo: true },
  { title: 'Green Tech Investment Forum', description: 'Connecting investors with sustainable technology startups.', category: 'Technology', industry: 'Renewable Energy', date: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000).toISOString(), location: 'San Francisco, CA', format: 'Hybrid', is_demo: true },
  { title: 'Web Performance Masterclass', description: 'Deep dive into optimizing frontend performance.', category: 'Technology', industry: 'Software Development', date: new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toISOString(), location: 'Virtual', format: 'Workshop', is_demo: true },
  { title: 'Digital Health Transformation Expo', description: 'Showcasing innovations in healthcare technology.', category: 'Healthcare', industry: 'Healthcare Technology', date: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000).toISOString(), location: 'Boston, MA', format: 'In-Person', is_demo: true },
  { title: 'FinTech Regulation Roundtable', description: 'Navigating the evolving regulatory landscape.', category: 'Finance', industry: 'Financial Services', date: new Date(Date.now() + 20 * 24 * 60 * 60 * 1000).toISOString(), location: 'Virtual', format: 'Panel', is_demo: true },
  { title: 'Generative Art Showcase & Mixer', description: 'Networking event for digital artists and creators.', category: 'Arts', industry: 'Arts & Entertainment', date: new Date(Date.now() + 25 * 24 * 60 * 60 * 1000).toISOString(), location: 'Los Angeles, CA', format: 'Networking', is_demo: true },
  { title: 'Seed Stage Fundraising Strategies', description: 'Workshop for founders seeking early investment.', category: 'Entrepreneurship', industry: 'Venture Capital & Startups', date: new Date(Date.now() + 12 * 24 * 60 * 60 * 1000).toISOString(), location: 'Virtual', format: 'Workshop', is_demo: true },
  { title: 'MLOps Best Practices Summit', description: 'Implementing and managing machine learning pipelines.', category: 'Technology', industry: 'Data Science', date: new Date(Date.now() + 50 * 24 * 60 * 60 * 1000).toISOString(), location: 'Virtual', format: 'Conference', is_demo: true },
];

// *** Add industry types for filtering ***
const TECH_INDUSTRIES = ['Technology'];
const MEDIA_INDUSTRIES = ['Music', 'Media', 'Film']; // Added Film

// Define types for fetched data if not already globally defined
// Using any for now, replace with specific types if Database types are available
type Profile = any; // Database['public']['Tables']['profiles']['Row'];
type Group = any; // Database['public']['Tables']['groups']['Row'];
type Event = any; // Database['public']['Tables']['events']['Row'];
type Connection = any; // Database['public']['Tables']['connections']['Row'];
type Interaction = any; // Database['public']['Tables']['interaction_history']['Row'];

// ADDED: Type for Recommendations (adjust based on your API response)
type Recommendation = {
  id: string;
  name?: string; // For profiles/groups
  title?: string; // For events
  reason?: string; // Optional: Why was this recommended?
  score?: number; // Optional: Recommendation score
  // Add other relevant fields from your API response
};

// Define type for interaction inserts explicitly
interface InteractionInsert {
  user_id: string;
  interaction_type: string;
  target_entity_type: string | null;
  target_entity_id: string | null;
  metadata?: any | null; // Allow any JSON structure or null
  is_demo: boolean;
  // timestamp and created_at will be handled by the database
}

export default function DemoEnvironmentPage() {
  const [demoMode, setDemoMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [demoProfilesCount, setDemoProfilesCount] = useState<number>(0);
  const [demoProfiles, setDemoProfiles] = useState<Profile[]>([]);
  const [demoConnectionsCount, setDemoConnectionsCount] = useState<number>(0);
  const [demoInteractionsCount, setDemoInteractionsCount] = useState<number>(0);
  const [demoGroupsCount, setDemoGroupsCount] = useState<number>(0);
  const [demoEventsCount, setDemoEventsCount] = useState<number>(0);
  const [celebrityFilter, setCelebrityFilter] = useState<'all' | 'tech' | 'media'>('all');
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  
  // ADDED: State for Recommendation Showcase
  const [selectedProfileId, setSelectedProfileId] = useState<string>('');
  const [profileRecommendations, setProfileRecommendations] = useState<Recommendation[]>([]);
  const [groupRecommendations, setGroupRecommendations] = useState<Recommendation[]>([]);
  const [eventRecommendations, setEventRecommendations] = useState<Recommendation[]>([]);
  const [recommendationLoading, setRecommendationLoading] = useState<boolean>(false);
  // END ADDED State
  
  // ADDED: State for seeded prospect ID
  const [seededProspectId, setSeededProspectId] = useState<string | null>(null);
  const [seededProspectName, setSeededProspectName] = useState<string | null>(null); // For button label

  // Add a new state for prospect form data
  const [prospectData, setProspectData] = useState<Partial<CelebrityProfile>>({
    first_name: 'Chris',
    last_name: 'Harder',
    email: 'chris.prospect@networkli.com', // Using a demo email
    title: 'Entrepreneur, Investor, Podcaster',
    company: 'For The Love Of Money',
    industry: 'Entrepreneurship, Personal Development, Media',
    location: 'Scottsdale, AZ', // Common location associated with him
    bio: 'Host of the "For The Love Of Money" podcast, dedicated to helping entrepreneurs build wealth and live generously. Investor and community builder.',
    skills: [], // Initialized empty, string version below controls input
    interests: [], // Initialized empty
    professional_goals: [], // Initialized empty
    values: [], // Initialized empty
    website: 'https://chrisharder.me/', // Example website
    linkedin_url: 'https://www.linkedin.com/in/chriswharder/', // Example LinkedIn
    github_url: '', // Unlikely to have a public GitHub focused profile
    role: 'user',
  });

  // State to hold the comma-separated string inputs for array fields
  const [prospectArrayInputs, setProspectArrayInputs] = useState({
      skillsString: 'Podcasting, Interviewing, Business Strategy, Investing, Networking, Public Speaking, Personal Branding, Community Building',
      interestsString: 'Entrepreneurship, Wealth Building, Personal Growth, Philanthropy, Masterminds, Generosity',
      goalsString: 'Helping others succeed, Building community, Impact investing, Financial freedom',
      valuesString: 'Generosity, Abundance Mindset, Authenticity, Connection, Impact'
  });

  const router = useRouter();
  const supabase = createClientComponentClient();

  // --- Utility function to fetch current counts AND profile data VIA API ---
  const fetchCurrentData = async () => {
    console.log("[Fetch Data] Calling API route /api/admin/demo-status...");
    try {
        // Add a unique timestamp as a cache buster for every fetch
        const timestamp = Date.now(); 
        const response = await fetch(`/api/admin/demo-status?t=${timestamp}`);

        if (!response.ok) {
            let errorMsg = `Failed to fetch demo data: ${response.statusText}`;
            try { const errorData = await response.json(); errorMsg = errorData.error || errorData.message || errorMsg; } catch (jsonError) { /* Ignore */ }
            throw new Error(errorMsg);
        }
        const data = await response.json();
        console.log("[Fetch Data] Received data:", data);

        // Update state with fetched data
        const profiles = data.profilesData ?? [];
        const counts = data.counts ?? {};
        const profileCount = counts.profiles ?? 0;
        const connCount = counts.connections ?? 0;
        const intCount = counts.interactions ?? 0;
        const groupCount = counts.groups ?? 0;
        const eventCount = counts.events ?? 0;

        setDemoProfiles(profiles);
        setDemoProfilesCount(profileCount);
        setDemoConnectionsCount(connCount);
        setDemoInteractionsCount(intCount);
        setDemoGroupsCount(groupCount);
        setDemoEventsCount(eventCount);

        // --- ADD DEBUG LOG --- 
        console.log(`[Fetch Data] State AFTER set: P=${profileCount}, C=${connCount}, I=${intCount}, G=${groupCount}, E=${eventCount}`);
        // --- END DEBUG LOG ---

    } catch(error: any) {
        console.error("[Fetch Data] Error fetching data via API:", error);
        toast.error(`Failed to fetch demo data: ${error.message}`);
    }
  };

  useEffect(() => {
    async function loadInitialState() {
      setLoading(true);
      try {
        setDemoMode(false);
        await fetchCurrentData();
      } catch (error) {
        console.error('Error loading initial demo state:', error);
      } finally {
        setLoading(false);
      }
    }
    loadInitialState();
  }, []);

  // Toggle demo mode
  const toggleDemoMode = async () => {
    setActionLoading('toggle');
    
    try {
      const newDemoModeState = !demoMode;
      setDemoMode(newDemoModeState);
      // Simulate delay
      await new Promise(resolve => setTimeout(resolve, 500)); 
      toast.success(`Demo mode ${newDemoModeState ? 'enabled' : 'disabled'} (simulated)`);
    } catch (error) {
      console.error('Error toggling demo mode (simulated):', error);
      toast.error('Failed to toggle demo mode');
    } finally {
      setActionLoading(null);
    }
  };

  // Seed celebrity profiles (Keep as is, maybe rename button to "Seed Sample Profiles")
  const seedCelebrityProfiles = async () => {
    setActionLoading('profiles');
    toast.loading(`Seeding ${celebrityFilter} celebrity profiles...`);
    console.log(`[Seed Profiles] Starting seed for filter: ${celebrityFilter}`);
    
    const celebritiesToSeed = SAMPLE_CELEBRITIES.filter(c => {
      if (celebrityFilter === 'all') return true;
      if (celebrityFilter === 'tech') return TECH_INDUSTRIES.includes(c.industry);
      if (celebrityFilter === 'media') return MEDIA_INDUSTRIES.includes(c.industry);
      return false;
    });
    console.log(`[Seed Profiles] Found ${celebritiesToSeed.length} celebrities matching filter.`);

    if (celebritiesToSeed.length === 0) {
       toast.dismiss();
       toast.error(`No celebrities found for filter: ${celebrityFilter}`);
       setActionLoading(null);
       return;
    }
    
    let seededCount = 0;
    let existingCount = 0;
    const errors: string[] = [];

    try {
      for (const celebrity of celebritiesToSeed) {
        console.log(`[Seed Profiles] Processing: ${celebrity.first_name} ${celebrity.last_name} (${celebrity.email})`);
        // Check if profile already exists
        const { data: existingProfile, error: checkError } = await supabase
          .from('profiles')
          .select('id')
          .eq('email', celebrity.email)
          .maybeSingle();
          
        if (checkError) {
            console.error(`[Seed Profiles] Error checking for ${celebrity.first_name} ${celebrity.last_name}:`, checkError);
            errors.push(`Failed check for ${celebrity.first_name} ${celebrity.last_name}: ${checkError.message}`);
            continue; // Skip this celebrity if check fails
        }
        
        if (!existingProfile) {
          console.log(`[Seed Profiles] ${celebrity.first_name} ${celebrity.last_name} does not exist. Attempting insert...`);
          const profileToInsert = {
              ...celebrity,
              is_demo: true, // Ensure this is set!
              // Map CelebrityProfile fields to DB columns if names differ, e.g.,
              // organizer_id: celebrity.id (if DB uses organizer_id for groups creator)
          };
          const { error: insertError } = await supabase
            .from('profiles')
            .insert([profileToInsert]); // Insert the modified object
          
          if (insertError) {
            console.error(`[Seed Profiles] Error inserting ${celebrity.first_name} ${celebrity.last_name}:`, insertError);
            errors.push(`Failed insert for ${celebrity.first_name} ${celebrity.last_name}: ${insertError.message}`);
          } else {
            console.log(`[Seed Profiles] Successfully inserted ${celebrity.first_name} ${celebrity.last_name}.`);
            seededCount++;
          }
        } else {
           console.log(`[Seed Profiles] ${celebrity.first_name} ${celebrity.last_name} already exists (ID: ${existingProfile.id}). Skipping insert.`);
           existingCount++;
        }
      }
      
      console.log("[Seed Profiles] Seeding loop finished. Refreshing profile list...");
      // Refresh the *entire* celebrity list from DB after seeding
      const { data: refreshedProfiles, error: refreshError } = await supabase
        .from('profiles')
        .select('*')
        .eq('is_celebrity', true);
      
      if (refreshError) {
         console.error("[Seed Profiles] Error refreshing profiles:", refreshError);
         toast.error('Seeding complete, but failed to refresh list.');
      } else {
        console.log(`[Seed Profiles] Refreshed list. Found ${refreshedProfiles?.length || 0} celebrity profiles.`);
      }
      
      toast.dismiss();
      let successMessage = `${seededCount} new profile(s) seeded.`;
      if (existingCount > 0) successMessage += ` ${existingCount} already existed.`;
      if (errors.length > 0) {
          toast.error(`Seeding finished with ${errors.length} errors. Check console.`);
          console.error("[Seed Profiles] Seeding errors encountered:", errors);
      } else {
          toast.success(successMessage);
      }

    } catch (error) {
      console.error('[Seed Profiles] Unexpected error during seeding process:', error);
      toast.dismiss();
      toast.error('Failed to seed celebrity profiles due to an unexpected error.');
    } finally {
      await fetchCurrentData(); // Make sure counts are refreshed
      setActionLoading(null);
      console.log("[Seed Profiles] Seed process finished.");
    }
  };

  // Seed Demo Connections (Calls API route)
  const seedDemoConnections = async (targetCount = 50) => {
    setActionLoading('connections');
    toast.loading(`Requesting connection seeding (target: ${targetCount})...`);
    let success = false;
    let resultMessage = '';
    let seededApiCount = 0;
    try {
        const response = await fetch('/api/admin/seed-connections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ targetCount }),
        });
        const result = await response.json();
        console.log("[Seed Connections] API Response:", result);
        resultMessage = result.message || 'Unknown API response';
        seededApiCount = result.seededCount || 0;
        if (!response.ok) throw new Error(result.error || result.message || `API Error (${response.status})`);
        success = true;
        toast.dismiss();
        if (result.errors && result.errors.length > 0) {
             toast.error(`Connections seeding API reported ${result.errors.length} errors.`);
        } else {
             toast.success(result.message || `${seededApiCount} connection RPCs executed.`);
        }
    } catch (error: any) {
        console.error('[Seed Connections] Error:', error);
        toast.dismiss();
        toast.error(`Failed to seed connections: ${error.message}`);
    } finally {
        console.log("[Seed Connections] Waiting 5 seconds before refreshing counts...");
        await new Promise(resolve => setTimeout(resolve, 5000)); // INCREASED DELAY
        console.log("[Seed Connections] Refreshing all data...");
        await fetchCurrentData();
        setActionLoading(null);
    }
  };

  // Seed Demo Interactions (Calls API route)
  const seedDemoInteractions = async (targetCount = 100) => {
    setActionLoading('interactions');
    toast.loading(`Requesting interaction seeding (target: ${targetCount})...`);
    let success = false;
    let resultMessage = '';
    let seededApiCount = 0;
    try {
        const response = await fetch('/api/admin/seed-interactions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ targetCount }),
        });
        const result = await response.json();
        console.log("[Seed Interactions] API Response:", result);
        resultMessage = result.message || 'Unknown API response';
        seededApiCount = result.seededCount || 0;
        if (!response.ok) throw new Error(result.error || result.message || `API Error (${response.status})`);
        success = true;
        toast.dismiss();
        if (result.errors && result.errors.length > 0) {
             toast.error(`Interactions seeding API reported ${result.errors.length} errors.`);
        } else {
             toast.success(result.message || `${seededApiCount} interaction RPCs executed.`);
        }
    } catch (error: any) {
      console.error('[Seed Interactions] Error:', error);
      toast.dismiss();
      toast.error(`Failed to seed interactions: ${error.message}`);
    } finally {
        console.log("[Seed Interactions] Waiting 5 seconds before refreshing counts...");
        await new Promise(resolve => setTimeout(resolve, 5000)); // INCREASED DELAY
        console.log("[Seed Interactions] Refreshing all data...");
        await fetchCurrentData();
        setActionLoading(null);
    }
  };

  // Create demo group with celebrity leader
  const createDemoGroup = async (celebrityId: string, name: string, category: string) => {
    setActionLoading('group');
    
    try {
      const { data, error } = await supabase
        .from('groups')
        .insert([{
          name,
          description: `Demo group led by a celebrity in the ${category} industry`,
          category,
          created_by: celebrityId,
          is_demo: true,
        }]);
      
      if (error) {
        console.error('Error creating demo group:', error);
        toast.error('Failed to create demo group');
        return;
      }
      
      // Refresh groups
      const { data: refreshedGroups } = await supabase
        .from('groups')
        .select('*')
        .eq('is_demo', true);
      
      if (refreshedGroups) {
        setDemoGroupsCount(refreshedGroups.length);
      }
      
      toast.success('Demo group created successfully');
    } catch (error) {
      console.error('Error creating demo group:', error);
      toast.error('Failed to create demo group');
    } finally {
      await fetchCurrentData(); // Refresh counts after creating group
      setActionLoading(null);
    }
  };

  // Create demo event with celebrity host
  const createDemoEvent = async (celebrityId: string, title: string, category: string) => {
    setActionLoading('event');
    
    try {
      // Create a future date for the event (2 weeks from now)
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 14);
      
      const { data, error } = await supabase
        .from('events')
        .insert([{
          title,
          description: `Demo event hosted by a celebrity in the ${category} industry`,
          date: futureDate.toISOString(),
          location: 'Virtual',
          format: 'Conference',
          category,
          host_id: celebrityId,
          is_demo: true,
        }]);
      
      if (error) {
        console.error('Error creating demo event:', error);
        toast.error('Failed to create demo event');
        return;
      }
      
      // Refresh events
      const { data: refreshedEvents } = await supabase
        .from('events')
        .select('*')
        .eq('is_demo', true);
      
      if (refreshedEvents) {
        setDemoEventsCount(refreshedEvents.length);
      }
      
      toast.success('Demo event created successfully');
    } catch (error) {
      console.error('Error creating demo event:', error);
      toast.error('Failed to create demo event');
    } finally {
      await fetchCurrentData(); // Refresh counts after creating event
      setActionLoading(null);
    }
  };

  // Reset demo environment (Calls API, API should handle counts indirectly)
  const resetDemoEnvironment = async () => {
    if (!confirm('Are you sure you want to reset the demo environment? This will remove all demo data from the database.')) {
      return;
    }

    setActionLoading('reset');
    toast.loading('Resetting environment...');
    try {
      // Call the server-side API route to perform deletions
      const response = await fetch('/api/admin/reset-demo', {
        method: 'POST',
      });

      const result = await response.json();

      if (!response.ok) {
        console.error('Error resetting demo environment:', result);
        throw new Error(result.error || 'Failed to reset demo environment via API');
      }

      // Instead of resetting local state directly, fetch the new counts (which should be 0)
      console.log("[Reset Demo] Refreshing counts after reset...");
      await fetchCurrentData();
      toast.dismiss();
      toast.success('Demo environment reset successfully');

    } catch (error: any) {
      console.error('Error calling reset demo API:', error);
      toast.dismiss();
      toast.error(error.message || 'Failed to reset demo environment');
    } finally {
      setActionLoading(null);
    }
  };

  // Wrapper for refresh button to handle loading state
  const handleRefreshClick = async () => {
    setIsRefreshing(true);
    console.log("[Refresh Button] Manually refreshing data...");
    try {
      await fetchCurrentData();
      toast.success("Demo data status refreshed.");
    } catch (error: any) {
       toast.error(`Failed to refresh status: ${error.message}`);
    } finally {
       setIsRefreshing(false);
    }
  }

  // Helper to check if a specific action is loading
  const isLoading = (action: string | null) => actionLoading === action;
  const isAnyLoading = actionLoading !== null || isRefreshing;

  // UPDATED: Seed Demo Groups & Memberships (Calls combined API route)
  const seedDemoGroupsAndMemberships = async () => {
    setActionLoading('groups'); // Keep using 'groups' key for loading state
    toast.loading(`Requesting group and membership seeding...`);
    let success = false;
    let seededGroupsCount = 0;
    let seededMembershipsCount = 0;
    try {
        // Call the COMBINED route, sending SAMPLE_GROUPS
        const response = await fetch('/api/admin/seed-groups-and-memberships', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ groups: SAMPLE_GROUPS }), 
        });
        const result = await response.json();
        console.log("[Seed Groups/Memberships] API Response:", result);
        seededGroupsCount = result.seededGroups || 0;
        seededMembershipsCount = result.seededMemberships || 0;

        if (!response.ok) throw new Error(result.error || result.message || `API Error (${response.status})`);
        
        success = true;
        toast.dismiss();
        if (result.errors && result.errors.length > 0) {
             toast.error(`Seeding API reported ${result.errors.length} errors.`);
        } else {
             toast.success(result.message || `${seededGroupsCount} groups & ${seededMembershipsCount} memberships created.`);
             // *** OPTIMISTIC UI UPDATE ***
             if (seededGroupsCount > 0) {
                // Update the count immediately based on API response
                setDemoGroupsCount(currentCount => currentCount + seededGroupsCount); 
                console.log(`[Seed Groups/Memberships] Optimistically updated Groups count.`);
             } 
             // *** END OPTIMISTIC UI UPDATE ***
        }
    } catch (error: any) {
        console.error('[Seed Groups/Memberships] Error:', error);
        toast.dismiss();
        toast.error(`Failed to seed groups/memberships: ${error.message}`);
    } finally {
        // Keep the delay before refreshing overall counts
        console.log("[Seed Groups/Memberships] Waiting 5 seconds before refreshing counts...");
        await new Promise(resolve => setTimeout(resolve, 5000)); 
        console.log("[Seed Groups/Memberships] Refreshing all data...");
        await fetchCurrentData(); // Fetch final consistent state
        setActionLoading(null);
    }
  };

  // UPDATED: Seed Demo Events & Attendance (Calls combined API route)
  const seedDemoEventsAndAttendance = async () => {
    setActionLoading('events'); // Use 'events' key for loading state
    toast.loading(`Requesting event and attendance seeding...`);
    let success = false;
    let seededEventsCount = 0;
    let seededAttendanceCount = 0;
    try {
        // Call the COMBINED route, sending SAMPLE_EVENTS
        const response = await fetch('/api/admin/seed-events-and-attendance', { // *** NEW ROUTE ***
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ events: SAMPLE_EVENTS }), // Send sample event data
        });
        const result = await response.json();
        console.log("[Seed Events/Attendance] API Response:", result);
        seededEventsCount = result.seededEvents || 0;
        seededAttendanceCount = result.seededAttendance || 0;

        if (!response.ok && response.status !== 207) { // Allow 207 Multi-Status (partial success)
             throw new Error(result.error || result.message || `API Error (${response.status})`);
        }
        
        success = true;
        toast.dismiss();
        if (result.errors && result.errors.length > 0) {
             toast.error(`Seeding API reported ${result.errors.length} errors during attendance insert.`);
             // Still might have seeded events, apply optimistic update for events
             if (seededEventsCount > 0) {
                setDemoEventsCount(currentCount => currentCount + seededEventsCount);
                console.log(`[Seed Events/Attendance] Optimistically updated Events count (despite attendance errors).`);
             }
        } else {
             toast.success(result.message || `${seededEventsCount} events & ${seededAttendanceCount} attendance records created.`);
             // *** OPTIMISTIC UI UPDATE ***
             if (seededEventsCount > 0) {
                // Update the count immediately based on API response
                setDemoEventsCount(currentCount => currentCount + seededEventsCount); 
                console.log(`[Seed Events/Attendance] Optimistically updated Events count.`);
             } 
             // *** END OPTIMISTIC UI UPDATE ***
        }
    } catch (error: any) {
        console.error('[Seed Events/Attendance] Error:', error);
        toast.dismiss();
        toast.error(`Failed to seed events/attendance: ${error.message}`);
    } finally {
        // Keep the delay before refreshing overall counts
        console.log("[Seed Events/Attendance] Waiting 5 seconds before refreshing counts...");
        await new Promise(resolve => setTimeout(resolve, 5000)); 
        console.log("[Seed Events/Attendance] Refreshing all data...");
        await fetchCurrentData(); // Fetch final consistent state
        setActionLoading(null);
    }
  };

  // ADDED: Fetch Recommendations Function
  const fetchRecommendations = async (type: 'profile' | 'group' | 'event') => {
    if (!selectedProfileId) {
      toast.error('Please select a demo profile first.');
      return;
    }
    setRecommendationLoading(true);
    setProfileRecommendations([]); // Clear previous results
    setGroupRecommendations([]);
    setEventRecommendations([]);

    // *** Get Access Token from Supabase Session ***
    let accessToken: string | null = null;
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.access_token) {
        accessToken = session.access_token;
      } else {
        console.warn("[Recommendations] No active session or access token found.");
        // Depending on requirements, you might want to toast.error or redirect here
        // For now, we'll proceed without the token and let the API handle the 401
      }
    } catch (error) {
      console.error("[Recommendations] Error getting auth session:", error);
      toast.error("Could not retrieve authentication details.");
      setRecommendationLoading(false);
      return;
    }
    // *** End Get Access Token ***

    // *** IMPORTANT: Adjust the API endpoint URL and parameters as needed ***
    const apiUrl = `/api/recommendations?profile_id=${selectedProfileId}&type=${type}&limit=5`; 
    console.log(`[Recommendations] Fetching ${type} recommendations from: ${apiUrl}`);

    // *** Prepare Headers ***
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    if (accessToken) {
      headers['Authorization'] = `Bearer ${accessToken}`;
      console.log("[Recommendations] Authorization header added.");
    }
    // *** End Prepare Headers ***
    
    try {
      const response = await fetch(apiUrl, { // Use GET by default
         method: 'GET', // Explicitly setting GET
         headers: headers,
         credentials: 'include' // Keep include for potential cookie fallback/other APIs
      }); 
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.message || `API Error (${response.status})`);
      }

      console.log(`[Recommendations] Received ${type} recommendations:`, data);

      // Assuming the API returns an array of recommendations under a key like 'recommendations'
      const recommendations = data.recommendations || []; 

      if (type === 'profile') {
        setProfileRecommendations(recommendations);
      } else if (type === 'group') {
        setGroupRecommendations(recommendations);
      } else {
        setEventRecommendations(recommendations);
      }
      toast.success(`Fetched ${type} recommendations for the selected profile.`);

    } catch (error: any) {
      console.error(`[Recommendations] Error fetching ${type} recommendations:`, error);
      toast.error(`Failed to fetch ${type} recommendations: ${error.message}`);
    } finally {
      setRecommendationLoading(false);
    }
  };
  // END ADDED Fetch Function

  // Unified handler for prospect form inputs
  const handleProspectInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;

    // Handle specific array-like string inputs separately
    if (name === 'skillsString' || name === 'interestsString' || name === 'goalsString' || name === 'valuesString') {
        setProspectArrayInputs(prev => ({ ...prev, [name]: value }));
    } else {
        // Handle regular profile fields
        setProspectData(prev => ({ ...prev, [name]: value }));
    }
  };

  // Function to handle seeding the prospect profile (MODIFIED)
  const seedProspectProfile = async () => {
    // Basic validation (e.g., require name and email)
    if (!prospectData.first_name || !prospectData.last_name || !prospectData.email) {
      toast.error('Please enter at least First Name, Last Name, and Email for the prospect.');
      return;
    }

    setActionLoading('prospect');
    toast.loading('Seeding prospect profile...');
    setSeededProspectId(null); // Clear old ID before seeding new one
    setSeededProspectName(null);

    // Prepare data for API (convert comma-separated strings to arrays)
    const apiData = {
      ...prospectData,
      skills: prospectArrayInputs.skillsString?.split(',').map(s => s.trim()).filter(Boolean) || [],
      interests: prospectArrayInputs.interestsString?.split(',').map(i => i.trim()).filter(Boolean) || [],
      professional_goals: prospectArrayInputs.goalsString?.split(',').map(g => g.trim()).filter(Boolean) || [],
      values: prospectArrayInputs.valuesString?.split(',').map(v => v.trim()).filter(Boolean) || [],
    };

    console.log("[Seed Prospect] Sending data to API:", apiData);

    try {
      const response = await fetch('/api/admin/seed-prospect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiData),
      });

      const result = await response.json();
      console.log("[Seed Prospect] API Response:", result);

      if (!response.ok) {
        throw new Error(result.error || result.message || `API Error (${response.status})`);
      }

      toast.dismiss();
      toast.success('Prospect profile seeded successfully!');
      
      // Store the new prospect's ID and name
      if (result.profileId) {
         setSeededProspectId(result.profileId);
         const fullName = `${prospectData.first_name || ''} ${prospectData.last_name || ''}`.trim();
         setSeededProspectName(fullName || 'Prospect');
         setSelectedProfileId(result.profileId); // Auto-select in showcase
      }
      
      // Optionally clear the form
      // setProspectData({ ...initialProspectState }); 
      // setProspectArrayInputs({ ...initialArrayInputState });
      
      // Refresh data to include the new prospect in the dropdown
      await fetchCurrentData(); 

    } catch (error: any) {
      console.error('[Seed Prospect] Error:', error);
      toast.dismiss();
      toast.error(`Failed to seed prospect: ${error.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // ADDED: Function to handle logging in as prospect
  const handleLoginAsProspect = async () => {
    if (!seededProspectId) {
      toast.error("No prospect has been seeded yet.");
      return;
    }

    setActionLoading('loginAsProspect');
    toast.loading('Generating sign-in link...');

    try {
       const response = await fetch('/api/admin/generate-signin-link', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ userId: seededProspectId }),
       });

       const result = await response.json();
       console.log("[Login As Prospect] API Response:", result);

       if (!response.ok) {
         throw new Error(result.error || result.message || `API Error (${response.status})`);
       }

       if (!result.signInLink) {
         throw new Error('Sign-in link was not returned from the API.');
       }
       
       toast.dismiss();
       toast.success('Redirecting to sign-in link...');
       
       // Redirect the browser to the magic link
       window.location.href = result.signInLink;

    } catch (error: any) {
       console.error('[Login As Prospect] Error:', error);
       toast.dismiss();
       toast.error(`Failed to generate sign-in link: ${error.message}`);
    } finally {
       // Keep loading state potentially, as browser will navigate away
       // setActionLoading(null); 
    }
  };

  return (
    <div className="py-6">
      <div className="px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <BeakerIcon className="h-8 w-8 text-indigo-600 mr-2" />
            <h1 className="text-2xl font-semibold text-gray-900">Demo Environment</h1>
          </div>
          <Link
            href="/admin"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark transition-colors"
          >
            <HomeIcon className="h-5 w-5 mr-2" aria-hidden="true" />
            Back to Dashboard
          </Link>
        </div>
        
        <p className="mt-2 text-sm text-gray-500 max-w-4xl">
          Create and manage a demonstration environment with celebrity profiles, groups, and events.
          This environment is perfect for showcasing the platform&apos;s matching capabilities during demos.
        </p>
        
        {/* Demo Mode & Status Section */}
        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* Demo Mode Toggle Card */}
          <div className="bg-white shadow rounded-lg p-6 flex flex-col justify-between">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-lg font-medium text-gray-900 flex items-center">
                  <BeakerIcon className="h-6 w-6 mr-2 text-indigo-500"/>
                  Demo Mode Status
                </h2>
                <p className="text-sm text-gray-500 mt-1">
                  Toggle demonstration features for the platform.
                </p>
              </div>
              <button
                onClick={toggleDemoMode}
                disabled={isAnyLoading || loading}
                className={`inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors disabled:opacity-50 ${ 
                  demoMode 
                    ? 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300 focus:ring-indigo-500'
                }`}
              >
                {loading ? (
                  <ArrowPathIcon className="h-5 w-5 animate-spin" />
                ) : demoMode ? (
                  <>
                    <CheckCircleIcon className="h-5 w-5 mr-2" /> Enabled
                  </>
                ) : (
                  <>
                    <ExclamationCircleIcon className="h-5 w-5 mr-2" /> Disabled
                  </>
                )}
              </button>
            </div>
             <p className="text-xs text-gray-400 mt-4">
                Note: This toggle is currently simulated and does not affect live system settings.
              </p>
          </div>

          {/* Demo Data Status Card */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
              <InformationCircleIcon className="h-6 w-6 mr-2 text-blue-500"/>
              Demo Data Status
            </h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-gray-600 flex items-center">
                  <UserCircleIcon className="h-5 w-5 mr-2 text-gray-400"/> Celebrity Profiles:
                </span>
                <span className={`font-medium px-2 py-0.5 rounded-full text-xs ${demoProfilesCount > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                  {loading ? 'Loading...' : `${demoProfilesCount} Seeded`}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 flex items-center">
                  <UsersIcon className="h-5 w-5 mr-2 text-gray-400"/> Demo Connections:
                </span>
                 <span className={`font-medium px-2 py-0.5 rounded-full text-xs ${demoConnectionsCount > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                  {loading ? 'Loading...' : `${demoConnectionsCount} Seeded`}
                </span>
              </div>
               <div className="flex justify-between items-center">
                <span className="text-gray-600 flex items-center">
                  <ChatBubbleOvalLeftEllipsisIcon className="h-5 w-5 mr-2 text-gray-400"/> Demo Interactions:
                </span>
                 <span className={`font-medium px-2 py-0.5 rounded-full text-xs ${demoInteractionsCount > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                  {loading ? 'Loading...' : `${demoInteractionsCount} Seeded`}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 flex items-center">
                  <UserGroupIcon className="h-5 w-5 mr-2 text-gray-400"/> Demo Groups:
                </span>
                 <span className={`font-medium px-2 py-0.5 rounded-full text-xs ${demoGroupsCount > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                  {loading ? 'Loading...' : `${demoGroupsCount} Seeded`}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 flex items-center">
                  <CalendarIcon className="h-5 w-5 mr-2 text-gray-400"/> Demo Events:
                </span>
                 <span className={`font-medium px-2 py-0.5 rounded-full text-xs ${demoEventsCount > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                  {loading ? 'Loading...' : `${demoEventsCount} Seeded`}
                </span>
              </div>
            </div>
          </div>
        </div>
        
         {/* Seeding Actions (Updated with Filter) */}
        <div className="mt-6 bg-white shadow rounded-lg p-6">
           <h2 className="text-lg font-medium text-gray-900 mb-4">Seeding Actions</h2>
           
           {/* Filter Selection */}
           <div className="mb-6">
             <label className="block text-sm font-medium text-gray-700 mb-2">Filter Celebrity Profiles:</label>
             <div className="flex space-x-4">
               {(['all', 'tech', 'media'] as const).map((filterOption) => (
                 <label key={filterOption} className="inline-flex items-center">
                   <input 
                     type="radio" 
                     className="form-radio h-4 w-4 text-indigo-600 transition duration-150 ease-in-out" 
                     name="celebrityFilter"
                     value={filterOption}
                     checked={celebrityFilter === filterOption}
                     onChange={() => setCelebrityFilter(filterOption)}
                     disabled={isAnyLoading}
                   />
                   <span className="ml-2 text-sm text-gray-700 capitalize">
                     {filterOption === 'tech' ? 'Tech/Entrepreneur' : 
                      filterOption === 'media' ? 'Media/Music' : 'All'}
                   </span>
                 </label>
               ))}
             </div>
           </div>

           <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {/* Seed Profiles Button */}
              <button
                onClick={seedCelebrityProfiles}
                disabled={isAnyLoading}
                className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading('profiles') ? (
                  <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                ) : (
                  <UserCircleIcon className="h-5 w-5 mr-2" />
                )}
                Seed {celebrityFilter !== 'all' ? celebrityFilter.charAt(0).toUpperCase() + celebrityFilter.slice(1) : 'All'} Profiles
              </button>

               {/* Seed Connections Button */}
              <button
                onClick={() => seedDemoConnections(50)}
                disabled={isAnyLoading}
                className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-teal-600 hover:bg-teal-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading('connections') ? (
                  <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                ) : (
                  <UsersIcon className="h-5 w-5 mr-2" />
                )}
                Seed Demo Connections ({demoConnectionsCount})
              </button>

               {/* Seed Interactions Button */}
               <button
                onClick={() => seedDemoInteractions(100)}
                disabled={isAnyLoading}
                className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-cyan-600 hover:bg-cyan-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading('interactions') ? (
                  <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                ) : (
                  <ChatBubbleOvalLeftEllipsisIcon className="h-5 w-5 mr-2" />
                )}
                Seed Demo Interactions ({demoInteractionsCount})
              </button>

               {/* UPDATED: Seed Groups & Memberships Button */}
               <button
                 onClick={seedDemoGroupsAndMemberships} // Call the combined function
                 disabled={isAnyLoading}
                 className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-purple-600 hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
               >
                {isLoading('groups') ? (
                    <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                 ) : (
                     <UserGroupIcon className="h-5 w-5 mr-2" />
                 )}
                 Seed Groups & Memberships ({demoGroupsCount}) 
               </button>

               {/* Seed Events Button - UPDATED onClick */}
               <div className="flex-1">
                 <button
                   onClick={seedDemoEventsAndAttendance} // *** UPDATED HANDLER ***
                   disabled={isAnyLoading || demoProfilesCount === 0}
                   title={demoProfilesCount === 0 ? "Seed profiles first" : "Seed demo events and attendance"}
                   className="w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                   {isLoading('events') ? (
                     <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                   ) : (
                     <CalendarIcon className="h-5 w-5 mr-2" />
                   )}
                   Seed Demo Events
                 </button>
               </div>
           </div>
        </div>
        
        {/* Celebrity Profiles List (Modified to remove individual actions) */}
        <div className="mt-6 bg-white shadow rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium text-gray-900">Demo Profiles ({demoProfilesCount})</h2>
          </div>
          
          {/* Celebrity List Table */}
          <div className="mt-4 flow-root">
            <div className="-mx-4 -my-2 overflow-x-auto sm:-mx-6 lg:-mx-8">
              <div className="inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead>
                    <tr>
                      <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-0">Name</th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Industry</th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Location</th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Skills</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {loading ? (
                       <tr><td colSpan={4} className="py-4 text-center text-sm text-gray-500">Loading profiles...</td></tr>
                    ) : demoProfiles.length === 0 ? (
                      <tr><td colSpan={4} className="py-4 pl-4 pr-3 text-sm text-gray-500 text-center sm:pl-0">No demo profiles found.</td></tr>
                    ) : (
                      demoProfiles.map((profile) => (
                        <tr key={profile.id}>
                          <td className="py-4 pl-4 pr-3 text-sm sm:pl-0">
                            <div className="flex items-center">
                              <div className="h-10 w-10 flex-shrink-0">
                                <img className="h-10 w-10 rounded-full" src={profile.avatar_url || `https://ui-avatars.com/api/?name=${profile.first_name}+${profile.last_name}&background=random`} alt="" />
                              </div>
                              <div className="ml-4">
                                <div className="font-medium text-gray-900">{`${profile.first_name ?? ''} ${profile.last_name ?? ''}`}</div>
                                <div className="text-gray-500 text-xs">{profile.title}</div>
                              </div>
                            </div>
                          </td>
                          <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{profile.industry}</td>
                          <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{profile.location}</td>
                          <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                            {Array.isArray(profile.skills) ? profile.skills.slice(0, 3).join(', ') + (profile.skills.length > 3 ? '...' : '') : 'N/A'}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* ADDED: Recommendation Showcase Section */}
        <div className="mt-6 bg-white shadow rounded-lg p-6">
           <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <PresentationChartLineIcon className="h-6 w-6 mr-2 text-green-600"/>
            Recommendation Showcase
           </h2>
           <p className="text-sm text-gray-500 mb-4">
            Select a seeded demo profile below to fetch recommendations using the live algorithm.
           </p>

           {demoProfilesCount === 0 ? (
              <p className="text-sm text-gray-500 italic">
                Seed some profiles first to enable the recommendation showcase.
              </p>
           ) : (
             <>
              {/* Profile Selector */}
              <div className="mb-4">
                 <label htmlFor="profileSelect" className="block text-sm font-medium text-gray-700">
                    Select Demo Profile:
                 </label>
                 <select
                    id="profileSelect"
                    name="profileSelect"
                    value={selectedProfileId}
                    onChange={(e) => {
                      setSelectedProfileId(e.target.value);
                      // Clear old recommendations when profile changes
                      setProfileRecommendations([]);
                      setGroupRecommendations([]);
                      setEventRecommendations([]);
                    }}
                    disabled={recommendationLoading || isAnyLoading}
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md disabled:opacity-50"
                  >
                    <option value="" disabled>-- Select a profile --</option>
                    {demoProfiles.map((profile) => (
                      <option key={profile.id} value={profile.id}>
                        {profile.first_name} {profile.last_name} ({profile.title || 'N/A'})
                      </option>
                    ))}
                  </select>
              </div>

              {/* Recommendation Buttons */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 mb-6">
                 <button
                   onClick={() => fetchRecommendations('profile')}
                   disabled={!selectedProfileId || recommendationLoading || isAnyLoading}
                   className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                   {recommendationLoading ? <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin"/> : <UsersIcon className="h-5 w-5 mr-2"/>}
                   Get Profile Recommendations
                 </button>
                 <button
                   onClick={() => fetchRecommendations('group')}
                   disabled={!selectedProfileId || recommendationLoading || isAnyLoading}
                   className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-purple-600 hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                   {recommendationLoading ? <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin"/> : <UserGroupIcon className="h-5 w-5 mr-2"/>}
                   Get Group Recommendations
                 </button>
                 <button
                   onClick={() => fetchRecommendations('event')}
                   disabled={!selectedProfileId || recommendationLoading || isAnyLoading}
                   className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-pink-600 hover:bg-pink-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                 >
                   {recommendationLoading ? <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin"/> : <CalendarIcon className="h-5 w-5 mr-2"/>}
                   Get Event Recommendations
                 </button>
              </div>

              {/* Recommendation Results */}
              {(profileRecommendations.length > 0 || groupRecommendations.length > 0 || eventRecommendations.length > 0) && (
                 <div className="mt-4 space-y-4">
                    {profileRecommendations.length > 0 && (
                      <div>
                         <h3 className="text-md font-medium text-gray-800 mb-2">Profile Recommendations:</h3>
                         <ul className="divide-y divide-gray-200 border rounded-md">
                           {profileRecommendations.map(rec => (
                             <li key={rec.id} className="px-4 py-3 text-sm">
                               <span className="font-medium text-gray-900">{rec.name || rec.id}</span>
                               {rec.reason && <span className="text-gray-500 ml-2 italic">({rec.reason})</span>}
                               {rec.score && <span className="text-gray-500 ml-2 text-xs">(Score: {rec.score.toFixed(3)})</span>}
                             </li>
                           ))}
                         </ul>
                      </div>
                    )}
                    {groupRecommendations.length > 0 && (
                       <div>
                         <h3 className="text-md font-medium text-gray-800 mb-2">Group Recommendations:</h3>
                          <ul className="divide-y divide-gray-200 border rounded-md">
                           {groupRecommendations.map(rec => (
                             <li key={rec.id} className="px-4 py-3 text-sm">
                               <span className="font-medium text-gray-900">{rec.name || rec.id}</span>
                               {rec.reason && <span className="text-gray-500 ml-2 italic">({rec.reason})</span>}
                               {rec.score && <span className="text-gray-500 ml-2 text-xs">(Score: {rec.score.toFixed(3)})</span>}
                             </li>
                           ))}
                         </ul>
                       </div>
                    )}
                    {eventRecommendations.length > 0 && (
                       <div>
                         <h3 className="text-md font-medium text-gray-800 mb-2">Event Recommendations:</h3>
                         <ul className="divide-y divide-gray-200 border rounded-md">
                           {eventRecommendations.map(rec => (
                             <li key={rec.id} className="px-4 py-3 text-sm">
                               <span className="font-medium text-gray-900">{rec.title || rec.id}</span>
                               {rec.reason && <span className="text-gray-500 ml-2 italic">({rec.reason})</span>}
                               {rec.score && <span className="text-gray-500 ml-2 text-xs">(Score: {rec.score.toFixed(3)})</span>}
                             </li>
                           ))}
                         </ul>
                       </div>
                    )}
                 </div>
              )}
             </>
           )}
        </div>
        {/* END ADDED Recommendation Showcase Section */}

        {/* Demo Groups and Events */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Demo Groups */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Demo Groups</h2>
            {demoGroupsCount === 0 ? (
              <p className="text-sm text-gray-500">
                No demo groups found. Create groups using the actions in the Celebrity Profiles section.
              </p>
            ) : (
              <ul className="space-y-4">
                {/* Add group list rendering logic here */}
              </ul>
            )}
          </div>
          
          {/* Demo Events */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Demo Events</h2>
            {demoEventsCount === 0 ? (
              <p className="text-sm text-gray-500">
                No demo events found. Create events using the actions in the Celebrity Profiles section.
              </p>
            ) : (
              <ul className="space-y-4">
                {/* Add event list rendering logic here */}
              </ul>
            )}
          </div>
        </div>
        
        {/* Reset Demo Environment Button */}
        <div className="mt-8 border-t pt-6 flex justify-end">
          <button
            onClick={resetDemoEnvironment}
            disabled={isAnyLoading}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading('reset') ? (
              <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
            ) : (
              <ArrowPathIcon className="h-5 w-5 mr-2" />
            )}
            Reset Demo Environment
          </button>
        </div>

        {/* Refresh Status Button */}
        <div className="mt-8 border-t pt-6 flex justify-end">
          <button
            onClick={handleRefreshClick}
            disabled={isAnyLoading}
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRefreshing ? (
              <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
            ) : (
              <ArrowPathIcon className="h-5 w-5 mr-2" />
            )}
            Refresh Status
          </button>
        </div>

        {/* ADDED: Prospect Seeding Form Section (MODIFIED) */}
        <div className="mt-6 bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <UserCircleIcon className="h-6 w-6 mr-2 text-orange-600" />
            Seed Custom Prospect Profile
          </h2>
          <p className="text-sm text-gray-500 mb-6">
            Enter the details of your current prospect. This will remove any previous prospect profile and create this new one as demo data.
            You can then select this prospect in the "Recommendation Showcase" below.
          </p>
          
          <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-6">
            {/* Form Fields - Point inputs to correct state */}
            <div className="sm:col-span-3">
              <label htmlFor="first_name" className="block text-sm font-medium text-gray-700">First Name *</label>
              <input type="text" name="first_name" id="first_name" value={prospectData.first_name} onChange={handleProspectInputChange} required className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            <div className="sm:col-span-3">
              <label htmlFor="last_name" className="block text-sm font-medium text-gray-700">Last Name *</label>
              <input type="text" name="last_name" id="last_name" value={prospectData.last_name} onChange={handleProspectInputChange} required className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            <div className="sm:col-span-3">
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email *</label>
              <input type="email" name="email" id="email" value={prospectData.email} onChange={handleProspectInputChange} required className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
             <div className="sm:col-span-3">
              <label htmlFor="title" className="block text-sm font-medium text-gray-700">Title</label>
              <input type="text" name="title" id="title" value={prospectData.title} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
             <div className="sm:col-span-3">
              <label htmlFor="company" className="block text-sm font-medium text-gray-700">Company</label>
              <input type="text" name="company" id="company" value={prospectData.company} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
             <div className="sm:col-span-3">
              <label htmlFor="industry" className="block text-sm font-medium text-gray-700">Industry</label>
              <input type="text" name="industry" id="industry" value={prospectData.industry} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            <div className="sm:col-span-3">
              <label htmlFor="location" className="block text-sm font-medium text-gray-700">Location</label>
              <input type="text" name="location" id="location" value={prospectData.location} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
             <div className="sm:col-span-3">
              <label htmlFor="website" className="block text-sm font-medium text-gray-700">Website</label>
              <input type="url" name="website" id="website" value={prospectData.website || ''} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            <div className="sm:col-span-6">
              <label htmlFor="bio" className="block text-sm font-medium text-gray-700">Bio</label>
              <textarea name="bio" id="bio" value={prospectData.bio} onChange={handleProspectInputChange} rows={3} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            {/* Array Inputs (use prospectArrayInputs state) */}
             <div className="sm:col-span-3">
              <label htmlFor="skillsString" className="block text-sm font-medium text-gray-700">Skills (comma-separated)</label>
              <input type="text" name="skillsString" id="skillsString" value={prospectArrayInputs.skillsString} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
             <div className="sm:col-span-3">
              <label htmlFor="interestsString" className="block text-sm font-medium text-gray-700">Interests (comma-separated)</label>
              <input type="text" name="interestsString" id="interestsString" value={prospectArrayInputs.interestsString} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            <div className="sm:col-span-3">
              <label htmlFor="goalsString" className="block text-sm font-medium text-gray-700">Prof. Goals (comma-separated)</label>
              <input type="text" name="goalsString" id="goalsString" value={prospectArrayInputs.goalsString} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
            <div className="sm:col-span-3">
              <label htmlFor="valuesString" className="block text-sm font-medium text-gray-700">Values (comma-separated)</label>
              <input type="text" name="valuesString" id="valuesString" value={prospectArrayInputs.valuesString} onChange={handleProspectInputChange} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" />
            </div>
          </div>
          
          {/* Submit Buttons Area */}
          <div className="mt-6 flex justify-end space-x-3">
             {/* Log in As Button - ADDED */}
            <button
              type="button"
              onClick={handleLoginAsProspect}
              disabled={!seededProspectId || isAnyLoading || actionLoading === 'loginAsProspect'}
              title={!seededProspectId ? "Seed prospect profile first" : `Log in as ${seededProspectName}`}
              className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {actionLoading === 'loginAsProspect' ? (
                <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
              ) : (
                <KeyIcon className="h-5 w-5 mr-2" />
              )}
              Log in as {seededProspectName || 'Prospect'}
            </button>
            
            {/* Seed Prospect Button - Existing */}
            <button
              type="button"
              onClick={seedProspectProfile}
              disabled={actionLoading === 'prospect' || isAnyLoading}
              className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-orange-600 hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-orange-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {actionLoading === 'prospect' ? (
                <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
              ) : (
                <PlusCircleIcon className="h-5 w-5 mr-2" />
              )}
              Seed Prospect Profile
            </button>
          </div>
        </div>
        {/* END Prospect Seeding Form Section */}
      </div>
    </div>
  );
} 