'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';

interface UserProfile {
  id: string;
  full_name: string;
  headline: string;
  bio: string;
  location: string;
  website: string;
  avatar_url: string | null;
  skills: string[];
  interests: string[];
  experience: Experience[];
  education: Education[];
}

interface Experience {
  id: number;
  company: string;
  title: string;
  start_date: string;
  end_date: string | null;
  description: string;
}

interface Education {
  id: number;
  institution: string;
  degree: string;
  field: string;
  start_date: string;
  end_date: string | null;
}

export default function ProfilePage() {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<Partial<UserProfile>>({});
  const [isSaving, setIsSaving] = useState(false);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUser(session.user);
      }
    };

    getUser();
  }, [supabase.auth]);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) return;
      
      setLoading(true);
      
      // In a real app, this would be fetched from the database
      // For this example, we'll use mock data
      const mockProfile: UserProfile = {
        id: user.id,
        full_name: user.user_metadata?.full_name || user.email?.split('@')[0] || '',
        headline: 'Product Manager at Tech Company',
        bio: 'Experienced product manager with a passion for user-centered design and data-driven decision making. Looking to connect with other product professionals and designers.',
        location: 'San Francisco, CA',
        website: 'https://example.com',
        avatar_url: user.user_metadata?.avatar_url || null,
        skills: ['Product Management', 'User Research', 'Data Analysis', 'Agile Methodologies', 'Wireframing'],
        interests: ['Technology', 'Design', 'Entrepreneurship', 'Artificial Intelligence'],
        experience: [
          {
            id: 1,
            company: 'Tech Company',
            title: 'Senior Product Manager',
            start_date: '2020-06',
            end_date: null,
            description: 'Leading product strategy and execution for the company\'s main B2B SaaS platform.'
          },
          {
            id: 2,
            company: 'Digital Agency',
            title: 'Product Manager',
            start_date: '2018-03',
            end_date: '2020-05',
            description: 'Managed the development of web and mobile applications for various clients.'
          }
        ],
        education: [
          {
            id: 1,
            institution: 'University of California, Berkeley',
            degree: 'Bachelor of Science',
            field: 'Computer Science',
            start_date: '2014-09',
            end_date: '2018-05'
          }
        ]
      };
      
      setProfile(mockProfile);
      setFormData(mockProfile);
      setLoading(false);
    };

    fetchProfile();
  }, [user]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleArrayChange = (value: string, field: 'skills' | 'interests', action: 'add' | 'remove') => {
    if (!formData[field]) return;
    
    if (action === 'add' && value.trim()) {
      setFormData(prev => ({
        ...prev,
        [field]: [...(prev[field] as string[] || []), value.trim()]
      }));
    } else if (action === 'remove') {
      setFormData(prev => ({
        ...prev,
        [field]: (prev[field] as string[] || []).filter(item => item !== value)
      }));
    }
  };

  const handleSkillsInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && e.currentTarget.value.trim()) {
      e.preventDefault();
      handleArrayChange(e.currentTarget.value, 'skills', 'add');
      e.currentTarget.value = '';
    }
  };

  const handleInterestsInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && e.currentTarget.value.trim()) {
      e.preventDefault();
      handleArrayChange(e.currentTarget.value, 'interests', 'add');
      e.currentTarget.value = '';
    }
  };

  const handleSaveProfile = async () => {
    setIsSaving(true);
    
    // In a real app, you would submit this to your API/database
    console.log('Saving profile:', formData);
    
    // Simulate API call
    setTimeout(() => {
      setProfile(formData as UserProfile);
      setIsEditing(false);
      setIsSaving(false);
    }, 1000);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">My Profile</h1>
        <p className="text-muted-foreground mt-2">
          Manage your professional profile information.
        </p>
      </div>

      <div className="bg-white rounded-lg border shadow-sm overflow-hidden">
        {/* Profile Header */}
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 h-40 relative">
          {/* Avatar */}
          <div className="absolute bottom-0 left-8 transform translate-y-1/2">
            <div className="w-32 h-32 rounded-full border-4 border-white bg-white overflow-hidden">
              {profile?.avatar_url ? (
                <img 
                  src={profile.avatar_url} 
                  alt={profile.full_name} 
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-200 text-gray-600 text-4xl font-bold">
                  {profile?.full_name?.charAt(0) || user?.email?.charAt(0) || '?'}
                </div>
              )}
            </div>
          </div>
          
          {/* Edit Button */}
          <div className="absolute bottom-4 right-4">
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="px-4 py-2 bg-white rounded-md text-sm font-medium text-gray-700 shadow hover:bg-gray-50"
            >
              {isEditing ? 'Cancel' : 'Edit Profile'}
            </button>
          </div>
        </div>
        
        {/* Profile Content */}
        <div className="mt-20 p-8">
          {isEditing ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                  <input
                    type="text"
                    name="full_name"
                    value={formData.full_name || ''}
                    onChange={handleInputChange}
                    className="w-full p-2 border rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Professional Headline</label>
                  <input
                    type="text"
                    name="headline"
                    value={formData.headline || ''}
                    onChange={handleInputChange}
                    className="w-full p-2 border rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location || ''}
                    onChange={handleInputChange}
                    className="w-full p-2 border rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Website</label>
                  <input
                    type="text"
                    name="website"
                    value={formData.website || ''}
                    onChange={handleInputChange}
                    className="w-full p-2 border rounded-md"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Bio</label>
                <textarea
                  name="bio"
                  rows={4}
                  value={formData.bio || ''}
                  onChange={handleInputChange}
                  className="w-full p-2 border rounded-md"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Skills</label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {formData.skills?.map(skill => (
                    <span key={skill} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm flex items-center">
                      {skill}
                      <button 
                        onClick={() => handleArrayChange(skill, 'skills', 'remove')}
                        className="ml-1 text-blue-800 hover:text-blue-900"
                      >
                        &times;
                      </button>
                    </span>
                  ))}
                </div>
                <input
                  type="text"
                  placeholder="Add a skill (press Enter)"
                  className="w-full p-2 border rounded-md"
                  onKeyDown={handleSkillsInputKeyDown}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Interests</label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {formData.interests?.map(interest => (
                    <span key={interest} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm flex items-center">
                      {interest}
                      <button 
                        onClick={() => handleArrayChange(interest, 'interests', 'remove')}
                        className="ml-1 text-green-800 hover:text-green-900"
                      >
                        &times;
                      </button>
                    </span>
                  ))}
                </div>
                <input
                  type="text"
                  placeholder="Add an interest (press Enter)"
                  className="w-full p-2 border rounded-md"
                  onKeyDown={handleInterestsInputKeyDown}
                />
              </div>
              
              <div className="pt-4 flex justify-end">
                <button
                  onClick={handleSaveProfile}
                  disabled={isSaving}
                  className="px-6 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save Profile'}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold">{profile?.full_name}</h2>
                <p className="text-gray-600 mt-1">{profile?.headline}</p>
                <div className="flex items-center mt-2 text-gray-500 text-sm">
                  <svg className="h-5 w-5 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                  </svg>
                  {profile?.location}
                </div>
                {profile?.website && (
                  <div className="flex items-center mt-1 text-blue-600 text-sm">
                    <svg className="h-5 w-5 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z" clipRule="evenodd" />
                    </svg>
                    <a href={profile.website} target="_blank" rel="noopener noreferrer">
                      {profile.website}
                    </a>
                  </div>
                )}
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">About</h3>
                <p className="text-gray-600">{profile?.bio}</p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Skills</h3>
                <div className="flex flex-wrap gap-2">
                  {profile?.skills.map(skill => (
                    <span key={skill} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Interests</h3>
                <div className="flex flex-wrap gap-2">
                  {profile?.interests.map(interest => (
                    <span key={interest} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                      {interest}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-4">Experience</h3>
                <div className="space-y-4">
                  {profile?.experience.map(exp => (
                    <div key={exp.id} className="border-l-2 border-gray-200 pl-4">
                      <h4 className="font-medium">{exp.title}</h4>
                      <p className="text-gray-600">{exp.company}</p>
                      <p className="text-sm text-gray-500">
                        {new Date(exp.start_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} - 
                        {exp.end_date ? new Date(exp.end_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : ' Present'}
                      </p>
                      <p className="text-sm text-gray-600 mt-2">{exp.description}</p>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-4">Education</h3>
                <div className="space-y-4">
                  {profile?.education.map(edu => (
                    <div key={edu.id} className="border-l-2 border-gray-200 pl-4">
                      <h4 className="font-medium">{edu.institution}</h4>
                      <p className="text-gray-600">{edu.degree}, {edu.field}</p>
                      <p className="text-sm text-gray-500">
                        {new Date(edu.start_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} - 
                        {edu.end_date ? new Date(edu.end_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : ' Present'}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 