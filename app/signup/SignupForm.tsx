'use client';

import React, { useState } from 'react';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import MultiSelect from '../components/MultiSelect';
import { skillOptions, interestOptions, goalOptions, valueOptions, lifestyleOptions } from '../lib/options';

const sections = [
  { id: 1, title: 'Basic Information', description: 'Let\'s start with the essentials' },
  { id: 2, title: 'Profile & Interests', description: 'Tell us about yourself and what you love' },
  { id: 3, title: 'Goals & Values', description: 'Help us understand what matters to you' },
];

export default function SignupForm() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    bio: '',
    skills: [] as string[],
    interests: [] as string[],
    goals: [] as string[],
    values: [] as string[],
    lifestyle: [] as string[],
    profileVisibility: 'public' as 'public' | 'private' | 'connections',
    emailNotifications: true,
    marketingEmails: false,
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentSection, setCurrentSection] = useState(1);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          firstName: formData.firstName,
          lastName: formData.lastName,
          fullName: `${formData.firstName} ${formData.lastName}`,
          bio: formData.bio,
          skills: formData.skills,
          interests: formData.interests,
          goals: formData.goals,
          values: formData.values,
          lifestyle: formData.lifestyle,
          profileVisibility: formData.profileVisibility,
          emailNotifications: formData.emailNotifications,
          marketingEmails: formData.marketingEmails,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.error || 'Something went wrong');
      }

      // Sign in the user after successful signup
      const result = await signIn('credentials', {
        email: formData.email,
        password: formData.password,
        redirect: false,
      });

      if (result?.error) {
        throw new Error(result.error);
      }

      router.push('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    
    if (type === 'checkbox') {
      const target = e.target as HTMLInputElement;
      setFormData(prev => ({
        ...prev,
        [name]: target.checked
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleMultiSelectChange = (name: string) => (value: string[]) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const nextSection = () => {
    setCurrentSection(prev => Math.min(prev + 1, 3));
  };

  const prevSection = () => {
    setCurrentSection(prev => Math.max(prev - 1, 1));
  };

  return (
    <div className="max-w-2xl mx-auto">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {sections.map((section) => (
            <div
              key={section.id}
              className={`flex-1 text-center ${
                currentSection >= section.id ? 'text-connection-blue' : 'text-gray-400'
              }`}
            >
              <div className="relative">
                <div
                  className={`w-8 h-8 mx-auto rounded-full flex items-center justify-center border-2 ${
                    currentSection >= section.id
                      ? 'border-connection-blue bg-white'
                      : 'border-gray-300 bg-gray-50'
                  }`}
                >
                  {currentSection > section.id ? (
                    <svg className="w-4 h-4 text-connection-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span className={currentSection >= section.id ? 'text-connection-blue' : 'text-gray-400'}>
                      {section.id}
                    </span>
                  )}
                </div>
                <div className="mt-2 text-sm font-medium">{section.title}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="relative pt-1">
          <div className="flex mb-2 items-center justify-between">
            <div className="flex-1">
              <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-100">
                <div
                  style={{ width: `${((currentSection - 1) / (sections.length - 1)) * 100}%` }}
                  className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-connection-blue transition-all duration-500"
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <form className="bg-white rounded-lg shadow-sm border border-gray-100 p-8" onSubmit={handleSubmit}>
        {error && (
          <div className="rounded-md bg-red-50 p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <div className="text-sm text-red-700">{error}</div>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-8">
          {/* Section title and description */}
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {sections[currentSection - 1].title}
            </h2>
            <p className="text-gray-500">
              {sections[currentSection - 1].description}
            </p>
          </div>

          {/* Section 1: Basic Information */}
          <div className={currentSection === 1 ? '' : 'hidden'}>
            <div className="space-y-4">
              <div>
                <label htmlFor="firstName" className="block text-sm font-medium text-gray-700">
                  First Name
                </label>
                <input
                  id="firstName"
                  name="firstName"
                  type="text"
                  required
                  className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-connection-blue focus:border-connection-blue focus:z-10 sm:text-sm"
                  value={formData.firstName}
                  onChange={handleChange}
                />
              </div>

              <div>
                <label htmlFor="lastName" className="block text-sm font-medium text-gray-700">
                  Last Name
                </label>
                <input
                  id="lastName"
                  name="lastName"
                  type="text"
                  required
                  className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-connection-blue focus:border-connection-blue focus:z-10 sm:text-sm"
                  value={formData.lastName}
                  onChange={handleChange}
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                  Email address
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-connection-blue focus:border-connection-blue focus:z-10 sm:text-sm"
                  value={formData.email}
                  onChange={handleChange}
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-connection-blue focus:border-connection-blue focus:z-10 sm:text-sm"
                  value={formData.password}
                  onChange={handleChange}
                />
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                  Confirm Password
                </label>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  autoComplete="new-password"
                  required
                  className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-connection-blue focus:border-connection-blue focus:z-10 sm:text-sm"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                />
              </div>
            </div>
          </div>

          {/* Section 2: Profile & Interests */}
          <div className={currentSection === 2 ? '' : 'hidden'}>
            <div className="space-y-6">
              <div>
                <label htmlFor="bio" className="block text-sm font-medium text-gray-700">
                  Bio
                </label>
                <textarea
                  id="bio"
                  name="bio"
                  rows={3}
                  className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-connection-blue focus:border-connection-blue focus:z-10 sm:text-sm"
                  placeholder="Tell us a bit about yourself..."
                  value={formData.bio}
                  onChange={handleChange}
                />
              </div>

              <MultiSelect
                label="Skills & Expertise"
                options={skillOptions}
                value={formData.skills}
                onChange={handleMultiSelectChange('skills')}
                placeholder="Select or type your skills..."
                maxItems={10}
              />

              <MultiSelect
                label="Interests"
                options={interestOptions}
                value={formData.interests}
                onChange={handleMultiSelectChange('interests')}
                placeholder="What are you interested in?"
                maxItems={10}
              />

              <MultiSelect
                label="Lifestyle"
                options={lifestyleOptions}
                value={formData.lifestyle}
                onChange={handleMultiSelectChange('lifestyle')}
                placeholder="What's your lifestyle like?"
                maxItems={5}
              />
            </div>
          </div>

          {/* Section 3: Goals & Values */}
          <div className={currentSection === 3 ? '' : 'hidden'}>
            <div className="space-y-6">
              <MultiSelect
                label="Goals"
                options={goalOptions}
                value={formData.goals}
                onChange={handleMultiSelectChange('goals')}
                placeholder="What are your goals?"
                maxItems={5}
              />

              <MultiSelect
                label="Values"
                options={valueOptions}
                value={formData.values}
                onChange={handleMultiSelectChange('values')}
                placeholder="What values are important to you?"
                maxItems={5}
              />

              <div>
                <label htmlFor="profileVisibility" className="block text-sm font-medium text-gray-700">
                  Profile Visibility
                </label>
                <select
                  id="profileVisibility"
                  name="profileVisibility"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-connection-blue focus:border-connection-blue sm:text-sm rounded-md"
                  value={formData.profileVisibility}
                  onChange={handleChange}
                >
                  <option value="public">Public - Anyone can view your profile</option>
                  <option value="connections">Connections Only - Only your connections can view your profile</option>
                  <option value="private">Private - Only you can view your profile</option>
                </select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center">
                  <input
                    id="emailNotifications"
                    name="emailNotifications"
                    type="checkbox"
                    className="h-4 w-4 text-connection-blue focus:ring-connection-blue border-gray-300 rounded"
                    checked={formData.emailNotifications}
                    onChange={handleChange}
                  />
                  <label htmlFor="emailNotifications" className="ml-2 block text-sm text-gray-700">
                    Receive email notifications about new connections and messages
                  </label>
                </div>

                <div className="flex items-center">
                  <input
                    id="marketingEmails"
                    name="marketingEmails"
                    type="checkbox"
                    className="h-4 w-4 text-connection-blue focus:ring-connection-blue border-gray-300 rounded"
                    checked={formData.marketingEmails}
                    onChange={handleChange}
                  />
                  <label htmlFor="marketingEmails" className="ml-2 block text-sm text-gray-700">
                    Receive occasional updates about new features and events
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Buttons */}
          <div className="flex justify-between pt-6 border-t border-gray-100 mt-8">
            {currentSection > 1 && (
              <button
                type="button"
                onClick={prevSection}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
              >
                <svg className="-ml-1 mr-2 h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M7.707 14.707a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l2.293 2.293a1 1 0 010 1.414z" clipRule="evenodd" />
                </svg>
                Previous
              </button>
            )}
            {currentSection < 3 ? (
              <button
                type="button"
                onClick={nextSection}
                className="ml-auto inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-connection-blue hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
              >
                Next
                <svg className="-mr-1 ml-2 h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M12.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            ) : (
              <button
                type="submit"
                disabled={loading}
                className="ml-auto inline-flex items-center px-6 py-3 border border-transparent rounded-md shadow-sm text-base font-medium text-white bg-connection-blue hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Creating account...
                  </>
                ) : (
                  'Create account'
                )}
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
} 