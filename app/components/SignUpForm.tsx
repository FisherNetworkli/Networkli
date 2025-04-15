'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { signIn } from 'next-auth/react';
import {
  skillCategories,
  interestCategories,
  valueCategories,
  industryCategories,
  experienceLevels,
  educationLevels
} from '../data/userProfileData';

interface FormData {
  // Basic Info
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  
  // Professional Info
  industry: string;
  experienceLevel: string;
  educationLevel: string;
  currentRole: string;
  company: string;
  
  // Skills & Interests
  technicalSkills: string[];
  softSkills: string[];
  businessSkills: string[];
  professionalInterests: string[];
  personalInterests: string[];
  learningInterests: string[];
  
  // Values
  workValues: string[];
  personalValues: string[];
  socialValues: string[];
}

const initialFormData: FormData = {
  firstName: '',
  lastName: '',
  email: '',
  password: '',
  industry: '',
  experienceLevel: '',
  educationLevel: '',
  currentRole: '',
  company: '',
  technicalSkills: [],
  softSkills: [],
  businessSkills: [],
  professionalInterests: [],
  personalInterests: [],
  learningInterests: [],
  workValues: [],
  personalValues: [],
  socialValues: []
};

export default function SignUpForm() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [errors, setErrors] = useState<Partial<FormData>>({});
  const [loading, setLoading] = useState(false);

  const totalSteps = 4;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    // Clear error when user starts typing
    if (errors[name as keyof FormData]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const handleMultiSelect = (category: keyof FormData, value: string) => {
    setFormData(prev => {
      const currentValues = prev[category] as string[];
      const newValues = currentValues.includes(value)
        ? currentValues.filter(v => v !== value)
        : [...currentValues, value];
      return { ...prev, [category]: newValues };
    });
  };

  const validateStep = (step: number): boolean => {
    const newErrors: Partial<FormData> = {};

    switch (step) {
      case 1:
        if (!formData.firstName) newErrors.firstName = 'First name is required';
        if (!formData.lastName) newErrors.lastName = 'Last name is required';
        if (!formData.email) newErrors.email = 'Email is required';
        if (!formData.password) newErrors.password = 'Password is required';
        break;
      case 2:
        if (!formData.industry) newErrors.industry = 'Industry is required';
        if (!formData.experienceLevel) newErrors.experienceLevel = 'Experience level is required';
        if (!formData.educationLevel) newErrors.educationLevel = 'Education level is required';
        break;
      case 3:
        if (formData.technicalSkills.length === 0 && 
            formData.softSkills.length === 0 && 
            formData.businessSkills.length === 0) {
          newErrors.technicalSkills = 'Select at least one skill';
        }
        break;
      case 4:
        if (formData.workValues.length === 0 && 
            formData.personalValues.length === 0 && 
            formData.socialValues.length === 0) {
          newErrors.workValues = 'Select at least one value';
        }
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, totalSteps));
    }
  };

  const handleBack = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (validateStep(currentStep)) {
      setLoading(true);
      try {
        // Register user
        const res = await fetch('/api/auth/register', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || 'Something went wrong');
        }

        // Sign in the user after successful registration
        const result = await signIn('credentials', {
          email: formData.email,
          password: formData.password,
          redirect: false,
        });

        if (result?.error) {
          throw new Error(result.error);
        }

        // Redirect to dashboard on success
        router.push('/dashboard');
      } catch (err) {
        console.error('Error submitting form:', err);
        setErrors({
          email: err instanceof Error ? err.message : 'An error occurred',
        });
      } finally {
        setLoading(false);
      }
    }
  };

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold mb-6">Basic Information</h2>
            <div>
              <label className="block text-sm font-medium text-gray-700">First Name</label>
              <input
                type="text"
                name="firstName"
                value={formData.firstName}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
              {errors.firstName && <p className="mt-1 text-sm text-red-600">{errors.firstName}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Last Name</label>
              <input
                type="text"
                name="lastName"
                value={formData.lastName}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
              {errors.lastName && <p className="mt-1 text-sm text-red-600">{errors.lastName}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Email</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
              {errors.email && <p className="mt-1 text-sm text-red-600">{errors.email}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
              {errors.password && <p className="mt-1 text-sm text-red-600">{errors.password}</p>}
            </div>
          </div>
        );
      case 2:
        return (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold mb-6">Professional Information</h2>
            <div>
              <label className="block text-sm font-medium text-gray-700">Industry</label>
              <select
                name="industry"
                value={formData.industry}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">Select Industry</option>
                {industryCategories.map(industry => (
                  <option key={industry} value={industry}>{industry}</option>
                ))}
              </select>
              {errors.industry && <p className="mt-1 text-sm text-red-600">{errors.industry}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Experience Level</label>
              <select
                name="experienceLevel"
                value={formData.experienceLevel}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">Select Experience Level</option>
                {experienceLevels.map(level => (
                  <option key={level} value={level}>{level}</option>
                ))}
              </select>
              {errors.experienceLevel && <p className="mt-1 text-sm text-red-600">{errors.experienceLevel}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Education Level</label>
              <select
                name="educationLevel"
                value={formData.educationLevel}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">Select Education Level</option>
                {educationLevels.map(level => (
                  <option key={level} value={level}>{level}</option>
                ))}
              </select>
              {errors.educationLevel && <p className="mt-1 text-sm text-red-600">{errors.educationLevel}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Current Role</label>
              <input
                type="text"
                name="currentRole"
                value={formData.currentRole}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Company</label>
              <input
                type="text"
                name="company"
                value={formData.company}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
          </div>
        );
      case 3:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold mb-6">Skills & Interests</h2>
            
            <div>
              <h3 className="text-lg font-medium mb-3">Technical Skills</h3>
              <div className="flex flex-wrap gap-2">
                {skillCategories.technical.map(skill => (
                  <button
                    key={skill}
                    type="button"
                    onClick={() => handleMultiSelect('technicalSkills', skill)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.technicalSkills.includes(skill)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {skill}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Soft Skills</h3>
              <div className="flex flex-wrap gap-2">
                {skillCategories.soft.map(skill => (
                  <button
                    key={skill}
                    type="button"
                    onClick={() => handleMultiSelect('softSkills', skill)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.softSkills.includes(skill)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {skill}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Business Skills</h3>
              <div className="flex flex-wrap gap-2">
                {skillCategories.business.map(skill => (
                  <button
                    key={skill}
                    type="button"
                    onClick={() => handleMultiSelect('businessSkills', skill)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.businessSkills.includes(skill)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {skill}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Professional Interests</h3>
              <div className="flex flex-wrap gap-2">
                {interestCategories.professional.map(interest => (
                  <button
                    key={interest}
                    type="button"
                    onClick={() => handleMultiSelect('professionalInterests', interest)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.professionalInterests.includes(interest)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {interest}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Personal Interests</h3>
              <div className="flex flex-wrap gap-2">
                {interestCategories.personal.map(interest => (
                  <button
                    key={interest}
                    type="button"
                    onClick={() => handleMultiSelect('personalInterests', interest)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.personalInterests.includes(interest)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {interest}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Learning Interests</h3>
              <div className="flex flex-wrap gap-2">
                {interestCategories.learning.map(interest => (
                  <button
                    key={interest}
                    type="button"
                    onClick={() => handleMultiSelect('learningInterests', interest)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.learningInterests.includes(interest)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {interest}
                  </button>
                ))}
              </div>
            </div>
          </div>
        );
      case 4:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold mb-6">Values</h2>
            
            <div>
              <h3 className="text-lg font-medium mb-3">Work Values</h3>
              <div className="flex flex-wrap gap-2">
                {valueCategories.work.map(value => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => handleMultiSelect('workValues', value)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.workValues.includes(value)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Personal Values</h3>
              <div className="flex flex-wrap gap-2">
                {valueCategories.personal.map(value => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => handleMultiSelect('personalValues', value)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.personalValues.includes(value)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-3">Social Values</h3>
              <div className="flex flex-wrap gap-2">
                {valueCategories.social.map(value => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => handleMultiSelect('socialValues', value)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      formData.socialValues.includes(value)
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {Array.from({ length: totalSteps }, (_, i) => i + 1).map(step => (
            <div
              key={step}
              className={`h-2 flex-1 mx-2 rounded ${
                step <= currentStep ? 'bg-blue-500' : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
        <div className="text-center text-sm text-gray-600">
          Step {currentStep} of {totalSteps}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {renderStep()}

        <div className="flex justify-between pt-6">
          {currentStep > 1 && (
            <button
              type="button"
              onClick={handleBack}
              className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
            >
              Back
            </button>
          )}
          {currentStep < totalSteps ? (
            <button
              type="button"
              onClick={handleNext}
              className="ml-auto px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              Next
            </button>
          ) : (
            <button
              type="submit"
              disabled={loading}
              className="ml-auto px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              {loading ? 'Creating account...' : 'Complete Sign Up'}
            </button>
          )}
        </div>
      </form>
    </div>
  );
} 