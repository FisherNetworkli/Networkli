'use client';

import { useFormContext } from 'react-hook-form';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { FormData } from '@/types/auth';

const industries = [
  'Technology',
  'Finance',
  'Healthcare',
  'Education',
  'Manufacturing',
  'Retail',
  'Media',
  'Other',
];

const experienceLevels = [
  'Entry-level',
  'Mid-level',
  'Senior',
  'Lead',
  'Executive',
];

const jobTitles = [
  'Software Engineer',
  'Product Manager',
  'Data Scientist',
  'UX Designer',
  'Marketing Manager',
  'Sales Representative',
  'Project Manager',
  'Business Analyst',
  'Content Writer',
  'Customer Support',
  'HR Manager',
  'Financial Analyst',
  'Operations Manager',
  'Research Scientist',
  'Teacher',
  'Doctor',
  'Lawyer',
  'Architect',
  'Other',
];

const educationLevels = [
  'High School',
  'Associate Degree',
  'Bachelor\'s Degree',
  'Master\'s Degree',
  'Doctorate',
  'Professional Certification',
  'Self-taught',
  'Other',
];

export function ProfessionalInfoStep() {
  const {
    register,
    formState: { errors },
  } = useFormContext<FormData>();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Professional Information</h2>
        <p className="text-gray-600 mb-6">
          Tell us about your professional background and experience.
        </p>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="title">Job Title</Label>
          <select
            id="title"
            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-networkly-orange focus-visible:ring-offset-2"
            {...register('title', {
              required: 'Job title is required',
            })}
          >
            <option value="">Select your job title</option>
            {jobTitles.map((title) => (
              <option key={title} value={title}>
                {title}
              </option>
            ))}
          </select>
          {errors.title && (
            <p className="text-sm text-red-500">{errors.title.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="company">Company</Label>
          <Input
            id="company"
            placeholder="Enter your company name"
            className="bg-white text-gray-900"
            {...register('company', {
              required: 'Company is required',
            })}
          />
          {errors.company && (
            <p className="text-sm text-red-500">{errors.company.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="industry">Industry</Label>
          <select
            id="industry"
            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-networkly-orange focus-visible:ring-offset-2"
            {...register('industry', {
              required: 'Industry is required',
            })}
          >
            <option value="">Select an industry</option>
            {industries.map((industry) => (
              <option key={industry} value={industry}>
                {industry}
              </option>
            ))}
          </select>
          {errors.industry && (
            <p className="text-sm text-red-500">{errors.industry.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="experience">Experience Level</Label>
          <select
            id="experience"
            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-networkly-orange focus-visible:ring-offset-2"
            {...register('experience', {
              required: 'Experience level is required',
            })}
          >
            <option value="">Select your experience level</option>
            {experienceLevels.map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
          {errors.experience && (
            <p className="text-sm text-red-500">{errors.experience.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="education">Education</Label>
          <select
            id="education"
            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-networkly-orange focus-visible:ring-offset-2"
            {...register('education', {
              required: 'Education is required',
            })}
          >
            <option value="">Select your education level</option>
            {educationLevels.map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
          {errors.education && (
            <p className="text-sm text-red-500">{errors.education.message}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ProfessionalInfoStep; 