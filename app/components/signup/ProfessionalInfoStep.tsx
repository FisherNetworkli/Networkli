'use client';

import { useFormContext } from 'react-hook-form';
import { SignupFormData } from '../../types/signup';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { InteractiveSelection } from '../../components/ui/interactive-selection';

const skills = [
  { value: 'javascript', label: 'JavaScript' },
  { value: 'typescript', label: 'TypeScript' },
  { value: 'react', label: 'React' },
  { value: 'nextjs', label: 'Next.js' },
  { value: 'nodejs', label: 'Node.js' },
  { value: 'python', label: 'Python' },
  { value: 'java', label: 'Java' },
  { value: 'cpp', label: 'C++' },
  { value: 'sql', label: 'SQL' },
  { value: 'aws', label: 'AWS' },
  { value: 'docker', label: 'Docker' },
  { value: 'kubernetes', label: 'Kubernetes' },
  { value: 'git', label: 'Git' },
  { value: 'cicd', label: 'CI/CD' },
  { value: 'testing', label: 'Testing' },
  { value: 'uiux', label: 'UI/UX' },
  { value: 'graphql', label: 'GraphQL' },
  { value: 'rest', label: 'REST APIs' },
  { value: 'ml', label: 'Machine Learning' },
  { value: 'datascience', label: 'Data Science' }
];

const industries = [
  { value: 'technology', label: 'Technology' },
  { value: 'finance', label: 'Finance' },
  { value: 'healthcare', label: 'Healthcare' },
  { value: 'education', label: 'Education' },
  { value: 'retail', label: 'Retail' },
  { value: 'manufacturing', label: 'Manufacturing' },
  { value: 'media', label: 'Media' },
  { value: 'realestate', label: 'Real Estate' },
  { value: 'transportation', label: 'Transportation' },
  { value: 'energy', label: 'Energy' },
  { value: 'agriculture', label: 'Agriculture' },
  { value: 'construction', label: 'Construction' },
  { value: 'hospitality', label: 'Hospitality' },
  { value: 'telecom', label: 'Telecommunications' },
  { value: 'consulting', label: 'Consulting' }
];

const experienceLevels = [
  { value: 'entry', label: 'Entry Level' },
  { value: 'junior', label: 'Junior' },
  { value: 'mid', label: 'Mid-Level' },
  { value: 'senior', label: 'Senior' },
  { value: 'lead', label: 'Lead' },
  { value: 'manager', label: 'Manager' },
  { value: 'director', label: 'Director' },
  { value: 'vp', label: 'VP' },
  { value: 'clevel', label: 'C-Level' },
  { value: 'founder', label: 'Founder' }
];

export function ProfessionalInfoStep() {
  const { register, formState: { errors }, setValue, watch } = useFormContext<SignupFormData>();
  const selectedSkills = watch('skills') || [];

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="title">Job Title</Label>
        <Input
          id="title"
          {...register('title', { required: 'Job title is required' })}
        />
        {errors.title && (
          <p className="text-sm text-red-500">{errors.title.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="company">Company</Label>
        <Input
          id="company"
          {...register('company', { required: 'Company is required' })}
        />
        {errors.company && (
          <p className="text-sm text-red-500">{errors.company.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="industry">Industry</Label>
        <InteractiveSelection
          options={industries}
          selectedValues={watch('industry') ? [watch('industry')] : []}
          onChange={(selected) => setValue('industry', selected[0] || '')}
          maxSelections={1}
        />
        {errors.industry && (
          <p className="text-sm text-red-500">{errors.industry.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="experience">Experience Level</Label>
        <InteractiveSelection
          options={experienceLevels}
          selectedValues={watch('experience') ? [watch('experience')] : []}
          onChange={(selected) => setValue('experience', selected[0] || '')}
          maxSelections={1}
        />
        {errors.experience && (
          <p className="text-sm text-red-500">{errors.experience.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label>Skills</Label>
        <InteractiveSelection
          options={skills}
          selectedValues={selectedSkills}
          onChange={(selected) => setValue('skills', selected)}
          maxSelections={5}
        />
        {errors.skills && (
          <p className="text-sm text-red-500">{errors.skills.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="bio">Professional Bio</Label>
        <textarea
          id="bio"
          className="w-full min-h-[100px] px-3 py-2 border rounded-md"
          {...register('bio', { 
            required: 'Bio is required',
            minLength: {
              value: 50,
              message: 'Bio must be at least 50 characters'
            }
          })}
        />
        {errors.bio && (
          <p className="text-sm text-red-500">{errors.bio.message}</p>
        )}
      </div>
    </div>
  );
}

export default ProfessionalInfoStep; 