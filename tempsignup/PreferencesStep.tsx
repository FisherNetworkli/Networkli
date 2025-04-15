'use client';

import { useEffect } from 'react';
import { Label } from '@/components/ui/label';
import { InteractiveSelection } from '@/components/ui/interactive-selection';
import { useFormContext } from 'react-hook-form';
import { FormData } from '@/types/auth';
import { INTEREST_OPTIONS, GOAL_OPTIONS, VALUE_OPTIONS, SKILL_OPTIONS } from '@/lib/profile-options';

interface PreferencesStepProps {
  data: {
    interests: string[];
    goals: string[];
    values: string[];
    bio: string;
    preferred_industries: string[];
    preferred_roles: string[];
    preferred_experience_levels: string[];
    preferred_company_sizes: string[];
    preferred_locations: string[];
    preferred_remote_policy: string;
  };
  onChange: (data: Partial<PreferencesStepProps['data']>) => void;
  onValidationChange?: (isValid: boolean) => void;
}

export function PreferencesStep() {
  const {
    register,
    formState: { errors },
    setValue,
    watch,
  } = useFormContext<FormData>();

  const selectedSkills = watch('skills') || [];
  const selectedInterests = watch('interests') || [];
  const selectedGoals = watch('professionalGoals') || [];
  const selectedValues = watch('values') || [];

  const groupBy = (option: { category?: string }) => option.category || 'Other';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4 text-gray-900">Your Skills & Preferences</h2>
        <p className="text-gray-600 mb-6">
          Help us understand your professional background and aspirations better.
        </p>
      </div>

      <div className="space-y-8">
        <div className="space-y-2">
          <Label htmlFor="skills" className="text-gray-900">Skills</Label>
          <InteractiveSelection
            id="skills"
            value={selectedSkills}
            onChange={(value) => setValue('skills', value)}
            options={SKILL_OPTIONS}
            groupBy={groupBy}
            maxSelected={10}
            placeholder="Select your skills..."
            searchPlaceholder="Search skills..."
            title="Select up to 10 skills"
            description="Choose skills that best represent your expertise"
            className="bg-white text-gray-900"
          />
          {errors.skills && (
            <p className="text-sm text-red-500">{errors.skills.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="interests" className="text-gray-900">Interests</Label>
          <InteractiveSelection
            id="interests"
            value={selectedInterests}
            onChange={(value) => setValue('interests', value)}
            options={INTEREST_OPTIONS}
            groupBy={groupBy}
            maxSelected={8}
            placeholder="Select your interests..."
            searchPlaceholder="Search interests..."
            title="Select up to 8 interests"
            description="Choose topics you're passionate about"
            className="bg-white text-gray-900"
          />
          {errors.interests && (
            <p className="text-sm text-red-500">{errors.interests.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="professionalGoals" className="text-gray-900">Professional Goals</Label>
          <InteractiveSelection
            id="professionalGoals"
            value={selectedGoals}
            onChange={(value) => setValue('professionalGoals', value)}
            options={GOAL_OPTIONS}
            groupBy={groupBy}
            maxSelected={5}
            placeholder="Select your professional goals..."
            searchPlaceholder="Search goals..."
            title="Select up to 5 professional goals"
            description="Choose goals that align with your career aspirations"
            className="bg-white text-gray-900"
          />
          {errors.professionalGoals && (
            <p className="text-sm text-red-500">{errors.professionalGoals.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="values" className="text-gray-900">Values</Label>
          <InteractiveSelection
            id="values"
            value={selectedValues}
            onChange={(value) => setValue('values', value)}
            options={VALUE_OPTIONS}
            groupBy={groupBy}
            maxSelected={6}
            placeholder="Select your values..."
            searchPlaceholder="Search values..."
            title="Select up to 6 values"
            description="Choose values that guide your professional decisions"
            className="bg-white text-gray-900"
          />
          {errors.values && (
            <p className="text-sm text-red-500">{errors.values.message}</p>
          )}
        </div>
      </div>
    </div>
  );
} 