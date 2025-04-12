'use client';

import { useEffect } from 'react';
import { Label } from '../ui/label';
import { InteractiveSelection } from '../ui/interactive-selection';
import { useFormContext } from 'react-hook-form';
import { SignupFormData } from '@/app/types/signup';
import { INTEREST_OPTIONS, GOAL_OPTIONS, VALUE_OPTIONS, SKILL_OPTIONS } from '@/lib/profile-options';

export function PreferencesStep() {
  const {
    register,
    setValue,
    watch,
    formState: { errors },
  } = useFormContext<SignupFormData>();

  const selectedInterests = watch('interests') || [];
  const selectedGoals = watch('lookingFor') || [];
  const selectedIndustries = watch('preferredIndustries') || [];
  const selectedRoles = watch('preferredRoles') || [];

  useEffect(() => {
    register('interests', { required: 'Please select at least one interest' });
    register('lookingFor', { required: 'Please select at least one goal' });
    register('preferredIndustries', { required: 'Please select at least one industry' });
    register('preferredRoles', { required: 'Please select at least one role' });
  }, [register]);

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <Label>What are your professional interests?</Label>
        <InteractiveSelection
          options={INTEREST_OPTIONS}
          selectedValues={selectedInterests}
          onChange={(value) => setValue('interests', value)}
          maxSelections={5}
        />
        {errors.interests && (
          <p className="text-sm text-red-500">{errors.interests.message}</p>
        )}
      </div>

      <div className="space-y-4">
        <Label>What are you looking for?</Label>
        <InteractiveSelection
          options={GOAL_OPTIONS}
          selectedValues={selectedGoals}
          onChange={(value) => setValue('lookingFor', value)}
          maxSelections={3}
        />
        {errors.lookingFor && (
          <p className="text-sm text-red-500">{errors.lookingFor.message}</p>
        )}
      </div>

      <div className="space-y-4">
        <Label>Which industries interest you most?</Label>
        <InteractiveSelection
          options={VALUE_OPTIONS}
          selectedValues={selectedIndustries}
          onChange={(value) => setValue('preferredIndustries', value)}
          maxSelections={3}
        />
        {errors.preferredIndustries && (
          <p className="text-sm text-red-500">{errors.preferredIndustries.message}</p>
        )}
      </div>

      <div className="space-y-4">
        <Label>What roles are you interested in?</Label>
        <InteractiveSelection
          options={SKILL_OPTIONS}
          selectedValues={selectedRoles}
          onChange={(value) => setValue('preferredRoles', value)}
          maxSelections={3}
        />
        {errors.preferredRoles && (
          <p className="text-sm text-red-500">{errors.preferredRoles.message}</p>
        )}
      </div>
    </div>
  );
} 