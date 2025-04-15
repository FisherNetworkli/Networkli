'use client';

import { useEffect } from 'react';
import { Label } from '../ui/label';
import { InteractiveSelection } from '../ui/interactive-selection';
import { useFormContext } from 'react-hook-form';
import { SignupFormData } from '@/app/types/signup';
import { VALUE_OPTIONS, GOAL_OPTIONS, INTEREST_OPTIONS, NETWORKING_PREFERENCES } from '@/lib/profile-options';

export function PreferencesStep() {
  const {
    register,
    setValue,
    watch,
    formState: { errors },
  } = useFormContext<SignupFormData>();

  useEffect(() => {
    register('values', { required: 'Please select your core values' });
    register('goals', { required: 'Please select your goals' });
    register('interests', { required: 'Please select your interests' });
    register('networkingStyle', { required: 'Please select your networking preferences' });
  }, [register]);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label>What are your core values?</Label>
        <InteractiveSelection
          options={VALUE_OPTIONS}
          value={watch('values')}
          onChange={(value) => setValue('values', value)}
          maxSelected={3}
          title="Select your core values"
          description="Choose up to 3 values that drive you"
        />
        {errors.values && (
          <p className="text-sm text-red-500">{errors.values.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label>What are your current goals?</Label>
        <InteractiveSelection
          options={GOAL_OPTIONS}
          value={watch('goals')}
          onChange={(value) => setValue('goals', value)}
          maxSelected={3}
          title="Select your goals"
          description="Choose up to 3 goals you're working towards"
        />
        {errors.goals && (
          <p className="text-sm text-red-500">{errors.goals.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label>What topics interest you most?</Label>
        <InteractiveSelection
          options={INTEREST_OPTIONS}
          value={watch('interests')}
          onChange={(value) => setValue('interests', value)}
          maxSelected={3}
          title="Select your interests"
          description="Choose up to 3 topics you're passionate about"
        />
        {errors.interests && (
          <p className="text-sm text-red-500">{errors.interests.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label>How do you prefer to connect?</Label>
        <InteractiveSelection
          options={NETWORKING_PREFERENCES}
          value={watch('networkingStyle')}
          onChange={(value) => setValue('networkingStyle', value)}
          maxSelected={3}
          title="Select your networking style"
          description="Choose up to 3 ways you prefer to connect"
        />
        {errors.networkingStyle && (
          <p className="text-sm text-red-500">{errors.networkingStyle.message}</p>
        )}
      </div>
    </div>
  );
} 