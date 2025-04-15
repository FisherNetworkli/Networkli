'use client';

import { useFormContext } from 'react-hook-form';
import { Label } from '@/components/ui/label';
import { InteractiveSelection } from '@/components/ui/interactive-selection';
import { motion } from 'framer-motion';

interface FormData {
  skills: string[];
  interests: string[];
  professionalGoals: string[];
  values: string[];
}

const skillOptions = [
  { label: 'Web Development', value: 'web-dev', category: 'Development' },
  { label: 'Mobile Development', value: 'mobile-dev', category: 'Development' },
  { label: 'UI/UX Design', value: 'ui-ux', category: 'Design' },
  { label: 'Data Science', value: 'data-science', category: 'Data' },
  { label: 'Machine Learning', value: 'ml', category: 'AI' },
  { label: 'Cloud Computing', value: 'cloud', category: 'Infrastructure' },
  { label: 'DevOps', value: 'devops', category: 'Infrastructure' },
  { label: 'Cybersecurity', value: 'security', category: 'Security' },
];

const interestOptions = [
  { label: 'Artificial Intelligence', value: 'ai', category: 'Technology' },
  { label: 'Blockchain', value: 'blockchain', category: 'Technology' },
  { label: 'Internet of Things', value: 'iot', category: 'Technology' },
  { label: 'Augmented Reality', value: 'ar', category: 'Technology' },
  { label: 'Virtual Reality', value: 'vr', category: 'Technology' },
  { label: 'Quantum Computing', value: 'quantum', category: 'Technology' },
  { label: 'Robotics', value: 'robotics', category: 'Technology' },
  { label: 'Sustainability', value: 'sustainability', category: 'Social Impact' },
];

const goalOptions = [
  { label: 'Start a Tech Company', value: 'startup', category: 'Entrepreneurship' },
  { label: 'Become a Tech Lead', value: 'tech-lead', category: 'Career Growth' },
  { label: 'Contribute to Open Source', value: 'open-source', category: 'Community' },
  { label: 'Build Personal Brand', value: 'personal-brand', category: 'Personal Development' },
  { label: 'Learn New Technologies', value: 'learn', category: 'Education' },
  { label: 'Network with Peers', value: 'network', category: 'Networking' },
];

const valueOptions = [
  { label: 'Innovation', value: 'innovation', category: 'Core Values' },
  { label: 'Collaboration', value: 'collaboration', category: 'Core Values' },
  { label: 'Sustainability', value: 'sustainability', category: 'Core Values' },
  { label: 'Diversity', value: 'diversity', category: 'Core Values' },
  { label: 'Excellence', value: 'excellence', category: 'Core Values' },
  { label: 'Integrity', value: 'integrity', category: 'Core Values' },
];

export function PreferencesStep() {
  const {
    register,
    formState: { errors },
    setValue,
    watch,
  } = useFormContext<FormData>();

  // Helper function to group options by category
  const groupByCategory = (option: { category?: string }) => {
    return option.category || 'Other';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8"
    >
      <div className="space-y-4">
        <Label className="text-sm font-medium text-gray-700">What are your technical skills?</Label>
        <InteractiveSelection
          id="skills"
          options={skillOptions}
          value={watch('skills') || []}
          onChange={(value) => setValue('skills', value)}
          maxSelected={5}
          placeholder="Select up to 5 skills..."
          className="bg-white rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-all duration-200"
          groupBy={groupByCategory}
        />
        {errors.skills && (
          <p className="text-sm text-red-500 mt-1">{errors.skills.message}</p>
        )}
      </div>

      <div className="space-y-4">
        <Label className="text-sm font-medium text-gray-700">What interests you?</Label>
        <InteractiveSelection
          id="interests"
          options={interestOptions}
          value={watch('interests') || []}
          onChange={(value) => setValue('interests', value)}
          maxSelected={4}
          placeholder="Select up to 4 interests..."
          className="bg-white rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-all duration-200"
          groupBy={groupByCategory}
        />
        {errors.interests && (
          <p className="text-sm text-red-500 mt-1">{errors.interests.message}</p>
        )}
      </div>

      <div className="space-y-4">
        <Label className="text-sm font-medium text-gray-700">What are your professional goals?</Label>
        <InteractiveSelection
          id="professionalGoals"
          options={goalOptions}
          value={watch('professionalGoals') || []}
          onChange={(value) => setValue('professionalGoals', value)}
          maxSelected={3}
          placeholder="Select up to 3 goals..."
          className="bg-white rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-all duration-200"
          groupBy={groupByCategory}
        />
        {errors.professionalGoals && (
          <p className="text-sm text-red-500 mt-1">{errors.professionalGoals.message}</p>
        )}
      </div>

      <div className="space-y-4">
        <Label className="text-sm font-medium text-gray-700">What are your core values?</Label>
        <InteractiveSelection
          id="values"
          options={valueOptions}
          value={watch('values') || []}
          onChange={(value) => setValue('values', value)}
          maxSelected={3}
          placeholder="Select up to 3 values..."
          className="bg-white rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-all duration-200"
          groupBy={groupByCategory}
        />
        {errors.values && (
          <p className="text-sm text-red-500 mt-1">{errors.values.message}</p>
        )}
      </div>
    </motion.div>
  );
} 