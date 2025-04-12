'use client';

import { useFormContext } from 'react-hook-form';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { SignupFormData } from '@/app/types/signup';

export function SocialLinksStep() {
  const {
    register,
    formState: { errors },
  } = useFormContext<SignupFormData>();

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold mb-4 text-gray-900">Your Social Links</h2>
        <p className="text-gray-600 mb-6">
          Add your professional profiles to help others connect with you.
        </p>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="linkedin">LinkedIn Profile</Label>
          <Input
            id="linkedin"
            type="url"
            placeholder="https://linkedin.com/in/your-profile"
            {...register('linkedin', {
              pattern: {
                value: /^https?:\/\/(www\.)?linkedin\.com\/in\/[\w-]+\/?$/,
                message: 'Please enter a valid LinkedIn profile URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.linkedin ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.linkedin && (
            <p className="text-sm text-red-500">{errors.linkedin.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="twitter">Twitter Profile</Label>
          <Input
            id="twitter"
            type="url"
            placeholder="https://twitter.com/your-handle"
            {...register('twitter', {
              pattern: {
                value: /^https?:\/\/(www\.)?twitter\.com\/[\w-]+\/?$/,
                message: 'Please enter a valid Twitter profile URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.twitter ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.twitter && (
            <p className="text-sm text-red-500">{errors.twitter.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="github">GitHub Profile</Label>
          <Input
            id="github"
            type="url"
            placeholder="https://github.com/your-username"
            {...register('github', {
              pattern: {
                value: /^https?:\/\/(www\.)?github\.com\/[\w-]+\/?$/,
                message: 'Please enter a valid GitHub profile URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.github ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.github && (
            <p className="text-sm text-red-500">{errors.github.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="portfolio">Portfolio Website</Label>
          <Input
            id="portfolio"
            type="url"
            placeholder="https://your-website.com"
            {...register('portfolio', {
              pattern: {
                value: /^https?:\/\/[\w-]+(\.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?$/,
                message: 'Please enter a valid website URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.portfolio ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.portfolio && (
            <p className="text-sm text-red-500">{errors.portfolio.message}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default SocialLinksStep; 