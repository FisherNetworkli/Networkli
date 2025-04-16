'use client';

import { useFormContext } from 'react-hook-form';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { FormData } from '@/types/auth';

export function SocialLinksStep() {
  const {
    register,
    formState: { errors },
  } = useFormContext<FormData>();

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
            {...register('linkedinUrl', {
              pattern: {
                value: /^https?:\/\/(www\.)?linkedin\.com\/in\/[\w-]+\/?$/,
                message: 'Please enter a valid LinkedIn profile URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.linkedinUrl ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.linkedinUrl && (
            <p className="text-sm text-red-500">{errors.linkedinUrl.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="twitter">Twitter Profile</Label>
          <Input
            id="twitter"
            type="url"
            placeholder="https://twitter.com/your-handle"
            {...register('twitterUrl', {
              pattern: {
                value: /^https?:\/\/(www\.)?twitter\.com\/[\w-]+\/?$/,
                message: 'Please enter a valid Twitter profile URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.twitterUrl ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.twitterUrl && (
            <p className="text-sm text-red-500">{errors.twitterUrl.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="github">GitHub Profile</Label>
          <Input
            id="github"
            type="url"
            placeholder="https://github.com/your-username"
            {...register('githubUrl', {
              pattern: {
                value: /^https?:\/\/(www\.)?github\.com\/[\w-]+\/?$/,
                message: 'Please enter a valid GitHub profile URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.githubUrl ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.githubUrl && (
            <p className="text-sm text-red-500">{errors.githubUrl.message}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="website">Personal Website</Label>
          <Input
            id="website"
            type="url"
            placeholder="https://your-website.com"
            {...register('websiteUrl', {
              pattern: {
                value: /^https?:\/\/([\w-]+\.)+[\w-]+(\/[\w-./?%&=]*)?$/,
                message: 'Please enter a valid website URL',
              },
            })}
            className={`bg-white text-gray-900 ${errors.websiteUrl ? 'border-red-500' : 'border-gray-300'}`}
          />
          {errors.websiteUrl && (
            <p className="text-sm text-red-500">{errors.websiteUrl.message}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default SocialLinksStep; 