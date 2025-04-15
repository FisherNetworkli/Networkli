'use client';

import { useFormContext } from 'react-hook-form';
import { SignupFormData } from '../../types/signup';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { Textarea } from '../../components/ui/textarea';
import { InteractiveSelection } from '../../components/ui/interactive-selection';

const skills = [
  // Business & Management
  { value: 'leadership', label: 'Leadership' },
  { value: 'project_management', label: 'Project Management' },
  { value: 'strategic_planning', label: 'Strategic Planning' },
  { value: 'team_building', label: 'Team Building' },
  { value: 'negotiation', label: 'Negotiation' },
  
  // Creative & Communication
  { value: 'public_speaking', label: 'Public Speaking' },
  { value: 'writing', label: 'Writing' },
  { value: 'content_creation', label: 'Content Creation' },
  { value: 'design_thinking', label: 'Design Thinking' },
  { value: 'brand_development', label: 'Brand Development' },
  
  // Technical & Analytical
  { value: 'data_analysis', label: 'Data Analysis' },
  { value: 'research', label: 'Research' },
  { value: 'problem_solving', label: 'Problem Solving' },
  { value: 'digital_literacy', label: 'Digital Literacy' },
  { value: 'technical_writing', label: 'Technical Writing' },
  
  // People & Service
  { value: 'customer_service', label: 'Customer Service' },
  { value: 'mentoring', label: 'Mentoring' },
  { value: 'conflict_resolution', label: 'Conflict Resolution' },
  { value: 'cultural_awareness', label: 'Cultural Awareness' },
  { value: 'relationship_building', label: 'Relationship Building' }
];

const industries = [
  // Traditional Industries
  { value: 'healthcare', label: 'Healthcare & Wellness' },
  { value: 'education', label: 'Education & Training' },
  { value: 'finance', label: 'Finance & Banking' },
  { value: 'legal', label: 'Legal & Law' },
  { value: 'government', label: 'Government & Public Service' },
  
  // Creative Industries
  { value: 'arts', label: 'Arts & Entertainment' },
  { value: 'media', label: 'Media & Communications' },
  { value: 'design', label: 'Design & Architecture' },
  { value: 'fashion', label: 'Fashion & Beauty' },
  { value: 'hospitality', label: 'Hospitality & Tourism' },
  
  // Business & Technology
  { value: 'technology', label: 'Technology & Digital' },
  { value: 'consulting', label: 'Consulting & Professional Services' },
  { value: 'retail', label: 'Retail & E-commerce' },
  { value: 'marketing', label: 'Marketing & Advertising' },
  { value: 'real_estate', label: 'Real Estate & Property' },
  
  // Industrial & Physical
  { value: 'manufacturing', label: 'Manufacturing & Production' },
  { value: 'construction', label: 'Construction & Engineering' },
  { value: 'transportation', label: 'Transportation & Logistics' },
  { value: 'energy', label: 'Energy & Utilities' },
  { value: 'agriculture', label: 'Agriculture & Environment' },
  
  // Emerging & Social
  { value: 'nonprofit', label: 'Nonprofit & Social Impact' },
  { value: 'sports', label: 'Sports & Recreation' },
  { value: 'science', label: 'Science & Research' },
  { value: 'sustainability', label: 'Sustainability & Green Tech' },
  { value: 'social_services', label: 'Social Services & Community' }
];

const professionalInterests = [
  // Growth & Development
  { value: 'mentorship', label: 'Mentoring Others' },
  { value: 'being_mentored', label: 'Being Mentored' },
  { value: 'skill_development', label: 'Learning New Skills' },
  { value: 'career_transition', label: 'Career Transition' },
  { value: 'leadership_development', label: 'Leadership Development' },

  // Innovation & Creation
  { value: 'startup', label: 'Starting a Business' },
  { value: 'innovation', label: 'Innovation & New Ideas' },
  { value: 'product_development', label: 'Product Development' },
  { value: 'creative_projects', label: 'Creative Projects' },
  { value: 'research_development', label: 'Research & Development' },

  // Impact & Purpose
  { value: 'social_impact', label: 'Social Impact' },
  { value: 'sustainability', label: 'Environmental Sustainability' },
  { value: 'dei', label: 'Diversity & Inclusion' },
  { value: 'community_building', label: 'Community Building' },
  { value: 'education_outreach', label: 'Educational Outreach' },

  // Business & Strategy
  { value: 'business_strategy', label: 'Business Strategy' },
  { value: 'market_expansion', label: 'Market Expansion' },
  { value: 'digital_transformation', label: 'Digital Transformation' },
  { value: 'international_business', label: 'International Business' },
  { value: 'organizational_change', label: 'Organizational Change' }
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
  const { register, formState: { errors }, watch, setValue } = useFormContext<SignupFormData>();

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="title">Professional Title</Label>
        <Input
          id="title"
          {...register('title', { required: 'Professional title is required' })}
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
        <Label htmlFor="bio">Bio (100-250 characters)</Label>
        <Textarea
          id="bio"
          {...register('bio', { 
            required: 'Bio is required',
            minLength: {
              value: 100,
              message: 'Bio must be at least 100 characters'
            },
            maxLength: {
              value: 250,
              message: 'Bio must not exceed 250 characters'
            }
          })}
          className="h-24"
        />
        {errors.bio && (
          <p className="text-sm text-red-500">{errors.bio.message}</p>
        )}
        <p className="text-sm text-gray-500">
          {watch('bio')?.length || 0}/250 characters
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="expertise">What are you good at?</Label>
        <Textarea
          id="expertise"
          {...register('expertise', { 
            required: 'Please tell us what you\'re good at'
          })}
          className="h-24"
          placeholder="Share your key strengths and areas of expertise..."
        />
        {errors.expertise && (
          <p className="text-sm text-red-500">{errors.expertise.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="needs">What do you need help with?</Label>
        <Textarea
          id="needs"
          {...register('needs', { 
            required: 'Please tell us what you need'
          })}
          className="h-24"
          placeholder="What kind of support or resources are you looking for?"
        />
        {errors.needs && (
          <p className="text-sm text-red-500">{errors.needs.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="meaningfulGoal">What's most meaningful to you right now?</Label>
        <Textarea
          id="meaningfulGoal"
          {...register('meaningfulGoal', { 
            required: 'Please share what\'s meaningful to you'
          })}
          className="h-24"
          placeholder="Tell us about what matters most to you at this moment in your professional journey - whether it's a goal you're working towards, a challenge you're navigating, or an impact you hope to make..."
        />
        {errors.meaningfulGoal && (
          <p className="text-sm text-red-500">{errors.meaningfulGoal.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label>Skills</Label>
        <InteractiveSelection
          options={skills}
          value={watch('skills')}
          onChange={(value) => setValue('skills', value)}
          maxSelected={5}
          title="Select your skills"
          description="Choose up to 5 skills you're proficient in"
        />
        {errors.skills && (
          <p className="text-sm text-red-500">{errors.skills.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label>Professional Interests</Label>
        <InteractiveSelection
          options={professionalInterests}
          value={watch('professionalInterests')}
          onChange={(value) => setValue('professionalInterests', value)}
          maxSelected={3}
          title="Select your professional interests"
          description="Choose up to 3 areas you're most interested in pursuing or developing"
        />
        {errors.professionalInterests && (
          <p className="text-sm text-red-500">{errors.professionalInterests.message}</p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="industry">Industry</Label>
        <InteractiveSelection
          options={industries}
          value={[watch('industry')].filter(Boolean)}
          onChange={([value]) => setValue('industry', value)}
          maxSelected={1}
          title="Select your industry"
        />
        {errors.industry && (
          <p className="text-sm text-red-500">{errors.industry.message}</p>
        )}
      </div>
    </div>
  );
}

export default ProfessionalInfoStep; 