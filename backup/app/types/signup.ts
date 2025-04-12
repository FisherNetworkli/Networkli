export interface SignupFormData {
  // Basic Info
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;

  // Professional Info
  title: string;
  company: string;
  industry: string;
  experience: string;
  skills: string[];
  bio: string;

  // Preferences
  interests: string[];
  lookingFor: string[];
  preferredIndustries: string[];
  preferredRoles: string[];

  // Social Links
  linkedin?: string;
  github?: string;
  portfolio?: string;
  twitter?: string;

  // Default Settings
  profileVisibility: 'public' | 'private' | 'connections';
  emailNotifications: boolean;
  marketingEmails: boolean;
}

export type SignupStep = 
  | 'basic-info'
  | 'professional-info'
  | 'preferences'
  | 'social-links'
  | 'summary';

export interface StepConfig {
  id: SignupStep;
  title: string;
  description: string;
  isComplete: (data: Partial<SignupFormData> | undefined) => boolean;
}

export const SIGNUP_STEPS: StepConfig[] = [
  {
    id: 'basic-info',
    title: 'Basic Information',
    description: 'Tell us about yourself',
    isComplete: (data) => 
      !!data?.firstName && 
      !!data?.lastName && 
      !!data?.email && 
      !!data?.password
  },
  {
    id: 'professional-info',
    title: 'Professional Details',
    description: 'Your work experience and skills',
    isComplete: (data) => 
      !!data?.title && 
      !!data?.company && 
      !!data?.industry && 
      (data?.skills?.length ?? 0) > 0
  },
  {
    id: 'preferences',
    title: 'Preferences',
    description: "What you're looking for",
    isComplete: (data) => 
      (data?.interests?.length ?? 0) > 0 && 
      (data?.lookingFor?.length ?? 0) > 0
  },
  {
    id: 'social-links',
    title: 'Social Links',
    description: 'Connect your profiles',
    isComplete: (data) => 
      !!data?.linkedin || 
      !!data?.github || 
      !!data?.portfolio
  },
  {
    id: 'summary',
    title: 'Review',
    description: 'Review your information',
    isComplete: () => true
  }
]; 