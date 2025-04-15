export interface SignupFormData {
  // Basic Info
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;
  zipCode: string;  // Added zipCode field

  // Professional Info
  title: string;
  company: string;
  industry: string;
  experience: string;
  skills: string[];
  professionalInterests: string[];
  bio: string;  // 100-250 character bio
  expertise: string;  // What you're good at
  needs: string;  // What you need
  meaningfulGoal: string;  // What's most meaningful to you right now
  termsAccepted: boolean;

  // Values & Preferences
  values: string[];  // Core values that drive you
  goals: string[];  // What you want to achieve
  interests: string[];  // Topics that interest you
  networkingStyle: string[];  // How you prefer to connect

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
      (data?.skills?.length ?? 0) > 0 &&
      !!data?.bio &&
      !!data?.expertise &&
      !!data?.needs &&
      !!data?.meaningfulGoal
  },
  {
    id: 'preferences',
    title: 'Preferences',
    description: "What you're looking for",
    isComplete: (data) => 
      (data?.interests?.length ?? 0) > 0 && 
      (data?.networkingStyle?.length ?? 0) > 0
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