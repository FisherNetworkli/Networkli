export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface UserProfile {
  id: string;
  userId: string;
  industry: string;
  experienceLevel: string;
  educationLevel: string;
  currentRole: string;
  company: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Skill {
  id: string;
  name: string;
  category: 'technical' | 'soft' | 'business';
}

export interface Interest {
  id: string;
  name: string;
  category: 'professional' | 'personal' | 'learning';
}

export interface UserActivity {
  id: string;
  userId: string;
  type: string;
  data: any;
  createdAt: Date;
}

export interface UserInteraction {
  id: string;
  userId: string;
  targetUserId: string;
  type: string;
  data: any;
  createdAt: Date;
} 