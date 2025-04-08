export interface User {
  id: string;
  email: string;
  name: string;
  title?: string;
  company?: string;
  skills: string[];
  interests: string[];
  connections: string[];
} 