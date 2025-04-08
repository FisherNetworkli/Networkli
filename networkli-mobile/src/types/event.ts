export interface Event {
  id: string;
  title: string;
  description: string;
  date: string;
  location: string;
  organizer: string;
  attendees: string[];
  tags: string[];
  imageUrl?: string;
} 