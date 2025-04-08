export interface Profile {
  id: string;
  name: string;
  title: string;
  company: string;
  location: string;
  avatar: string;
  skills: string[];
  interests: string[];
}

export interface Match {
  id: string;
  created_at: string;
  user1: Profile;
  user2: Profile;
}

export interface MatchNotification {
  id: string;
  match_id: string;
  user_id: string;
  seen: boolean;
  created_at: string;
  match?: Match;
} 