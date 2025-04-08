export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          email: string
          full_name: string | null
          avatar_url: string | null
          title: string | null
          company: string | null
          industry: string | null
          bio: string | null
          location: string | null
          website: string | null
          role: 'user' | 'admin'
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          email: string
          full_name?: string | null
          avatar_url?: string | null
          title?: string | null
          company?: string | null
          industry?: string | null
          bio?: string | null
          location?: string | null
          website?: string | null
          role?: 'user' | 'admin'
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          email?: string
          full_name?: string | null
          avatar_url?: string | null
          title?: string | null
          company?: string | null
          industry?: string | null
          bio?: string | null
          location?: string | null
          website?: string | null
          role?: 'user' | 'admin'
          created_at?: string
          updated_at?: string
        }
      }
      skills: {
        Row: {
          id: string
          name: string
          category: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          name: string
          category?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          name?: string
          category?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      user_skills: {
        Row: {
          id: string
          user_id: string
          skill_id: string
          level: number
          years_experience: number
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          skill_id: string
          level: number
          years_experience: number
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          skill_id?: string
          level?: number
          years_experience?: number
          created_at?: string
          updated_at?: string
        }
      }
      topics: {
        Row: {
          id: string
          name: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          name: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          name?: string
          created_at?: string
          updated_at?: string
        }
      }
      user_interests: {
        Row: {
          id: string
          user_id: string
          topic_id: string
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          topic_id: string
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          topic_id?: string
          created_at?: string
        }
      }
      events: {
        Row: {
          id: string
          title: string
          description: string | null
          format: 'in_person' | 'virtual' | 'hybrid'
          date: string
          location: string | null
          virtual_link: string | null
          max_attendees: number | null
          organizer_id: string
          image_url: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          title: string
          description?: string | null
          format: 'in_person' | 'virtual' | 'hybrid'
          date: string
          location?: string | null
          virtual_link?: string | null
          max_attendees?: number | null
          organizer_id: string
          image_url?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          title?: string
          description?: string | null
          format?: 'in_person' | 'virtual' | 'hybrid'
          date?: string
          location?: string | null
          virtual_link?: string | null
          max_attendees?: number | null
          organizer_id?: string
          image_url?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      event_skills: {
        Row: {
          event_id: string
          skill_id: string
          required_level: number
        }
        Insert: {
          event_id: string
          skill_id: string
          required_level: number
        }
        Update: {
          event_id?: string
          skill_id?: string
          required_level?: number
        }
      }
      event_topics: {
        Row: {
          event_id: string
          topic_id: string
        }
        Insert: {
          event_id: string
          topic_id: string
        }
        Update: {
          event_id?: string
          topic_id?: string
        }
      }
      event_attendees: {
        Row: {
          event_id: string
          profile_id: string
          status: string
          registered_at: string
        }
        Insert: {
          event_id: string
          profile_id: string
          status: string
          registered_at?: string
        }
        Update: {
          event_id?: string
          profile_id?: string
          status?: string
          registered_at?: string
        }
      }
      connections: {
        Row: {
          id: string
          requester_id: string
          receiver_id: string
          status: 'pending' | 'accepted' | 'rejected'
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          requester_id: string
          receiver_id: string
          status?: 'pending' | 'accepted' | 'rejected'
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          requester_id?: string
          receiver_id?: string
          status?: 'pending' | 'accepted' | 'rejected'
          created_at?: string
          updated_at?: string
        }
      }
      messages: {
        Row: {
          id: string
          sender_id: string
          receiver_id: string
          content: string
          read: boolean
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          sender_id: string
          receiver_id: string
          content: string
          read?: boolean
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          sender_id?: string
          receiver_id?: string
          content?: string
          read?: boolean
          created_at?: string
          updated_at?: string
        }
      }
      message_attachments: {
        Row: {
          id: string
          message_id: string
          type: string
          url: string
          name: string
          created_at: string
        }
        Insert: {
          id?: string
          message_id: string
          type: string
          url: string
          name: string
          created_at?: string
        }
        Update: {
          id?: string
          message_id?: string
          type?: string
          url?: string
          name?: string
          created_at?: string
        }
      }
    }
    Functions: {
      get_user_features: {
        Args: {
          user_id: string
        }
        Returns: {
          skill_names: string[]
          interest_names: string[]
        }
      }
      calculate_match_score: {
        Args: {
          user1_id: string
          user2_id: string
        }
        Returns: number
      }
      get_recommended_connections: {
        Args: {
          p_user_id: string
          p_limit?: number
        }
        Returns: {
          id: string
          name: string
          title: string
          company: string
          match_score: number
          mutual_connections: number
          skills: string[]
          interests: string[]
        }[]
      }
      get_recommended_events: {
        Args: {
          p_user_id: string
          p_limit?: number
        }
        Returns: {
          id: string
          title: string
          description: string
          date: string
          format: string
          location: string
          match_score: number
          topics: string[]
          required_skills: string[]
        }[]
      }
      get_recommended_groups: {
        Args: {
          p_user_id: string
          p_limit?: number
        }
        Returns: {
          id: string
          name: string
          description: string
          industry: string
          member_count: number
          match_score: number
          focus_areas: string[]
          relevant_skills: string[]
        }[]
      }
    }
  }
} 