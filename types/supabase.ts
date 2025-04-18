export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  public: {
    Tables: {
      activity_tracking: {
        Row: {
          activity_data: Json | null
          activity_type: string
          created_at: string | null
          id: string
          user_id: string | null
        }
        Insert: {
          activity_data?: Json | null
          activity_type: string
          created_at?: string | null
          id?: string
          user_id?: string | null
        }
        Update: {
          activity_data?: Json | null
          activity_type?: string
          created_at?: string | null
          id?: string
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "activity_tracking_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      blog_posts: {
        Row: {
          author: string
          category: string
          content: string
          created_at: string | null
          date: string | null
          excerpt: string
          id: string
          image: string
          published: boolean | null
          read_time: string
          slug: string
          tags: string[]
          title: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          author: string
          category: string
          content: string
          created_at?: string | null
          date?: string | null
          excerpt: string
          id?: string
          image: string
          published?: boolean | null
          read_time: string
          slug: string
          tags?: string[]
          title: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          author?: string
          category?: string
          content?: string
          created_at?: string | null
          date?: string | null
          excerpt?: string
          id?: string
          image?: string
          published?: boolean | null
          read_time?: string
          slug?: string
          tags?: string[]
          title?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "blog_posts_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
      connections: {
        Row: {
          created_at: string | null
          id: string
          is_demo: boolean | null
          receiver_id: string | null
          requester_id: string | null
          status: string | null
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          is_demo?: boolean | null
          receiver_id?: string | null
          requester_id?: string | null
          status?: string | null
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          is_demo?: boolean | null
          receiver_id?: string | null
          requester_id?: string | null
          status?: string | null
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "connections_receiver_id_fkey"
            columns: ["receiver_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "connections_requester_id_fkey"
            columns: ["requester_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      contact_submissions: {
        Row: {
          created_at: string | null
          email: string
          id: string
          message: string
          name: string
          status: string
          subject: string
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          email: string
          id?: string
          message: string
          name: string
          status?: string
          subject: string
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          email?: string
          id?: string
          message?: string
          name?: string
          status?: string
          subject?: string
          updated_at?: string | null
        }
        Relationships: []
      }
      content: {
        Row: {
          author_id: string
          content: string
          created_at: string | null
          id: string
          published_at: string | null
          slug: string
          status: string
          tags: string[]
          title: string
          type: string
          updated_at: string | null
        }
        Insert: {
          author_id: string
          content: string
          created_at?: string | null
          id?: string
          published_at?: string | null
          slug: string
          status?: string
          tags?: string[]
          title: string
          type: string
          updated_at?: string | null
        }
        Update: {
          author_id?: string
          content?: string
          created_at?: string | null
          id?: string
          published_at?: string | null
          slug?: string
          status?: string
          tags?: string[]
          title?: string
          type?: string
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "fk_content_author"
            columns: ["author_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
      event_attendees: {
        Row: {
          event_id: string
          is_demo: boolean | null
          registered_at: string
          role: string | null
          status: string | null
          user_id: string
        }
        Insert: {
          event_id: string
          is_demo?: boolean | null
          registered_at?: string
          role?: string | null
          status?: string | null
          user_id: string
        }
        Update: {
          event_id?: string
          is_demo?: boolean | null
          registered_at?: string
          role?: string | null
          status?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "event_attendees_event_id_fkey"
            columns: ["event_id"]
            isOneToOne: false
            referencedRelation: "events"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "event_attendees_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      events: {
        Row: {
          category: string | null
          created_at: string | null
          date: string
          description: string | null
          format: string | null
          id: string
          image_url: string | null
          industry: string | null
          is_demo: boolean | null
          location: string | null
          organizer_id: string
          organizer_name: string | null
          title: string
          updated_at: string | null
        }
        Insert: {
          category?: string | null
          created_at?: string | null
          date: string
          description?: string | null
          format?: string | null
          id?: string
          image_url?: string | null
          industry?: string | null
          is_demo?: boolean | null
          location?: string | null
          organizer_id: string
          organizer_name?: string | null
          title: string
          updated_at?: string | null
        }
        Update: {
          category?: string | null
          created_at?: string | null
          date?: string
          description?: string | null
          format?: string | null
          id?: string
          image_url?: string | null
          industry?: string | null
          is_demo?: boolean | null
          location?: string | null
          organizer_id?: string
          organizer_name?: string | null
          title?: string
          updated_at?: string | null
        }
        Relationships: []
      }
      group_members: {
        Row: {
          group_id: string
          is_demo: boolean | null
          joined_at: string
          role: string
          user_id: string
        }
        Insert: {
          group_id: string
          is_demo?: boolean | null
          joined_at?: string
          role?: string
          user_id: string
        }
        Update: {
          group_id?: string
          is_demo?: boolean | null
          joined_at?: string
          role?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "group_members_group_id_fkey"
            columns: ["group_id"]
            isOneToOne: false
            referencedRelation: "groups"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "group_members_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      groups: {
        Row: {
          category: string | null
          created_at: string
          description: string | null
          id: string
          image_url: string | null
          industry: string | null
          is_demo: boolean | null
          location: string | null
          name: string
          organizer_id: string | null
          updated_at: string
        }
        Insert: {
          category?: string | null
          created_at?: string
          description?: string | null
          id?: string
          image_url?: string | null
          industry?: string | null
          is_demo?: boolean | null
          location?: string | null
          name: string
          organizer_id?: string | null
          updated_at?: string
        }
        Update: {
          category?: string | null
          created_at?: string
          description?: string | null
          id?: string
          image_url?: string | null
          industry?: string | null
          is_demo?: boolean | null
          location?: string | null
          name?: string
          organizer_id?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "groups_organizer_id_fkey"
            columns: ["organizer_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      interaction_history: {
        Row: {
          created_at: string | null
          id: string
          interaction_type: string
          is_demo: boolean | null
          metadata: Json | null
          target_entity_id: string | null
          target_entity_type: string | null
          timestamp: string
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          interaction_type: string
          is_demo?: boolean | null
          metadata?: Json | null
          target_entity_id?: string | null
          target_entity_type?: string | null
          timestamp?: string
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          interaction_type?: string
          is_demo?: boolean | null
          metadata?: Json | null
          target_entity_id?: string | null
          target_entity_type?: string | null
          timestamp?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "interaction_history_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      interests: {
        Row: {
          created_at: string | null
          id: string
          name: string
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          name: string
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          name?: string
          updated_at?: string | null
        }
        Relationships: []
      }
      job_applications: {
        Row: {
          cover_letter: string | null
          created_at: string | null
          email: string
          experience: string
          id: string
          name: string
          phone: string
          position: string
          resume: string
          status: string
          updated_at: string | null
        }
        Insert: {
          cover_letter?: string | null
          created_at?: string | null
          email: string
          experience: string
          id?: string
          name: string
          phone: string
          position: string
          resume: string
          status?: string
          updated_at?: string | null
        }
        Update: {
          cover_letter?: string | null
          created_at?: string | null
          email?: string
          experience?: string
          id?: string
          name?: string
          phone?: string
          position?: string
          resume?: string
          status?: string
          updated_at?: string | null
        }
        Relationships: []
      }
      messages: {
        Row: {
          content: string
          created_at: string | null
          id: string
          read: boolean | null
          receiver_id: string | null
          sender_id: string | null
          updated_at: string | null
        }
        Insert: {
          content: string
          created_at?: string | null
          id?: string
          read?: boolean | null
          receiver_id?: string | null
          sender_id?: string | null
          updated_at?: string | null
        }
        Update: {
          content?: string
          created_at?: string | null
          id?: string
          read?: boolean | null
          receiver_id?: string | null
          sender_id?: string | null
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "messages_receiver_id_fkey"
            columns: ["receiver_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "messages_sender_id_fkey"
            columns: ["sender_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      notifications: {
        Row: {
          created_at: string | null
          id: string
          message: string
          read: boolean | null
          related_entity_id: string | null
          related_entity_type: string | null
          type: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          message: string
          read?: boolean | null
          related_entity_id?: string | null
          related_entity_type?: string | null
          type: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          message?: string
          read?: boolean | null
          related_entity_id?: string | null
          related_entity_type?: string | null
          type?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      profile_skills: {
        Row: {
          created_at: string | null
          id: string
          profile_id: string
          skill_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          profile_id: string
          skill_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          profile_id?: string
          skill_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "profile_skills_profile_id_fkey"
            columns: ["profile_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "profile_skills_skill_id_fkey"
            columns: ["skill_id"]
            isOneToOne: false
            referencedRelation: "skills"
            referencedColumns: ["id"]
          },
        ]
      }
      profile_views: {
        Row: {
          created_at: string | null
          id: string
          metadata: Json | null
          profile_id: string
          referrer: string | null
          source: string | null
          view_date: string
          visitor_id: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          metadata?: Json | null
          profile_id: string
          referrer?: string | null
          source?: string | null
          view_date?: string
          visitor_id?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          metadata?: Json | null
          profile_id?: string
          referrer?: string | null
          source?: string | null
          view_date?: string
          visitor_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "profile_views_profile_id_fkey"
            columns: ["profile_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "profile_views_visitor_id_fkey"
            columns: ["visitor_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          bio: string | null
          bio_embedding: Json | null
          city: string | null
          company: string | null
          country_code: string | null
          county: string | null
          created_at: string | null
          domain_id: number | null
          email: string
          email_notifications: boolean | null
          experience_level: string | null
          expertise: string | null
          expertise_embedding: Json | null
          first_name: string | null
          full_name: string | null
          github_url: string | null
          goals_embedding: Json | null
          headline: string | null
          id: string
          industry: string | null
          interests: string[] | null
          is_celebrity: boolean | null
          is_demo: boolean | null
          is_premium: boolean | null
          is_prospect: boolean | null
          last_name: string | null
          lat: number | null
          linkedin_url: string | null
          lng: number | null
          location: string | null
          marketing_emails: boolean | null
          meaningful_goals: string | null
          needs: string | null
          needs_embedding: Json | null
          portfolio_url: string | null
          professional_goals: string[] | null
          profile_visibility: string | null
          role: Database["public"]["Enums"]["user_role"] | null
          skills: string[] | null
          state: string | null
          state_code: string | null
          title: string | null
          twitter_url: string | null
          updated_at: string | null
          values: string[] | null
          website: string | null
          zip_code: string | null
        }
        Insert: {
          avatar_url?: string | null
          bio?: string | null
          bio_embedding?: Json | null
          city?: string | null
          company?: string | null
          country_code?: string | null
          county?: string | null
          created_at?: string | null
          domain_id?: number | null
          email: string
          email_notifications?: boolean | null
          experience_level?: string | null
          expertise?: string | null
          expertise_embedding?: Json | null
          first_name?: string | null
          full_name?: string | null
          github_url?: string | null
          goals_embedding?: Json | null
          headline?: string | null
          id?: string
          industry?: string | null
          interests?: string[] | null
          is_celebrity?: boolean | null
          is_demo?: boolean | null
          is_premium?: boolean | null
          is_prospect?: boolean | null
          last_name?: string | null
          lat?: number | null
          linkedin_url?: string | null
          lng?: number | null
          location?: string | null
          marketing_emails?: boolean | null
          meaningful_goals?: string | null
          needs?: string | null
          needs_embedding?: Json | null
          portfolio_url?: string | null
          professional_goals?: string[] | null
          profile_visibility?: string | null
          role?: Database["public"]["Enums"]["user_role"] | null
          skills?: string[] | null
          state?: string | null
          state_code?: string | null
          title?: string | null
          twitter_url?: string | null
          updated_at?: string | null
          values?: string[] | null
          website?: string | null
          zip_code?: string | null
        }
        Update: {
          avatar_url?: string | null
          bio?: string | null
          bio_embedding?: Json | null
          city?: string | null
          company?: string | null
          country_code?: string | null
          county?: string | null
          created_at?: string | null
          domain_id?: number | null
          email?: string
          email_notifications?: boolean | null
          experience_level?: string | null
          expertise?: string | null
          expertise_embedding?: Json | null
          first_name?: string | null
          full_name?: string | null
          github_url?: string | null
          goals_embedding?: Json | null
          headline?: string | null
          id?: string
          industry?: string | null
          interests?: string[] | null
          is_celebrity?: boolean | null
          is_demo?: boolean | null
          is_premium?: boolean | null
          is_prospect?: boolean | null
          last_name?: string | null
          lat?: number | null
          linkedin_url?: string | null
          lng?: number | null
          location?: string | null
          marketing_emails?: boolean | null
          meaningful_goals?: string | null
          needs?: string | null
          needs_embedding?: Json | null
          portfolio_url?: string | null
          professional_goals?: string[] | null
          profile_visibility?: string | null
          role?: Database["public"]["Enums"]["user_role"] | null
          skills?: string[] | null
          state?: string | null
          state_code?: string | null
          title?: string | null
          twitter_url?: string | null
          updated_at?: string | null
          values?: string[] | null
          website?: string | null
          zip_code?: string | null
        }
        Relationships: []
      }
      skills: {
        Row: {
          created_at: string | null
          id: string
          name: string
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          name: string
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          name?: string
          updated_at?: string | null
        }
        Relationships: []
      }
      subscriptions: {
        Row: {
          cancel_at: string | null
          cancel_at_period_end: boolean
          canceled_at: string | null
          created_at: string
          current_period_end: string
          current_period_start: string
          ended_at: string | null
          id: string
          price_id: string
          quantity: number
          status: string
          subscription_id: string
          trial_end: string | null
          trial_start: string | null
          user_id: string | null
        }
        Insert: {
          cancel_at?: string | null
          cancel_at_period_end?: boolean
          canceled_at?: string | null
          created_at?: string
          current_period_end: string
          current_period_start: string
          ended_at?: string | null
          id?: string
          price_id: string
          quantity: number
          status: string
          subscription_id: string
          trial_end?: string | null
          trial_start?: string | null
          user_id?: string | null
        }
        Update: {
          cancel_at?: string | null
          cancel_at_period_end?: boolean
          canceled_at?: string | null
          created_at?: string
          current_period_end?: string
          current_period_start?: string
          ended_at?: string | null
          id?: string
          price_id?: string
          quantity?: number
          status?: string
          subscription_id?: string
          trial_end?: string | null
          trial_start?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "subscriptions_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      system_settings: {
        Row: {
          created_at: string | null
          id: string
          key: string
          updated_at: string | null
          value: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          key: string
          updated_at?: string | null
          value: string
        }
        Update: {
          created_at?: string | null
          id?: string
          key?: string
          updated_at?: string | null
          value?: string
        }
        Relationships: []
      }
      user_interests: {
        Row: {
          created_at: string | null
          id: string
          interest_id: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          interest_id: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          interest_id?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_interests_interest_id_fkey"
            columns: ["interest_id"]
            isOneToOne: false
            referencedRelation: "interests"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "user_interests_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
      user_preferences: {
        Row: {
          created_at: string | null
          id: string
          interests: string[] | null
          networking_style: string[] | null
          professional_goals: string[] | null
          updated_at: string | null
          user_id: string | null
          values: string[] | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          interests?: string[] | null
          networking_style?: string[] | null
          professional_goals?: string[] | null
          updated_at?: string | null
          user_id?: string | null
          values?: string[] | null
        }
        Update: {
          created_at?: string | null
          id?: string
          interests?: string[] | null
          networking_style?: string[] | null
          professional_goals?: string[] | null
          updated_at?: string | null
          user_id?: string | null
          values?: string[] | null
        }
        Relationships: [
          {
            foreignKeyName: "user_preferences_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      user_skills: {
        Row: {
          created_at: string | null
          id: string
          skill_name: string
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          skill_name: string
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          skill_name?: string
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "user_skills_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      users: {
        Row: {
          created_at: string | null
          email: string | null
          id: string
          image: string | null
          name: string | null
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          email?: string | null
          id?: string
          image?: string | null
          name?: string | null
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          email?: string | null
          id?: string
          image?: string | null
          name?: string | null
          updated_at?: string | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      delete_demo_profiles: {
        Args: Record<PropertyKey, never>
        Returns: undefined
      }
      execute_sql: {
        Args: { sql: string }
        Returns: undefined
      }
      get_database_info: {
        Args: Record<PropertyKey, never>
        Returns: Json
      }
      get_group_member_counts: {
        Args: Record<PropertyKey, never>
        Returns: {
          group_id: string
          count: number
        }[]
      }
      record_profile_view: {
        Args: {
          profile_id: string
          visitor_id: string
          source?: string
          referrer?: string
          metadata?: Json
        }
        Returns: string
      }
      seed_demo_connection: {
        Args: { p_requester_id: string; p_receiver_id: string }
        Returns: undefined
      }
      seed_demo_interaction: {
        Args: {
          p_user_id: string
          p_interaction_type: string
          p_target_entity_type: string
          p_target_entity_id: string
          p_metadata?: Json
        }
        Returns: undefined
      }
    }
    Enums: {
      user_role:
        | "user"
        | "admin"
        | "organizer"
        | "premium"
        | "demo"
        | "demo_user"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DefaultSchema = Database[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof Database },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof Database },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends { schema: keyof Database }
  ? Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      user_role: ["user", "admin", "organizer", "premium", "demo", "demo_user"],
    },
  },
} as const
