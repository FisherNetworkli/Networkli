import { createClient } from '@supabase/supabase-js';
import Config from 'react-native-config';
import 'react-native-url-polyfill/auto';
import { User } from '../types/user';
import { Recommendation } from '../types/recommendation';
import { Event } from '../types/event';
import { Group } from '../types/group';

// Initialize Supabase client
const supabaseUrl = Config.SUPABASE_URL;
const supabaseKey = Config.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error('Missing Supabase configuration');
}

const supabase = createClient(supabaseUrl, supabaseKey);

export interface ApiService {
  getProfile(): Promise<User>;
  updateProfile(data: Partial<User>): Promise<User>;
  getRecommendations(): Promise<Recommendation[]>;
  createConnection(userId: string): Promise<void>;
  getConnections(): Promise<User[]>;
  syncBumbleData(): Promise<void>;
  getRecommendedEvents(): Promise<Event[]>;
  getRecommendedGroups(): Promise<Group[]>;
}

class ApiServiceImpl implements ApiService {
  async getProfile(): Promise<User> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('User not found');

      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user.id)
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching profile:', error);
      throw error;
    }
  }

  async updateProfile(data: Partial<User>): Promise<User> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('User not found');

      const { data, error } = await supabase
        .from('profiles')
        .update(data)
        .eq('id', user.id)
        .select('*')
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error updating profile:', error);
      throw error;
    }
  }

  async getRecommendations(): Promise<Recommendation[]> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return [];

      const { data, error } = await supabase
        .rpc('get_recommendations', { user_id: user.id })
        .select('*');

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      throw error;
    }
  }

  async createConnection(userId: string): Promise<void> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('User not found');

      const { error } = await supabase
        .from('connections')
        .insert([
          {
            user_id: user.id,
            connected_user_id: userId,
            status: 'pending'
          }
        ]);

      if (error) throw error;
    } catch (error) {
      console.error('Error creating connection:', error);
      throw error;
    }
  }

  async getConnections(): Promise<User[]> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return [];

      const { data, error } = await supabase
        .from('connections')
        .select(`
          connected_user_id,
          connected_user:profiles!connected_user_id(*)
        `)
        .eq('user_id', user.id)
        .eq('status', 'accepted');

      if (error) throw error;
      return data.map(d => d.connected_user) || [];
    } catch (error) {
      console.error('Error fetching connections:', error);
      throw error;
    }
  }

  async syncBumbleData(): Promise<void> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('User not found');

      const { error } = await supabase
        .functions.invoke('sync-bumble-data', {
          body: { user_id: user.id }
        });

      if (error) throw error;
    } catch (error) {
      console.error('Error syncing Bumble data:', error);
      throw error;
    }
  }

  async getRecommendedEvents(): Promise<Event[]> {
    try {
      const response = await supabase.get('/api/recommendations/events');
      return response.data;
    } catch (error) {
      console.error('Error fetching recommended events:', error);
      throw error;
    }
  }

  async getRecommendedGroups(): Promise<Group[]> {
    try {
      const response = await supabase.get('/api/recommendations/groups');
      return response.data;
    } catch (error) {
      console.error('Error fetching recommended groups:', error);
      throw error;
    }
  }
}

export const api = new ApiServiceImpl(); 