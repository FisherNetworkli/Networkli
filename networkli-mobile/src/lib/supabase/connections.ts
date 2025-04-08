import { supabase } from './client';

export interface PotentialConnection {
  id: string;
  name: string;
  title: string;
  company: string;
  location: string;
  avatar: string;
  skills: string[];
  interests: string[];
  synergyScore: number;
  matchReasons: string[];
}

export async function getPotentialConnections(limit = 10): Promise<PotentialConnection[]> {
  const { data, error } = await supabase
    .rpc('get_potential_connections', { limit_val: limit });

  if (error) throw error;
  return data || [];
}

export async function recordSwipe(targetUserId: string, action: 'left' | 'right') {
  const { error } = await supabase
    .from('connection_swipes')
    .insert({
      target_user_id: targetUserId,
      action: action
    });

  if (error) throw error;
}

export async function getMatches() {
  const { data, error } = await supabase
    .from('matches')
    .select(`
      id,
      created_at,
      user1:user1_id (
        id,
        name,
        title,
        avatar
      ),
      user2:user2_id (
        id,
        name,
        title,
        avatar
      )
    `)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data || [];
}

export function subscribeToNewMatches(callback: (match: any) => void) {
  return supabase
    .channel('matches')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'match_notifications',
        filter: `user_id=eq.${supabase.auth.getUser()?.data.user?.id}`
      },
      async (payload) => {
        // Fetch the full match details
        const { data: matchData, error } = await supabase
          .from('matches')
          .select(`
            id,
            created_at,
            user1:user1_id (
              id,
              name,
              title,
              avatar
            ),
            user2:user2_id (
              id,
              name,
              title,
              avatar
            )
          `)
          .eq('id', payload.new.match_id)
          .single();

        if (!error && matchData) {
          callback(matchData);
        }
      }
    )
    .subscribe();
}

export async function markNotificationAsSeen(notificationId: string) {
  const { error } = await supabase
    .from('match_notifications')
    .update({ seen: true })
    .eq('id', notificationId);

  if (error) throw error;
} 