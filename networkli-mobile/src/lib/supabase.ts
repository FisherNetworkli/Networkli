import 'react-native-url-polyfill/auto'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { createClient } from '@supabase/supabase-js'
import { Database } from './database.types'

const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false
  },
  realtime: {
    params: {
      eventsPerSecond: 10
    }
  }
})

// Helper to get the current user's ID safely
export const getCurrentUserId = async () => {
  const { data: { user } } = await supabase.auth.getUser()
  return user?.id
}

// Helper to get the current user's profile
export const getCurrentProfile = async () => {
  const userId = await getCurrentUserId()
  if (!userId) return null

  const { data: profile, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single()

  if (error) {
    console.error('Error fetching profile:', error)
    return null
  }

  return profile
}

// Typed database functions
export const db = {
  // Profiles
  profiles: {
    get: async (userId: string) => {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single()
      if (error) throw error
      return data
    },
    update: async (userId: string, updates: Partial<Database['public']['Tables']['profiles']['Update']>) => {
      const { data, error } = await supabase
        .from('profiles')
        .update(updates)
        .eq('id', userId)
      if (error) throw error
      return data
    }
  },

  // Connections
  connections: {
    getRecommended: async (limit = 10) => {
      const userId = await getCurrentUserId()
      if (!userId) throw new Error('No user logged in')
      
      const { data, error } = await supabase
        .rpc('get_recommended_connections', { p_user_id: userId, p_limit: limit })
      if (error) throw error
      return data
    },
    create: async (connectedUserId: string) => {
      const userId = await getCurrentUserId()
      if (!userId) throw new Error('No user logged in')

      const { data, error } = await supabase
        .from('connections')
        .insert({
          user_id: userId,
          connected_user_id: connectedUserId,
          status: 'pending'
        })
      if (error) throw error
      return data
    }
  },

  // Events
  events: {
    getRecommended: async (limit = 5) => {
      const userId = await getCurrentUserId()
      if (!userId) throw new Error('No user logged in')

      const { data, error } = await supabase
        .rpc('get_recommended_events', { p_user_id: userId, p_limit: limit })
      if (error) throw error
      return data
    },
    create: async (event: Omit<Database['public']['Tables']['events']['Insert'], 'id' | 'created_at' | 'updated_at'>) => {
      const { data, error } = await supabase
        .from('events')
        .insert(event)
      if (error) throw error
      return data
    }
  },

  // Groups
  groups: {
    getRecommended: async (limit = 5) => {
      const userId = await getCurrentUserId()
      if (!userId) throw new Error('No user logged in')

      const { data, error } = await supabase
        .rpc('get_recommended_groups', { p_user_id: userId, p_limit: limit })
      if (error) throw error
      return data
    },
    create: async (group: Omit<Database['public']['Tables']['groups']['Insert'], 'id' | 'created_at' | 'updated_at'>) => {
      const { data, error } = await supabase
        .from('groups')
        .insert(group)
      if (error) throw error
      return data
    }
  },

  // Messages
  messages: {
    send: async (receiverId: string, content: string, attachments?: Database['public']['Tables']['message_attachments']['Insert'][]) => {
      const userId = await getCurrentUserId()
      if (!userId) throw new Error('No user logged in')

      // Start a transaction to insert message and attachments
      const { data: message, error: messageError } = await supabase
        .from('messages')
        .insert({
          sender_id: userId,
          receiver_id: receiverId,
          content,
          read: false
        })
        .select()
        .single()

      if (messageError) throw messageError

      if (attachments && attachments.length > 0) {
        const { error: attachmentError } = await supabase
          .from('message_attachments')
          .insert(attachments.map(attachment => ({
            ...attachment,
            message_id: message.id
          })))

        if (attachmentError) throw attachmentError
      }

      return message
    },
    getConversation: async (otherUserId: string, limit = 50) => {
      const userId = await getCurrentUserId()
      if (!userId) throw new Error('No user logged in')

      const { data, error } = await supabase
        .from('messages')
        .select(`
          *,
          attachments:message_attachments(*)
        `)
        .or(`sender_id.eq.${userId},receiver_id.eq.${userId}`)
        .or(`sender_id.eq.${otherUserId},receiver_id.eq.${otherUserId}`)
        .order('created_at', { ascending: false })
        .limit(limit)

      if (error) throw error
      return data
    },
    subscribeToNewMessages: (callback: (message: Database['public']['Tables']['messages']['Row']) => void) => {
      const userId = getCurrentUserId()
      if (!userId) throw new Error('No user logged in')

      return supabase
        .channel('messages')
        .on(
          'postgres_changes',
          {
            event: 'INSERT',
            schema: 'public',
            table: 'messages',
            filter: `receiver_id=eq.${userId}`
          },
          callback
        )
        .subscribe()
    }
  }
} 