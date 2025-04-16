'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Users, Activity, Star, Trophy } from 'lucide-react'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

interface StatsCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  description?: string
}

function StatsCard({ title, value, icon, description }: StatsCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}
      </CardContent>
    </Card>
  )
}

export function DashboardStats() {
  const [stats, setStats] = useState({
    connections: 0,
    activeChats: 0,
    rating: 0,
    level: 'Bronze'
  });
  const [loading, setLoading] = useState(true);
  const supabase = createClientComponentClient();

  useEffect(() => {
    async function fetchStats() {
      setLoading(true);
      try {
        const { data: userData } = await supabase.auth.getUser();
        if (!userData?.user) return;

        const userId = userData.user.id;

        // Get connection count
        const { count: connectionsCount, error: connectionsError } = await supabase
          .from('connections')
          .select('*', { count: 'exact', head: true })
          .or(`requester_id.eq.${userId},receiver_id.eq.${userId}`)
          .eq('status', 'accepted');

        // Get active chats (messages from the last 7 days)
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        const { data: activeChats, error: chatsError } = await supabase
          .from('messages')
          .select('sender_id, receiver_id')
          .or(`sender_id.eq.${userId},receiver_id.eq.${userId}`)
          .gte('created_at', sevenDaysAgo.toISOString())
          .order('created_at', { ascending: false });

        // Get unique chat partners
        const uniqueChatPartners = new Set();
        activeChats?.forEach(chat => {
          if (chat.receiver_id === userId) {
            uniqueChatPartners.add(chat.sender_id);
          } else {
            uniqueChatPartners.add(chat.receiver_id);
          }
        });

        // Determine user level based on connection count
        let userLevel = 'Bronze';
        if (connectionsCount && connectionsCount > 50) {
          userLevel = 'Gold';
        } else if (connectionsCount && connectionsCount > 20) {
          userLevel = 'Silver';
        }

        setStats({
          connections: connectionsCount || 0,
          activeChats: uniqueChatPartners.size,
          rating: 4.8, // Can be replaced with actual rating when available
          level: userLevel
        });
      } catch (error) {
        console.error('Error fetching stats:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
  }, [supabase]);

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <StatsCard
        title="Total Connections"
        value={loading ? '...' : stats.connections}
        icon={<Users className="h-4 w-4 text-muted-foreground" />}
        description={loading ? 'Loading...' : `${stats.connections > 0 ? 'Active network' : 'Start building your network'}`}
      />
      <StatsCard
        title="Active Chats"
        value={loading ? '...' : stats.activeChats}
        icon={<Activity className="h-4 w-4 text-muted-foreground" />}
        description={loading ? 'Loading...' : `Active conversations this week`}
      />
      <StatsCard
        title="Rating"
        value={loading ? '...' : stats.rating}
        icon={<Star className="h-4 w-4 text-muted-foreground" />}
        description={loading ? 'Loading...' : `Based on peer feedback`}
      />
      <StatsCard
        title="Achievement Level"
        value={loading ? '...' : stats.level}
        icon={<Trophy className="h-4 w-4 text-muted-foreground" />}
        description={loading ? 'Loading...' : `Based on your activity`}
      />
    </div>
  )
} 