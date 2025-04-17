import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useSupabase } from '@/app/supabase-provider'
import Avatar from '@/app/components/Avatar'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'
import { Loader2, UserPlus, Check } from 'lucide-react'

type Recommendation = {
  id: string
  created_at: string
  avatar_url: string | null
  full_name: string
  headline: string | null
  match_score: number
  match_reason: string
  is_connected: boolean
  is_requested: boolean
}

interface AlignedMembersListProps {
  userId: string
  entityType: 'group' | 'event'
  entityId: string
  limit?: number
}

export default function AlignedMembersList({
  userId,
  entityType,
  entityId,
  limit = 5,
}: AlignedMembersListProps) {
  const { supabase } = useSupabase()
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [pendingRequests, setPendingRequests] = useState<Record<string, boolean>>({})

  useEffect(() => {
    async function fetchRecommendations() {
      try {
        setLoading(true)
        setError(null)

        const response = await fetch(
          `/api/recommendations/${entityType}/${entityId}/members?userId=${userId}&limit=${limit}`,
          {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          }
        )

        if (!response.ok) {
          throw new Error('Failed to fetch recommendations')
        }

        const data = await response.json()
        setRecommendations(data.recommendations || [])
      } catch (err) {
        console.error('Error fetching recommendations:', err)
        setError('Unable to load recommendations. Please try again later.')
      } finally {
        setLoading(false)
      }
    }

    if (userId && entityType && entityId) {
      fetchRecommendations()
    }
  }, [userId, entityType, entityId, limit])

  const handleConnect = async (profileId: string) => {
    try {
      setPendingRequests((prev) => ({ ...prev, [profileId]: true }))

      const { error } = await supabase.from('connection_requests').insert([
        {
          sender_id: userId,
          receiver_id: profileId,
          status: 'pending',
        },
      ])

      if (error) throw error

      // Update the local state to show the request was sent
      setRecommendations((prev) =>
        prev.map((rec) =>
          rec.id === profileId ? { ...rec, is_requested: true } : rec
        )
      )

      router.refresh()
    } catch (err) {
      console.error('Error sending connection request:', err)
    } finally {
      setPendingRequests((prev) => ({ ...prev, [profileId]: false }))
    }
  }

  const viewProfile = (profileId: string) => {
    router.push(`/profile/${profileId}`)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (error) {
    return <div className="text-center text-red-500 py-4">{error}</div>
  }

  if (recommendations.length === 0) {
    return (
      <div className="text-center text-muted-foreground py-4">
        No aligned members found. Check back later for new recommendations.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {recommendations.map((recommendation) => (
        <Card key={recommendation.id} className="overflow-hidden hover:shadow-md transition-shadow">
          <CardContent className="p-4">
            <div className="flex items-start gap-4">
              <div onClick={() => viewProfile(recommendation.id)} className="cursor-pointer">
                <Avatar
                  url={recommendation.avatar_url}
                  size={50}
                  name={recommendation.full_name}
                />
              </div>
              <div className="flex-1 min-w-0">
                <div
                  onClick={() => viewProfile(recommendation.id)}
                  className="cursor-pointer hover:underline"
                >
                  <h3 className="font-medium text-foreground truncate">
                    {recommendation.full_name}
                  </h3>
                </div>
                {recommendation.headline && (
                  <p className="text-sm text-muted-foreground truncate">
                    {recommendation.headline}
                  </p>
                )}
                <div className="mt-2">
                  <p className="text-sm text-muted-foreground">{recommendation.match_reason}</p>
                </div>
              </div>
              <div className="flex-shrink-0">
                {recommendation.is_connected ? (
                  <Button variant="outline" size="sm" disabled className="whitespace-nowrap">
                    <Check className="h-4 w-4 mr-1" />
                    Connected
                  </Button>
                ) : recommendation.is_requested ? (
                  <Button variant="outline" size="sm" disabled className="whitespace-nowrap">
                    Request Sent
                  </Button>
                ) : (
                  <Button
                    onClick={() => handleConnect(recommendation.id)}
                    variant="default"
                    size="sm"
                    disabled={!!pendingRequests[recommendation.id]}
                    className="whitespace-nowrap"
                  >
                    {pendingRequests[recommendation.id] ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <UserPlus className="h-4 w-4 mr-1" />
                    )}
                    Connect
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
} 