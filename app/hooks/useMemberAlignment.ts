import { useState, useEffect } from "react";
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

export interface AlignedMember {
  id: string;
  full_name: string;
  avatar_url?: string;
  title?: string;
  company?: string;
  similarity_score: number;
  alignment_reason?: string;
}

interface UseMemberAlignmentOptions {
  userId: string;
  entityType: "group" | "event";
  entityId: string;
  limit?: number;
  minSimilarity?: number;
  enabled?: boolean;
}

interface UseMemberAlignmentResult {
  alignedMembers: AlignedMember[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch member alignment data based on a user and group/event.
 */
export function useMemberAlignment({
  userId,
  entityType,
  entityId,
  limit = 5,
  minSimilarity = 0.1,
  enabled = true,
}: UseMemberAlignmentOptions): UseMemberAlignmentResult {
  const supabase = createClientComponentClient();
  const [alignedMembers, setAlignedMembers] = useState<AlignedMember[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchAlignment = async () => {
    if (!userId || !entityType || !entityId || !enabled) return;

    setIsLoading(true);
    setError(null);

    try {
      // Prepare API URL with query parameters
      const url = new URL(`/api/member-alignment/${userId}`, window.location.origin);
      url.searchParams.append("entity_type", entityType);
      url.searchParams.append("entity_id", entityId);
      url.searchParams.append("limit", limit.toString());
      url.searchParams.append("min_similarity", minSimilarity.toString());

      // Call the API
      const response = await fetch(url.toString(), {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(
          `Failed to fetch alignment data: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      setAlignedMembers(data.aligned_members || []);
    } catch (err) {
      console.error("Error fetching member alignment:", err);
      setError(err instanceof Error ? err : new Error(String(err)));
      setAlignedMembers([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch data on initial render and when dependencies change
  useEffect(() => {
    if (enabled) {
      fetchAlignment();
    }
  }, [userId, entityType, entityId, limit, minSimilarity, enabled]);

  return {
    alignedMembers,
    isLoading,
    error,
    refetch: fetchAlignment,
  };
} 