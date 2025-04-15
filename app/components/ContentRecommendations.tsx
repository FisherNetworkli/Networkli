import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';

interface Content {
  id: string;
  title: string;
  content: string;
  type: string;
  tags: string[];
  relevanceScore: number;
  authorId: string;
  createdAt: string;
  updatedAt: string;
  publishedAt: string | null;
}

interface ContentRecommendationsProps {
  limit?: number;
  type?: string;
  tag?: string;
}

export default function ContentRecommendations({
  limit = 3,
  type,
  tag
}: ContentRecommendationsProps) {
  const [recommendations, setRecommendations] = useState<Content[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        const params = new URLSearchParams();
        if (limit) params.append('limit', limit.toString());
        if (type) params.append('type', type);
        if (tag) params.append('tag', tag);

        const response = await fetch(`/api/content?${params.toString()}`);
        if (!response.ok) {
          throw new Error('Failed to fetch recommendations');
        }

        const data = await response.json();
        setRecommendations(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [limit, type, tag]);

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
        <div className="space-y-3">
          {[...Array(limit)].map((_, i) => (
            <div key={i} className="h-24 bg-gray-200 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500">
        Error loading recommendations: {error}
      </div>
    );
  }

  if (recommendations.length === 0) {
    return (
      <div className="text-gray-500">
        No recommendations available at this time.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {recommendations.map((content) => (
        <Link
          href={`/content/${content.id}`}
          key={content.id}
          className="block group"
        >
          <div className="bg-white rounded-lg shadow-md overflow-hidden transition-transform duration-300 group-hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm text-gray-500">{content.type}</span>
                {content.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <h3 className="text-xl font-semibold mb-2 group-hover:text-blue-600">
                {content.title}
              </h3>
              <p className="text-gray-600 line-clamp-2">
                {content.content}
              </p>
              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-gray-500">
                  {new Date(content.publishedAt || content.createdAt).toLocaleDateString()}
                </div>
                <div className="text-sm text-gray-500">
                  Relevance: {Math.round(content.relevanceScore * 100)}%
                </div>
              </div>
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
} 