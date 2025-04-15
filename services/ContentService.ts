import { prisma } from '../lib/prisma';
import { ContentRecommender } from '../ml/models/content_recommender';
import { Redis } from 'ioredis';
import { logger } from '../lib/logger';

interface Content {
  id: string;
  title: string;
  content: string;
  type: string;
  status: string;
  tags: string[];
  authorId: string;
  createdAt: Date;
  updatedAt: Date;
  publishedAt: Date | null;
  relevanceScore?: number;
  isRelevant?: boolean;
}

interface UserInterest {
  interest: {
    name: string;
  };
}

interface User {
  id: string;
}

export class ContentService {
  private model: ContentRecommender;
  private redis: Redis;
  private readonly CACHE_TTL = 3600; // 1 hour

  constructor() {
    this.model = new ContentRecommender();
    this.redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
  }

  /**
   * Get content recommendations for a user
   */
  async getRecommendations(
    userId: string,
    limit: number = 10
  ): Promise<Content[]> {
    try {
      // Check cache first
      const cacheKey = `content:recommendations:${userId}`;
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }

      // Get user interests
      const userInterests = await prisma.userInterest.findMany({
        where: { userId },
        include: { interest: true }
      });

      const interests = userInterests.map((ui: UserInterest) => ui.interest.name);

      // Get all content
      const content = await prisma.content.findMany({
        where: {
          status: 'PUBLISHED',
          publishedAt: { lte: new Date() }
        },
        orderBy: { publishedAt: 'desc' }
      });

      // Score content relevance
      const scoredContent = await Promise.all(
        content.map(async (item: Content) => {
          const [score, isRelevant] = await this.model.predict(
            item.content,
            interests
          );

          return {
            ...item,
            relevanceScore: score,
            isRelevant
          };
        })
      );

      // Filter and sort by relevance
      const recommendations = scoredContent
        .filter((item: Content) => item.isRelevant)
        .sort((a: Content, b: Content) => 
          (b.relevanceScore || 0) - (a.relevanceScore || 0)
        )
        .slice(0, limit);

      // Cache results
      await this.redis.setex(
        cacheKey,
        this.CACHE_TTL,
        JSON.stringify(recommendations)
      );

      return recommendations;
    } catch (error) {
      logger.error('Error getting content recommendations:', error);
      throw error;
    }
  }

  /**
   * Create new content
   */
  async createContent(data: {
    title: string;
    content: string;
    authorId: string;
    type: string;
    tags?: string[];
  }): Promise<Content> {
    try {
      const content = await prisma.content.create({
        data: {
          title: data.title,
          content: data.content,
          authorId: data.authorId,
          type: data.type,
          status: 'DRAFT',
          tags: data.tags || []
        }
      });

      return content;
    } catch (error) {
      logger.error('Error creating content:', error);
      throw error;
    }
  }

  /**
   * Update content
   */
  async updateContent(
    contentId: string,
    data: {
      title?: string;
      content?: string;
      status?: string;
      tags?: string[];
    }
  ): Promise<Content> {
    try {
      const content = await prisma.content.update({
        where: { id: contentId },
        data
      });

      // Invalidate cache
      await this.invalidateContentCache(contentId);

      return content;
    } catch (error) {
      logger.error('Error updating content:', error);
      throw error;
    }
  }

  /**
   * Delete content
   */
  async deleteContent(contentId: string): Promise<void> {
    try {
      await prisma.content.delete({
        where: { id: contentId }
      });

      // Invalidate cache
      await this.invalidateContentCache(contentId);
    } catch (error) {
      logger.error('Error deleting content:', error);
      throw error;
    }
  }

  /**
   * Get content by ID
   */
  async getContentById(contentId: string): Promise<Content> {
    try {
      const content = await prisma.content.findUnique({
        where: { id: contentId },
        include: {
          author: {
            select: {
              id: true,
              name: true,
              image: true
            }
          }
        }
      });

      if (!content) {
        throw new Error('Content not found');
      }

      return content;
    } catch (error) {
      logger.error('Error getting content:', error);
      throw error;
    }
  }

  /**
   * Invalidate content-related caches
   */
  private async invalidateContentCache(contentId: string): Promise<void> {
    try {
      // Get all users who might have this content in their recommendations
      const users = await prisma.user.findMany({
        select: { id: true }
      });

      // Delete recommendation caches for all users
      await Promise.all(
        users.map((user: User) =>
          this.redis.del(`content:recommendations:${user.id}`)
        )
      );
    } catch (error) {
      logger.error('Error invalidating content cache:', error);
    }
  }
} 