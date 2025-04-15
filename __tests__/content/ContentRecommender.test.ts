/// <reference types="jest" />

import { ContentRecommender } from '../../ml/models/content_recommender';
import { prisma } from '../../lib/prisma';
import { ContentService } from '../../services/ContentService';

// Mock Redis
jest.mock('ioredis', () => {
  return jest.fn().mockImplementation(() => ({
    get: jest.fn(),
    setex: jest.fn(),
    del: jest.fn(),
  }));
});

// Mock Prisma
jest.mock('../../lib/prisma', () => ({
  prisma: {
    userInterest: {
      findMany: jest.fn(),
    },
    content: {
      findMany: jest.fn(),
    },
    user: {
      findMany: jest.fn(),
    },
  },
}));

describe('ContentRecommender', () => {
  let recommender: ContentRecommender;
  let contentService: ContentService;

  beforeEach(() => {
    recommender = new ContentRecommender();
    contentService = new ContentService();
  });

  describe('predict', () => {
    it('should return relevance score and boolean for content', async () => {
      const content = 'Test content about machine learning';
      const interests = ['AI', 'machine learning', 'data science'];

      const [score, isRelevant] = await recommender.predict(content, interests);

      expect(typeof score).toBe('number');
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
      expect(typeof isRelevant).toBe('boolean');
    });

    it('should handle empty interests list', async () => {
      const content = 'Test content';
      const interests: string[] = [];

      const [score, isRelevant] = await recommender.predict(content, interests);

      expect(score).toBe(0);
      expect(isRelevant).toBe(false);
    });
  });

  describe('ContentService', () => {
    const mockUser = {
      id: 'user1',
      interests: [
        { interest: { name: 'AI' } },
        { interest: { name: 'machine learning' } },
      ],
    };

    const mockContent = [
      {
        id: 'content1',
        title: 'AI Basics',
        content: 'Introduction to artificial intelligence',
        type: 'article',
        status: 'PUBLISHED',
        publishedAt: new Date(),
        tags: ['AI', 'machine learning'],
      },
      {
        id: 'content2',
        title: 'Web Development',
        content: 'Learn web development',
        type: 'article',
        status: 'PUBLISHED',
        publishedAt: new Date(),
        tags: ['web', 'development'],
      },
    ];

    beforeEach(() => {
      // Reset mocks
      jest.clearAllMocks();

      // Setup mock implementations
      (prisma.userInterest.findMany as jest.Mock).mockResolvedValue(mockUser.interests);
      (prisma.content.findMany as jest.Mock).mockResolvedValue(mockContent);
    });

    it('should return recommendations for a user', async () => {
      const recommendations = await contentService.getRecommendations('user1', 10);

      expect(Array.isArray(recommendations)).toBe(true);
      expect(recommendations.length).toBeGreaterThan(0);
      expect(recommendations[0]).toHaveProperty('relevanceScore');
      expect(recommendations[0]).toHaveProperty('isRelevant');
    });

    it('should filter recommendations by type', async () => {
      const recommendations = await contentService.getRecommendations('user1', 10, 'article');

      expect(Array.isArray(recommendations)).toBe(true);
      recommendations.forEach(rec => {
        expect(rec.type).toBe('article');
      });
    });

    it('should filter recommendations by tag', async () => {
      const recommendations = await contentService.getRecommendations('user1', 10, undefined, 'AI');

      expect(Array.isArray(recommendations)).toBe(true);
      recommendations.forEach(rec => {
        expect(rec.tags).toContain('AI');
      });
    });

    it('should handle cache hits', async () => {
      const cachedRecommendations = mockContent.map(c => ({
        ...c,
        relevanceScore: 0.8,
        isRelevant: true,
      }));

      // Mock Redis cache hit
      const redis = contentService['redis'];
      (redis.get as jest.Mock).mockResolvedValue(JSON.stringify(cachedRecommendations));

      const recommendations = await contentService.getRecommendations('user1', 10);

      expect(recommendations).toEqual(cachedRecommendations);
      expect(prisma.content.findMany).not.toHaveBeenCalled();
    });

    it('should handle errors gracefully', async () => {
      (prisma.userInterest.findMany as jest.Mock).mockRejectedValue(new Error('Database error'));

      await expect(contentService.getRecommendations('user1', 10))
        .rejects
        .toThrow('Database error');
    });
  });
}); 