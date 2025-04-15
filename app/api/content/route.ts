import { NextRequest, NextResponse } from 'next/server';
import { ContentService } from '../../../services/ContentService';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const contentService = new ContentService();

export async function GET(req: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const searchParams = req.nextUrl.searchParams;
    const limit = parseInt(searchParams.get('limit') || '10');
    const type = searchParams.get('type');
    const tag = searchParams.get('tag');

    const recommendations = await contentService.getRecommendations(
      session.user.id,
      limit
    );

    // Filter by type and tag if provided
    let filteredContent = recommendations;
    if (type) {
      filteredContent = filteredContent.filter(item => item.type === type);
    }
    if (tag) {
      filteredContent = filteredContent.filter(item => item.tags.includes(tag));
    }

    return NextResponse.json(filteredContent);
  } catch (error) {
    console.error('Error getting content recommendations:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const body = await req.json();
    const { title, content, type, tags } = body;

    if (!title || !content || !type) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    const newContent = await contentService.createContent({
      title,
      content,
      type,
      tags,
      authorId: session.user.id
    });

    return NextResponse.json(newContent);
  } catch (error) {
    console.error('Error creating content:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 