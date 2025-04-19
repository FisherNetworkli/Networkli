import { NextResponse } from 'next/server';

export async function GET(request: Request, { params }: { params: { userId: string } }) {
  const { userId } = params;
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get('limit') || '10';
  try {
    const apiUrl = `${process.env.EXTERNAL_API_BASE_URL}/recommend/events/${userId}?limit=${limit}`;
    const res = await fetch(apiUrl, { headers: { Authorization: `Bearer ${process.env.EXTERNAL_API_KEY}` } });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    console.error('Error fetching event recommendations:', error);
    return NextResponse.json({ error: 'Failed to fetch event recommendations' }, { status: 500 });
  }
} 