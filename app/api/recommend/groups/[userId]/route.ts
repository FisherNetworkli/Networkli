import { NextResponse } from 'next/server';

export async function GET(request: Request, { params }: { params: { userId: string } }) {
  const { userId } = params;
  const url = new URL(request.url);
  const limit = url.searchParams.get('limit') ?? '10';
  const apiUrl = process.env.NEXT_PUBLIC_PYTHON_API_URL || process.env.NEXT_PUBLIC_API_URL;

  const res = await fetch(`${apiUrl}/recommend/groups/${userId}?limit=${limit}`, {
    headers: { 'Content-Type': 'application/json' },
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
} 