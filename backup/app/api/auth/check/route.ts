import { getServerSession } from 'next-auth';
import { NextResponse } from 'next/server';
import { authOptions } from '../[...nextauth]/auth';

export async function GET() {
  const session = await getServerSession(authOptions);

  if (session) {
    // Redirect based on user role
    if (session.user?.role === 'ADMIN') {
      return NextResponse.json({ redirect: '/admin' });
    } else {
      return NextResponse.json({ redirect: '/dashboard' });
    }
  }

  return NextResponse.json({ redirect: null });
} 