import { NextResponse } from 'next/server';
import { prisma } from '../../../lib/prisma';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      name,
      email,
      phone,
      linkedin,
      github,
      portfolio,
      experience,
      availability,
      salary,
      referral,
      videoUrl
    } = body;

    // Validate required fields
    if (!name || !email || !experience || !availability || !videoUrl) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Create application submission
    const submission = await prisma.applicationSubmission.create({
      data: {
        name,
        email,
        phone: phone || null,
        linkedin: linkedin || null,
        github: github || null,
        portfolio: portfolio || null,
        experience,
        availability,
        salary: salary || null,
        referral: referral || null,
        videoUrl,
        status: 'PENDING', // Default status
      },
    });

    return NextResponse.json(submission);
  } catch (error) {
    console.error('Error creating application submission:', error);
    return NextResponse.json(
      { error: 'Failed to submit application' },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  try {
    const submissions = await prisma.applicationSubmission.findMany({
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json(submissions);
  } catch (error) {
    console.error('Error fetching application submissions:', error);
    return NextResponse.json(
      { error: 'Failed to fetch applications' },
      { status: 500 }
    );
  }
} 