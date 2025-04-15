import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { z } from 'zod';

// Define the sign-up schema
const signUpSchema = z.object({
  // Basic Info
  email: z.string().email(),
  password: z.string().min(8),
  firstName: z.string().min(1),
  lastName: z.string().min(1),
  
  // Professional Info
  title: z.string().optional(),
  company: z.string().optional(),
  industry: z.string().optional(),
  experience: z.string().optional(),
  skills: z.array(z.string()).optional(),
  bio: z.string().optional(),
  expertise: z.array(z.string()).optional(),
  needs: z.array(z.string()).optional(),
  meaningfulGoals: z.string().optional(),
  
  // Preferences
  interests: z.array(z.string()).optional(),
  lookingFor: z.array(z.string()).optional(),
  preferredIndustries: z.array(z.string()).optional(),
  preferredRoles: z.array(z.string()).optional(),
  
  // Social Links
  linkedin: z.string().url().optional(),
  github: z.string().url().optional(),
  portfolio: z.string().url().optional(),
  twitter: z.string().url().optional(),
  
  // Default Settings
  fullName: z.string().optional(),
  profileVisibility: z.enum(['public', 'private']).optional(),
  emailNotifications: z.boolean().optional(),
  marketingEmails: z.boolean().optional(),
});

export async function POST(request: Request) {
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json(
        { error: 'Missing Supabase configuration' },
        { status: 500 }
      );
    }

    const supabase = createClient(supabaseUrl, supabaseKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    });

    const body = await request.json();
    const validatedData = signUpSchema.parse(body);

    const { data, error } = await supabase.auth.admin.createUser({
      email: validatedData.email,
      password: validatedData.password,
      email_confirm: true,
      user_metadata: {
        firstName: validatedData.firstName,
        lastName: validatedData.lastName,
        bio: validatedData.bio,
        expertise: validatedData.expertise || [],
        needs: validatedData.needs || [],
        meaningfulGoals: validatedData.meaningfulGoals,
        profileVisibility: validatedData.profileVisibility || 'public',
        emailNotifications: validatedData.emailNotifications ?? true,
      }
    });

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    return NextResponse.json({ user: data.user }, { status: 201 });
  } catch (error) {
    console.error('Error:', error);
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 