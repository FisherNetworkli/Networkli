import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { z } from 'zod';
import 'dotenv/config';

// Validation schema for registration
const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  firstName: z.string(),
  lastName: z.string(),
  confirmPassword: z.string(),
  zipCode: z.string().regex(/^\d{5}$/, "ZIP code must be 5 digits"),
  skills: z.array(z.string()).optional(),
  interests: z.array(z.string()).optional(),
  professionalGoals: z.array(z.string()).optional(),
  values: z.array(z.string()).optional(),
  networkingStyle: z.array(z.string()).optional(),
  title: z.string().optional(),
  company: z.string().optional(),
  industry: z.string().optional(),
  experience: z.string().optional(),
  bio: z.string().optional(),
  expertise: z.string().optional(),
  needs: z.string().optional(),
  meaningfulGoal: z.string().optional(),
  linkedin: z.string().optional(),
  github: z.string().optional(),
  portfolio: z.string().optional(),
  twitter: z.string().optional(),
  profileVisibility: z.string().optional(),
  emailNotifications: z.boolean(),
  marketingEmails: z.boolean(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export async function POST(req: Request) {
  console.log('=== Starting registration process ===');
  try {
    const cookieStore = cookies();
    console.log('Creating Supabase clients...');
    const supabase = createRouteHandlerClient({ cookies }, { 
      supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
      supabaseKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      options: {
        db: {
          schema: 'public'
        }
      }
    });
    
    // Create service role client for admin operations
    console.log('Environment variables check:');
    console.log('- NEXT_PUBLIC_SUPABASE_URL:', process.env.NEXT_PUBLIC_SUPABASE_URL);
    console.log('- SUPABASE_SERVICE_ROLE_KEY:', process.env.SUPABASE_SERVICE_ROLE_KEY?.substring(0, 10) + '...');
    
    const serviceRoleClient = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!,
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false
        },
        db: {
          schema: 'public'
        }
      }
    );

    console.log('Service role client created with config:', {
      url: process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
      keyLength: process.env.SUPABASE_SERVICE_ROLE_KEY?.length
    });
    
    console.log('Parsing request body...');
    const data = await req.json();
    console.log('Request data:', {
      ...data,
      password: '[REDACTED]',
      confirmPassword: '[REDACTED]'
    });
    
    // Validate input
    console.log('Validating input data...');
    const validatedData = registerSchema.parse(data);
    console.log('Input validation successful');

    // Check if email already exists in auth.users
    console.log('Checking if email already exists...');
    const { data: existingUser } = await serviceRoleClient.auth.admin.listUsers();
    
    if (existingUser?.users.some(user => user.email === validatedData.email)) {
      console.log('Email already exists');
      return NextResponse.json(
        { error: 'Email already registered' },
        { status: 400 }
      );
    }

    // Create user with Supabase Auth using service role client
    console.log('Creating user with Supabase Auth...');
    console.log('Service role client config:', {
      url: process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
      keyLength: process.env.SUPABASE_SERVICE_ROLE_KEY?.length
    });

    const { data: authData, error: authError } = await serviceRoleClient.auth.admin.createUser({
      email: validatedData.email,
      password: validatedData.password,
      email_confirm: true,
      user_metadata: {
        firstName: validatedData.firstName,
        lastName: validatedData.lastName,
        email: validatedData.email,
        skills: validatedData.skills || [],
        interests: validatedData.interests || [],
        professionalGoals: validatedData.professionalGoals || [],
        values: validatedData.values || [],
      }
    });

    if (authError) {
      console.error('Supabase Auth error:', {
        message: authError.message,
        status: authError.status,
        name: authError.name,
        code: authError.code,
        stack: authError.stack
      });
      return NextResponse.json(
        { error: authError.message },
        { status: 400 }
      );
    }

    if (!authData?.user) {
      console.error('No user data returned from auth signup');
      return NextResponse.json(
        { error: 'Failed to create user - no user data returned' },
        { status: 500 }
      );
    }

    const userId = authData.user.id;

    // Initialize profile with basic info
    const { error: basicProfileError } = await serviceRoleClient
      .from('profiles')
      .update({
        email: validatedData.email,
        first_name: validatedData.firstName,
        last_name: validatedData.lastName,
        full_name: `${validatedData.firstName} ${validatedData.lastName}`,
        profile_visibility: validatedData.profileVisibility || 'public',
        email_notifications: validatedData.emailNotifications,
        marketing_emails: validatedData.marketingEmails
      })
      .eq('id', userId);

    if (basicProfileError) {
      console.error('Error updating basic profile:', basicProfileError);
      await serviceRoleClient.auth.admin.deleteUser(userId);
      return NextResponse.json(
        { error: 'Failed to update basic profile', details: basicProfileError.message },
        { status: 500 }
      );
    }

    // Validate zip code and update location if provided
    if (validatedData.zipCode) {
      try {
        // Make API call to cities endpoint to validate zip code
        const cityResponse = await fetch(
          `/api/cities?query=${validatedData.zipCode}`
        );
        
        if (!cityResponse.ok) {
          console.error('Failed to validate location:', await cityResponse.text());
          return NextResponse.json(
            { error: 'Failed to validate location' },
            { status: 400 }
          );
        }
        
        const locationData = await cityResponse.json();
        const formattedLocation = `${locationData.name}, ${locationData.adminCode1}`;
        
        // Update profile with location details
        const { error: locationUpdateError } = await serviceRoleClient
          .from('profiles')
          .update({
            location: formattedLocation,
            city: locationData.name,
            state: locationData.adminName1,
            state_code: locationData.adminCode1,
            county: locationData.adminName2,
            zip_code: locationData.postalcode,
            lat: locationData.lat,
            lng: locationData.lng,
            country_code: locationData.countryCode
          })
          .eq('id', userId);

        if (locationUpdateError) {
          console.error('Error updating location details:', locationUpdateError);
          // We don't need to fail the entire registration for location problems
          console.warn('Continuing registration despite location update failure');
        }
      } catch (error) {
        console.error('Error validating zip code:', error);
        // We don't need to fail the entire registration for location problems
        console.warn('Continuing registration despite location validation failure');
      }
    }

    // Update profile with professional details
    const { error: updateError } = await serviceRoleClient
      .from('profiles')
      .update({
        title: validatedData.title,
        company: validatedData.company,
        industry: validatedData.industry,
        experience_level: validatedData.experience,
        bio: validatedData.bio,
        expertise: validatedData.expertise,
        needs: validatedData.needs,
        meaningful_goals: validatedData.meaningfulGoal,
        linkedin_url: validatedData.linkedin,
        github_url: validatedData.github,
        portfolio_url: validatedData.portfolio,
        twitter_url: validatedData.twitter
      })
      .eq('id', userId);

    if (updateError) {
      console.error('Error updating professional details:', updateError);
      await serviceRoleClient.auth.admin.deleteUser(userId);
      return NextResponse.json(
        { error: 'Failed to update professional details', details: updateError.message },
        { status: 500 }
      );
    }

    // Create user preferences
    const { error: preferencesError } = await serviceRoleClient
      .from('user_preferences')
      .insert({
        user_id: userId,
        interests: validatedData.interests || [],
        professional_goals: validatedData.professionalGoals || [],
        values: validatedData.values || [],
        networking_style: validatedData.networkingStyle || []
      });

    if (preferencesError) {
      console.error('Error creating user preferences:', preferencesError);
      // Roll back user creation
      await serviceRoleClient.auth.admin.deleteUser(userId);
      return NextResponse.json(
        { error: 'Failed to create user preferences', details: preferencesError.message },
        { status: 500 }
      );
    }

    // Create user skills if provided
    if (validatedData.skills && validatedData.skills.length > 0) {
      const { error: skillsError } = await serviceRoleClient
        .from('user_skills')
        .insert(
          validatedData.skills.map(skill => ({
            user_id: userId,
            skill_name: skill
          }))
        );

      if (skillsError) {
        console.error('Error creating user skills:', skillsError);
        // Roll back user and preferences creation
        await serviceRoleClient.auth.admin.deleteUser(userId);
        return NextResponse.json(
          { error: 'Failed to create user skills', details: skillsError.message },
          { status: 500 }
        );
      }
    }

    console.log('=== Registration completed successfully ===');
    return NextResponse.json({ 
      message: 'Registration successful',
      user: authData.user
    });
  } catch (error) {
    console.error('Registration process error:', error);
    if (error instanceof z.ZodError) {
      console.error('Validation error:', error.errors);
      return NextResponse.json(
        { error: 'Invalid input data', details: error.errors },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: 'Registration failed', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
} 