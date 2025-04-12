import { createClient } from '@/lib/supabase/server';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { hash } from 'bcryptjs';

export async function POST(request: Request) {
  try {
    const supabase = createClient();
    const json = await request.json();
    const {
      email,
      password,
      firstName,
      lastName,
      title,
      company,
      industry,
      experience,
      skills,
      bio,
      interests,
      lookingFor,
      preferredIndustries,
      preferredRoles,
      linkedin,
      github,
      portfolio,
      twitter,
      profileVisibility,
      emailNotifications,
      marketingEmails,
    } = json;

    // Hash the password
    const hashedPassword = await hash(password, 10);

    // Create the user in Supabase Auth
    const { data: authData, error: authError } = await supabase.auth.signUp({
      email,
      password: hashedPassword,
      options: {
        data: {
          firstName,
          lastName,
          profileVisibility,
          emailNotifications,
          marketingEmails,
        },
      },
    });

    if (authError) {
      return NextResponse.json(
        { error: authError.message },
        { status: 400 }
      );
    }

    if (!authData.user) {
      return NextResponse.json(
        { error: 'User creation failed' },
        { status: 400 }
      );
    }

    // Create user preferences
    const { error: preferencesError } = await supabase
      .from('user_preferences')
      .insert({
        user_id: authData.user.id,
        interests,
        looking_for: lookingFor,
        preferred_industries: preferredIndustries,
        preferred_roles: preferredRoles,
      });

    if (preferencesError) {
      // Rollback user creation
      await supabase.auth.admin.deleteUser(authData.user.id);
      return NextResponse.json(
        { error: 'Failed to create user preferences' },
        { status: 400 }
      );
    }

    // Update professional info in profiles table
    const { error: profileError } = await supabase
      .from('profiles')
      .update({
        title,
        company,
        industry,
        experience_level: experience,
        bio,
        linkedin_url: linkedin,
        github_url: github,
        portfolio_url: portfolio,
        twitter_url: twitter,
      })
      .eq('id', authData.user.id);

    if (profileError) {
      // Rollback user creation and preferences
      await supabase.auth.admin.deleteUser(authData.user.id);
      return NextResponse.json(
        { error: 'Failed to update profile information' },
        { status: 400 }
      );
    }

    // Add user skills
    if (skills && skills.length > 0 && authData.user?.id) {
      const skillInserts = skills.map((skill: string) => ({
        profile_id: authData.user?.id,
        skill_id: skill,
        level: 'intermediate', // Default level, can be customized later
        years_of_experience: 0, // Default value, can be customized later
      }));

      const { error: skillsError } = await supabase
        .from('user_skills')
        .insert(skillInserts);

      if (skillsError) {
        // Log the error but don't rollback (skills can be added later)
        console.error('Failed to add user skills:', skillsError);
      }
    }

    return NextResponse.json({
      user: authData.user,
      message: 'User created successfully',
    });
  } catch (error) {
    console.error('Signup error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 