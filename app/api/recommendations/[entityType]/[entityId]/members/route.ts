import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@/app/utils/supabase/server";
import { cookies } from "next/headers";

export async function GET(
  req: NextRequest,
  { params }: { params: { entityType: string; entityId: string } }
) {
  const { entityType, entityId } = params;
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");
  const limit = parseInt(searchParams.get("limit") || "10", 10);
  
  if (!userId) {
    return NextResponse.json(
      { error: "User ID is required" },
      { status: 400 }
    );
  }

  if (!["event", "group"].includes(entityType)) {
    return NextResponse.json(
      { error: "Invalid entity type. Must be 'event' or 'group'" },
      { status: 400 }
    );
  }

  const cookieStore = cookies();
  const supabase = createClient(cookieStore);

  try {
    // First, verify user is part of the event/group
    let isMember = false;
    
    if (entityType === "event") {
      const { data: eventAttendance } = await supabase
        .from("event_attendance")
        .select("*")
        .eq("event_id", entityId)
        .eq("user_id", userId)
        .eq("status", "attending")
        .single();
      
      isMember = !!eventAttendance;
    } else {
      const { data: groupMembership } = await supabase
        .from("group_members")
        .select("*")
        .eq("group_id", entityId)
        .eq("user_id", userId)
        .single();
      
      isMember = !!groupMembership;
    }

    if (!isMember) {
      return NextResponse.json(
        { error: `User is not a member of this ${entityType}` },
        { status: 403 }
      );
    }

    // Fetch user profile to get skills and interests
    const { data: userProfile } = await supabase
      .from("profiles")
      .select("skills, interests")
      .eq("id", userId)
      .single();

    if (!userProfile) {
      return NextResponse.json(
        { error: "User profile not found" },
        { status: 404 }
      );
    }

    // Get other members of the event/group
    let membersQuery;
    if (entityType === "event") {
      membersQuery = supabase
        .from("event_attendance")
        .select(`
          user_id,
          profiles:user_id(
            id, 
            full_name,
            headline,
            avatar_url,
            skills,
            interests
          )
        `)
        .eq("event_id", entityId)
        .eq("status", "attending")
        .neq("user_id", userId);
    } else {
      membersQuery = supabase
        .from("group_members")
        .select(`
          user_id,
          profiles:user_id(
            id, 
            full_name,
            headline,
            avatar_url,
            skills,
            interests
          )
        `)
        .eq("group_id", entityId)
        .neq("user_id", userId);
    }

    const { data: members, error: membersError } = await membersQuery;

    if (membersError) {
      console.error(`Error fetching ${entityType} members:`, membersError);
      return NextResponse.json(
        { error: `Failed to fetch ${entityType} members` },
        { status: 500 }
      );
    }

    // Check existing connections
    const { data: connections } = await supabase
      .from("connections")
      .select("to_user_id")
      .eq("from_user_id", userId);

    const connectedUserIds = new Set(
      connections?.map((c) => c.to_user_id) || []
    );

    // Process and score the recommendations
    const recommendations = members
      .map((member) => {
        const profile = member.profiles;
        if (!profile) return null;

        // Calculate match score based on shared skills and interests
        const userSkills = new Set(userProfile.skills || []);
        const userInterests = new Set(userProfile.interests || []);
        const memberSkills = new Set(profile.skills || []);
        const memberInterests = new Set(profile.interests || []);

        const sharedSkills = [...userSkills].filter(skill => memberSkills.has(skill));
        const sharedInterests = [...userInterests].filter(interest => memberInterests.has(interest));
        
        const maxPossibleMatches = 
          Math.max(userSkills.size, 1) + Math.max(userInterests.size, 1);
        
        const matchScore = 
          (sharedSkills.length + sharedInterests.length) / 
          (maxPossibleMatches || 1);
        
        // Generate match reasons
        const matchReasons = [];
        
        if (sharedSkills.length > 0) {
          matchReasons.push(
            `You both ${sharedSkills.length === 1 ? 'have' : 'share'} the skill: ${sharedSkills.slice(0, 2).join(', ')}${sharedSkills.length > 2 ? ', and more' : ''}`
          );
        }
        
        if (sharedInterests.length > 0) {
          matchReasons.push(
            `You both ${sharedInterests.length === 1 ? 'are interested in' : 'share interests in'}: ${sharedInterests.slice(0, 2).join(', ')}${sharedInterests.length > 2 ? ', and more' : ''}`
          );
        }
        
        if (matchReasons.length === 0) {
          matchReasons.push(`You're both part of the same ${entityType === "event" ? "event" : "group"}`);
        }

        return {
          id: profile.id,
          name: profile.full_name,
          headline: profile.headline,
          avatar_url: profile.avatar_url,
          skills: profile.skills,
          interests: profile.interests,
          match_score: matchScore,
          match_reasons: matchReasons,
          already_connected: connectedUserIds.has(profile.id)
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.match_score - a.match_score)
      .slice(0, limit);

    return NextResponse.json({
      recommendations,
      total: recommendations.length,
      entity_type: entityType,
      entity_id: entityId
    });
  } catch (error) {
    console.error("Error generating recommendations:", error);
    return NextResponse.json(
      { error: "Failed to generate recommendations" },
      { status: 500 }
    );
  }
} 