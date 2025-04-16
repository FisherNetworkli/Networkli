import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';

// Define valid interaction types
const VALID_INTERACTION_TYPES = [
  'PROFILE_VIEW',
  'CONNECT_REQUEST',
  'SWIPE_LEFT',
  'SWIPE_RIGHT',
  'MESSAGE_SENT',
  'RECOMMENDATION_CLICK',
  'NETWORK_INTENT'
];

// Define valid entity types
const VALID_ENTITY_TYPES = [
  'PROFILE',
  'CONNECTION',
  'MESSAGE',
  'RECOMMENDATION'
];

export async function POST(request: NextRequest) {
  const cookieStore = cookies();
  const supabase = createRouteHandlerClient({ cookies: () => cookieStore });
  
  try {
    // Authenticate user
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    
    if (sessionError || !session) {
      console.error('Authentication error:', sessionError);
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }
    
    // Parse request body
    const body = await request.json();
    const { 
      interaction_type, 
      target_entity_type, 
      target_entity_id,
      metadata = {}
    } = body;
    
    // Validate required fields
    if (!interaction_type || !target_entity_type || !target_entity_id) {
      return NextResponse.json(
        { error: 'Missing required fields: interaction_type, target_entity_type, target_entity_id' },
        { status: 400 }
      );
    }
    
    // Validate interaction type
    if (!VALID_INTERACTION_TYPES.includes(interaction_type)) {
      return NextResponse.json(
        { error: `Invalid interaction_type. Must be one of: ${VALID_INTERACTION_TYPES.join(', ')}` },
        { status: 400 }
      );
    }
    
    // Validate entity type
    if (!VALID_ENTITY_TYPES.includes(target_entity_type)) {
      return NextResponse.json(
        { error: `Invalid target_entity_type. Must be one of: ${VALID_ENTITY_TYPES.join(', ')}` },
        { status: 400 }
      );
    }
    
    // Check if the interaction_history table exists
    const { data: tableExists } = await supabase
      .from('information_schema.tables')
      .select('*')
      .eq('table_schema', 'public')
      .eq('table_name', 'interaction_history')
      .single();
    
    // Create the table if it doesn't exist
    if (!tableExists) {
      const { error: createTableError } = await supabase.rpc('create_interaction_history_table');
      if (createTableError) {
        console.error('Error creating interaction_history table:', createTableError);
        return NextResponse.json(
          { error: 'Could not create interaction tracking table' },
          { status: 500 }
        );
      }
    }
    
    // Record the interaction
    const { data, error } = await supabase
      .from('interaction_history')
      .insert({
        user_id: session.user.id,
        interaction_type,
        target_entity_type,
        target_entity_id,
        metadata
      })
      .select('id, created_at');
    
    if (error) {
      console.error('Error recording interaction:', error);
      return NextResponse.json(
        { error: 'Failed to record interaction' },
        { status: 500 }
      );
    }
    
    // Handle specific interaction types with additional actions
    if (interaction_type === 'SWIPE_RIGHT' && target_entity_type === 'PROFILE') {
      // Create a connection request when user swipes right
      const { error: connectionError } = await supabase
        .from('connections')
        .insert({
          requester_id: session.user.id,
          receiver_id: target_entity_id,
          status: 'pending',
          initiated_via: 'swipe',
          message: 'I found your professional profile interesting and would like to connect.',
          connection_purpose: 'networking'
        });
      
      if (connectionError) {
        console.error('Error creating connection request after swipe right:', connectionError);
        // We still return success for the interaction, but log the connection error
      } else {
        // Log an additional action for analytics about networking intent
        await supabase
          .from('interaction_history')
          .insert({
            user_id: session.user.id,
            interaction_type: 'NETWORK_INTENT',
            target_entity_type: 'PROFILE',
            target_entity_id: target_entity_id,
            metadata: {
              context: 'professional_networking',
              initiated_via: 'swipe',
              connection_requested: true
            }
          });
      }
    }
    
    return NextResponse.json({
      success: true,
      message: 'Interaction recorded successfully',
      interaction_id: data?.[0]?.id,
      timestamp: data?.[0]?.created_at
    });
    
  } catch (error) {
    console.error('Error in interaction API:', error);
    return NextResponse.json(
      { error: 'An unexpected error occurred' },
      { status: 500 }
    );
  }
} 