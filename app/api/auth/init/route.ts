import { NextResponse } from 'next/server'
import { createAdminClient } from '@/utils/supabase/server'

export async function GET() {
  try {
    const supabaseAdmin = createAdminClient()
    // Initialize auth schema
    const initQueries = [
      // Create auth schema if it doesn't exist
      `CREATE SCHEMA IF NOT EXISTS auth;`,
      
      // Create users table
      `CREATE TABLE IF NOT EXISTS auth.users (
        id uuid NOT NULL PRIMARY KEY,
        instance_id uuid,
        aud character varying(255),
        role character varying(255),
        email character varying(255),
        encrypted_password character varying(255),
        email_confirmed_at timestamp with time zone,
        invited_at timestamp with time zone,
        confirmation_token character varying(255),
        confirmation_sent_at timestamp with time zone,
        recovery_token character varying(255),
        recovery_sent_at timestamp with time zone,
        email_change_token_new character varying(255),
        email_change character varying(255),
        email_change_sent_at timestamp with time zone,
        last_sign_in_at timestamp with time zone,
        raw_app_meta_data jsonb,
        raw_user_meta_data jsonb,
        is_super_admin boolean,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        phone character varying(255) DEFAULT NULL::character varying,
        phone_confirmed_at timestamp with time zone,
        phone_change character varying(255) DEFAULT ''::character varying,
        phone_change_token character varying(255) DEFAULT ''::character varying,
        phone_change_sent_at timestamp with time zone,
        confirmed_at timestamp with time zone GENERATED ALWAYS AS (LEAST(email_confirmed_at, phone_confirmed_at)) STORED,
        email_change_token_current character varying(255) DEFAULT ''::character varying,
        email_change_confirm_status smallint DEFAULT 0,
        banned_until timestamp with time zone,
        reauthentication_token character varying(255) DEFAULT ''::character varying,
        reauthentication_sent_at timestamp with time zone,
        is_sso_user boolean DEFAULT false,
        deleted_at timestamp with time zone
      );`,

      // Create identities table
      `CREATE TABLE IF NOT EXISTS auth.identities (
        id text NOT NULL,
        user_id uuid NOT NULL,
        identity_data jsonb NOT NULL,
        provider text NOT NULL,
        last_sign_in_at timestamp with time zone,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        email text GENERATED ALWAYS AS (lower(identity_data->>'email')) STORED,
        CONSTRAINT identities_pkey PRIMARY KEY (provider, id),
        CONSTRAINT identities_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE
      );`,

      // Create instances table
      `CREATE TABLE IF NOT EXISTS auth.instances (
        id uuid NOT NULL,
        uuid uuid,
        raw_base_config text,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        CONSTRAINT instances_pkey PRIMARY KEY (id)
      );`,

      // Create refresh tokens table
      `CREATE TABLE IF NOT EXISTS auth.refresh_tokens (
        instance_id uuid,
        id bigint NOT NULL,
        token character varying(255),
        user_id character varying(255),
        revoked boolean,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        parent character varying(255),
        session_id uuid,
        CONSTRAINT refresh_tokens_pkey PRIMARY KEY (id)
      );`,

      // Create sessions table
      `CREATE TABLE IF NOT EXISTS auth.sessions (
        id uuid NOT NULL,
        user_id uuid NOT NULL,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        factor_id uuid,
        aal character varying(255),
        not_after timestamp with time zone,
        CONSTRAINT sessions_pkey PRIMARY KEY (id),
        CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE
      );`,

      // Create mfa factors table
      `CREATE TABLE IF NOT EXISTS auth.mfa_factors (
        id uuid NOT NULL,
        user_id uuid NOT NULL,
        friendly_name text,
        factor_type auth.factor_type NOT NULL,
        status auth.factor_status NOT NULL,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        secret text,
        CONSTRAINT mfa_factors_pkey PRIMARY KEY (id),
        CONSTRAINT mfa_factors_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE
      );`,

      // Create mfa challenges table
      `CREATE TABLE IF NOT EXISTS auth.mfa_challenges (
        id uuid NOT NULL,
        factor_id uuid NOT NULL,
        created_at timestamp with time zone,
        verified_at timestamp with time zone,
        ip_address inet,
        CONSTRAINT mfa_challenges_pkey PRIMARY KEY (id),
        CONSTRAINT mfa_challenges_factor_id_fkey FOREIGN KEY (factor_id) REFERENCES auth.mfa_factors(id) ON DELETE CASCADE
      );`,

      // Create mfa amr claims table
      `CREATE TABLE IF NOT EXISTS auth.mfa_amr_claims (
        id uuid NOT NULL,
        session_id uuid NOT NULL,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        authentication_method text NOT NULL,
        CONSTRAINT mfa_amr_claims_pkey PRIMARY KEY (id),
        CONSTRAINT mfa_amr_claims_session_id_fkey FOREIGN KEY (session_id) REFERENCES auth.sessions(id) ON DELETE CASCADE
      );`,

      // Create sso domains table
      `CREATE TABLE IF NOT EXISTS auth.sso_domains (
        id uuid NOT NULL,
        sso_provider_id uuid NOT NULL,
        domain text NOT NULL,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        CONSTRAINT sso_domains_pkey PRIMARY KEY (id)
      );`,

      // Create sso providers table
      `CREATE TABLE IF NOT EXISTS auth.sso_providers (
        id uuid NOT NULL,
        resource_id text,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        CONSTRAINT sso_providers_pkey PRIMARY KEY (id)
      );`,

      // Create saml providers table
      `CREATE TABLE IF NOT EXISTS auth.saml_providers (
        id uuid NOT NULL,
        sso_provider_id uuid NOT NULL,
        entity_id text NOT NULL,
        metadata_xml text NOT NULL,
        metadata_url text,
        attribute_mapping jsonb,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        CONSTRAINT saml_providers_pkey PRIMARY KEY (id),
        CONSTRAINT saml_providers_sso_provider_id_fkey FOREIGN KEY (sso_provider_id) REFERENCES auth.sso_providers(id) ON DELETE CASCADE
      );`,

      // Create saml relay states table
      `CREATE TABLE IF NOT EXISTS auth.saml_relay_states (
        id uuid NOT NULL,
        sso_provider_id uuid NOT NULL,
        request_id text NOT NULL,
        for_email text,
        redirect_to text,
        from_ip_address inet,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        CONSTRAINT saml_relay_states_pkey PRIMARY KEY (id),
        CONSTRAINT saml_relay_states_sso_provider_id_fkey FOREIGN KEY (sso_provider_id) REFERENCES auth.sso_providers(id) ON DELETE CASCADE
      );`,

      // Create sso providers table
      `CREATE TABLE IF NOT EXISTS auth.flow_state (
        id uuid NOT NULL,
        user_id uuid,
        auth_code text NOT NULL,
        code_challenge_method auth.code_challenge_method NOT NULL,
        code_challenge text NOT NULL,
        provider_type text NOT NULL,
        provider_access_token text,
        provider_refresh_token text,
        created_at timestamp with time zone,
        updated_at timestamp with time zone,
        authentication_method text NOT NULL,
        CONSTRAINT flow_state_pkey PRIMARY KEY (id),
        CONSTRAINT flow_state_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE
      );`,

      // Grant necessary permissions
      `GRANT USAGE ON SCHEMA auth TO postgres, anon, authenticated, service_role;`,
      `GRANT ALL ON ALL TABLES IN SCHEMA auth TO postgres, service_role;`,
      `GRANT SELECT ON ALL TABLES IN SCHEMA auth TO anon, authenticated;`,
      
      // Create necessary types if they don't exist
      `DO $$ BEGIN
        CREATE TYPE auth.factor_type AS ENUM ('totp', 'webauthn');
        EXCEPTION
        WHEN duplicate_object THEN null;
      END $$;`,

      `DO $$ BEGIN
        CREATE TYPE auth.factor_status AS ENUM ('unverified', 'verified');
        EXCEPTION
        WHEN duplicate_object THEN null;
      END $$;`,

      `DO $$ BEGIN
        CREATE TYPE auth.code_challenge_method AS ENUM ('s256', 'plain');
        EXCEPTION
        WHEN duplicate_object THEN null;
      END $$;`
    ];

    // Execute all queries
    for (const query of initQueries) {
      const { error } = await supabaseAdmin?.rpc('exec', { query }) || {};
      if (error) throw error;
    }

    return NextResponse.json({ status: 'success', message: 'Auth schema initialized' });
  } catch (error) {
    console.error('Auth init error:', error);
    return NextResponse.json({
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
} 