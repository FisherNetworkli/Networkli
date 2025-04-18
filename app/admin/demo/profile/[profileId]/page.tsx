import React from 'react';
import { createAdminClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';
import Image from 'next/image';

export default async function ProfilePage({ params }: { params: { profileId: string }}) {
  const supabaseAdmin = createAdminClient();
  const { data: profile, error } = await supabaseAdmin
    .from('profiles')
    .select('*')
    .eq('id', params.profileId)
    .single();
  if (error || !profile) {
    return <div className="p-6">Profile not found or error: {error?.message}</div>;
  }

  return (
    <div className="max-w-3xl mx-auto py-10 px-4 sm:px-6 lg:px-8">
      <h1 className="text-2xl font-bold mb-6">{profile.first_name} {profile.last_name}</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          {profile.avatar_url && (
            <Image src={profile.avatar_url} alt="Avatar" width={120} height={120} className="rounded-full" />
          )}
          <p><strong>Email:</strong> {profile.email}</p>
          <p><strong>Role:</strong> {profile.role}</p>
          <p><strong>Title:</strong> {profile.title}</p>
          <p><strong>Company:</strong> {profile.company}</p>
          <p><strong>Industry:</strong> {profile.industry}</p>
          <p><strong>Location:</strong> {profile.location}</p>
          <p><strong>Demo:</strong> {profile.is_demo ? 'Yes' : 'No'}</p>
          <p><strong>Celebrity:</strong> {profile.is_celebrity ? 'Yes' : 'No'}</p>
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-2">Bio</h2>
          <p className="mb-4 whitespace-pre-wrap">{profile.bio}</p>
          <h2 className="text-xl font-semibold mb-2">Skills</h2>
          <ul className="list-disc list-inside mb-4">
            {Array.isArray(profile.skills) ? profile.skills.map((s, i) => <li key={i}>{s}</li>) : <li>N/A</li>}
          </ul>
          <h2 className="text-xl font-semibold mb-2">Interests</h2>
          <ul className="list-disc list-inside mb-4">
            {Array.isArray(profile.interests) ? profile.interests.map((i, idx) => <li key={idx}>{i}</li>) : <li>N/A</li>}
          </ul>
          {profile.professional_goals?.length > 0 && (
            <>
              <h2 className="text-xl font-semibold mb-2">Professional Goals</h2>
              <ul className="list-disc list-inside mb-4">
                {profile.professional_goals.map((g, i) => <li key={i}>{g}</li>)}
              </ul>
            </>
          )}
          {profile.values?.length > 0 && (
            <>
              <h2 className="text-xl font-semibold mb-2">Values</h2>
              <ul className="list-disc list-inside">
                {profile.values.map((v, i) => <li key={i}>{v}</li>)}
              </ul>
            </>
          )}
        </div>
      </div>
    </div>
  );
} 