'use client';

import { useEffect, useState } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { useDemoUser } from '@/hooks/useDemoUser';

interface Group {
  id: string;
  name: string;
  description: string;
  category: string;
  member_count?: number;
  image_url?: string;
  created_at: string;
  is_private?: boolean;
}

interface GroupMemberCount {
  group_id: string;
  count: number;
}

export default function GroupsPage() {
  const isDemoUser = useDemoUser();
  const [role, setRole] = useState<string | null>(null);
  const [groups, setGroups] = useState<Group[]>([]);
  const [loading, setLoading] = useState(true);
  const supabase = createClientComponentClient();

  // Demo placeholder groups for demo/demo_user roles
  const demoGroups: Group[] = [
    {
      id: '1',
      name: 'Software Engineers Network',
      description: 'A community of software engineers sharing knowledge, job opportunities, and supporting each other in career development.',
      category: 'Technology',
      member_count: 1250,
      image_url: 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3',
      created_at: new Date().toISOString(),
      is_private: false,
    },
    {
      id: '2',
      name: 'Women in Tech',
      description: 'Supporting and promoting women in technical roles through mentorship, resources, and community events.',
      category: 'Professional',
      member_count: 875,
      image_url: 'https://images.unsplash.com/photo-1573164574001-518958d9baa2?ixlib=rb-4.0.3',
      created_at: new Date().toISOString(),
      is_private: false,
    },
    {
      id: '3',
      name: 'Startup Founders Club',
      description: 'Exclusive group for founders to connect, share experiences, and help each other solve challenges in building successful startups.',
      category: 'Entrepreneurship',
      member_count: 456,
      image_url: 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?ixlib=rb-4.0.3',
      created_at: new Date().toISOString(),
      is_private: true,
    },
    {
      id: '4',
      name: 'AI Research Community',
      description: 'Discussing the latest advancements in artificial intelligence and machine learning research.',
      category: 'Technology',
      member_count: 620,
      image_url: 'https://images.unsplash.com/photo-1531297484001-80022131f5a1?ixlib=rb-4.0.3',
      created_at: new Date().toISOString(),
      is_private: false,
    },
  ];

  useEffect(() => {
    const fetchRole = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;
      const { data: profile } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', session.user.id)
        .single();
      setRole(profile?.role || null);
    };
    fetchRole();
  }, [supabase]);

  useEffect(() => {
    const fetchGroups = async () => {
      try {
        // First, get all groups
        const { data: groupsData, error: groupsError } = await supabase
          .from('groups')
          .select('*')
          .order('created_at', { ascending: false });

        if (groupsError) {
          console.error('Error fetching groups:', groupsError);
          // Demo-only fallback for prospect user
          if (isDemoUser) setGroups(demoGroups);
          else setGroups([]);
          return;
        }

        // If we have groups, get member counts for each group
        if (groupsData && groupsData.length > 0) {
          // Get member counts using a direct count query
          const { data: memberCounts, error: countError } = await supabase.rpc('get_group_member_counts');

          if (countError) {
            console.error('Error fetching member counts with RPC:', countError);
            console.log('Falling back to direct count query...');
            // Fallback: Count directly with a query for each group
            const countsPromises = groupsData.map(async (group) => {
              const { count, error } = await supabase
                .from('group_members')
                .select('*', { count: 'exact', head: true })
                .eq('group_id', group.id);
              return { group_id: group.id, count: error ? 0 : (count || 0) };
            });
            const countResults = await Promise.all(countsPromises);
            // Merge the groups with their member counts
            const groupsWithCounts = groupsData.map(group => {
              const countObj = countResults.find((mc: GroupMemberCount) => mc.group_id === group.id);
              return { ...group, member_count: countObj ? countObj.count : 0, is_private: false };
            });
            setGroups(groupsWithCounts);
          } else {
            // Merge the groups with their member counts
            const groupsWithCounts = groupsData.map(group => {
              const countObj = memberCounts?.find((mc: GroupMemberCount) => mc.group_id === group.id);
              return { ...group, member_count: countObj ? countObj.count : 0, is_private: false };
            });
            setGroups(groupsWithCounts);
          }
        } else {
          // No groups: demo fallback only for prospect user
          if (isDemoUser) setGroups(demoGroups);
          else setGroups([]);
        }
      } catch (error) {
        console.error('Error:', error);
        if (isDemoUser) setGroups(demoGroups);
        else setGroups([]);
      } finally {
        setLoading(false);
      }
    };

    fetchGroups();
  }, [supabase, isDemoUser]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold">Professional Groups</h1>
        <div className="flex space-x-2">
          <Link
            href="/discover?tab=groups"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded"
          >
            Discover Groups
          </Link>
          {(role === 'organizer' || role === 'admin') && (
            <Button asChild>
              <Link href="/groups/create">Create Group</Link>
            </Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {groups.map((group) => (
          <div
            key={group.id}
            className="bg-white rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow duration-300"
          >
            {group.image_url && (
              <div className="h-48 overflow-hidden">
                <img
                  src={group.image_url}
                  alt={group.name}
                  className="w-full h-full object-cover"
                />
              </div>
            )}
            <div className="p-6">
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-xl font-bold text-gray-800">{group.name}</h3>
                {group.is_private && (
                  <span className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full">
                    Private
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-600 mb-2">
                <span className="font-medium">Category:</span> {group.category}
              </p>
              <p className="text-sm text-gray-600 mb-4">
                <span className="font-medium">Members:</span> {(group.member_count || 0).toLocaleString()}
              </p>
              <p className="text-gray-700 mb-6 line-clamp-3">{group.description}</p>
              <Link
                href={`/groups/${group.id}`}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded text-sm w-full block text-center"
              >
                View Group
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 