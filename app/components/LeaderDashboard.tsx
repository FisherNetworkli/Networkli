import React, { useState, useEffect } from 'react';
import { Card } from '@/app/components/ui/card';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/app/components/ui/select';
import { Button } from '@/app/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/app/components/ui/tabs';
import dynamic from 'next/dynamic';
import { supabase } from '@/lib/supabase/client';
import { Data, Layout, Config } from 'plotly.js';

// Dynamically import Plotly for client-side rendering
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type CategoryType = 'skills' | 'interests' | 'professionalGoals' | 'values';

interface GroupMember {
  id: string;
  name: string;
  skills: string[];
  interests: string[];
  professionalGoals: string[];
  values: string[];
  joinedAt: string;
  lastActive: string;
}

interface GroupMemberResponse {
  id: string;
  user_id: string;
  joined_at: string;
  users: {
    id: string;
    full_name: string;
    skills: string[];
    interests: string[];
    professional_goals: string[];
    values: string[];
    last_active: string;
  };
}

interface LeaderDashboardProps {
  groupId: string;
  groupName: string;
}

interface NetworkNode {
  id: string;
  name: string;
  size: number;
  x: number;
  y: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
}

interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export default function LeaderDashboard({ groupId, groupName }: LeaderDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedCategory, setSelectedCategory] = useState<CategoryType>('skills');
  const [members, setMembers] = useState<GroupMember[]>([]);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);

  // Fetch group members and their data
  useEffect(() => {
    const fetchMembers = async () => {
      const { data: rawData, error } = await supabase
        .from('group_members')
        .select(`
          id,
          user_id,
          joined_at,
          users (
            id,
            full_name,
            skills,
            interests,
            professional_goals,
            values,
            last_active
          )
        `)
        .eq('group_id', groupId);

      if (error) {
        console.error('Error fetching members:', error);
        return;
      }

      const data = rawData as unknown as GroupMemberResponse[];
      const formattedMembers = data.map(member => ({
        id: member.user_id,
        name: member.users.full_name,
        skills: member.users.skills || [],
        interests: member.users.interests || [],
        professionalGoals: member.users.professional_goals || [],
        values: member.users.values || [],
        joinedAt: member.joined_at,
        lastActive: member.users.last_active
      }));

      setMembers(formattedMembers);
    };

    fetchMembers();
  }, [groupId]);

  // Update network visualization when categories or members change
  useEffect(() => {
    if (members.length === 0) return;

    // Calculate node positions using a simple circular layout
    const radius = 1;
    const nodes = members.map((member, i) => {
      const angle = (2 * Math.PI * i) / members.length;
      return {
        id: member.id,
        name: member.name,
        size: 20,
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle)
      };
    });

    const edges: NetworkEdge[] = [];
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        const similarity = calculateSimilarity(members[i], members[j], selectedCategory);
        if (similarity > 0.2) {
          edges.push({
            source: members[i].id,
            target: members[j].id,
            weight: similarity
          });
        }
      }
    }

    setNetworkData({ nodes, edges });
  }, [members, selectedCategory]);

  const calculateSimilarity = (user1: GroupMember, user2: GroupMember, category: CategoryType) => {
    const set1 = new Set(user1[category] as string[]);
    const set2 = new Set(user2[category] as string[]);
    
    if (set1.size === 0 && set2.size === 0) return 0;
    
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  };

  const getGroupMetrics = () => {
    const totalMembers = members.length;
    const activeMembers = members.filter(m => 
      new Date(m.lastActive) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
    ).length;
    
    const allSkills = new Set(members.flatMap(m => m.skills));
    const allInterests = new Set(members.flatMap(m => m.interests));
    
    return {
      totalMembers,
      activeMembers,
      skillsDiversity: allSkills.size,
      interestsDiversity: allInterests.size
    };
  };

  const categoryOptions = [
    { label: 'Skills', value: 'skills' },
    { label: 'Interests', value: 'interests' },
    { label: 'Professional Goals', value: 'professionalGoals' },
    { label: 'Values', value: 'values' }
  ] as const;

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">{groupName} Dashboard</h1>
        <Select
          value={selectedCategory}
          onValueChange={(value: CategoryType) => setSelectedCategory(value)}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select category" />
          </SelectTrigger>
          <SelectContent>
            {categoryOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Tabs defaultValue={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4">
              <h3 className="text-lg font-semibold">Total Members</h3>
              <p className="text-3xl font-bold">{getGroupMetrics().totalMembers}</p>
            </Card>
            <Card className="p-4">
              <h3 className="text-lg font-semibold">Active Members</h3>
              <p className="text-3xl font-bold">{getGroupMetrics().activeMembers}</p>
            </Card>
            <Card className="p-4">
              <h3 className="text-lg font-semibold">Skills Diversity</h3>
              <p className="text-3xl font-bold">{getGroupMetrics().skillsDiversity}</p>
            </Card>
            <Card className="p-4">
              <h3 className="text-lg font-semibold">Interests Diversity</h3>
              <p className="text-3xl font-bold">{getGroupMetrics().interestsDiversity}</p>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="network">
          {networkData && (
            <Card className="p-4">
              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'text+markers' as const,
                    x: networkData.nodes.map((node) => node.x),
                    y: networkData.nodes.map((node) => node.y),
                    text: networkData.nodes.map((node) => node.name),
                    textposition: 'bottom center' as const,
                    marker: {
                      size: networkData.nodes.map((node) => node.size),
                      color: '#3b82f6'
                    },
                    hoverinfo: 'text' as const
                  } as unknown as Partial<Data>,
                  {
                    type: 'scatter',
                    mode: 'lines' as const,
                    x: networkData.edges.flatMap((edge) => {
                      const sourceNode = networkData.nodes.find((n) => n.id === edge.source);
                      const targetNode = networkData.nodes.find((n) => n.id === edge.target);
                      return sourceNode && targetNode ? [sourceNode.x, targetNode.x, null] : [];
                    }),
                    y: networkData.edges.flatMap((edge) => {
                      const sourceNode = networkData.nodes.find((n) => n.id === edge.source);
                      const targetNode = networkData.nodes.find((n) => n.id === edge.target);
                      return sourceNode && targetNode ? [sourceNode.y, targetNode.y, null] : [];
                    }),
                    line: {
                      color: '#94a3b8',
                      width: 1
                    },
                    hoverinfo: 'none' as const
                  } as unknown as Partial<Data>
                ]}
                layout={{
                  title: 'Member Connections',
                  showlegend: false,
                  hovermode: 'closest' as const,
                  margin: { b: 40, l: 40, r: 40, t: 40 },
                  xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                  yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                  width: 800,
                  height: 600,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)'
                } as Partial<Layout>}
                config={{ responsive: true, displayModeBar: false } as Partial<Config>}
              />
            </Card>
          )}
        </TabsContent>

        <TabsContent value="members">
          <Card className="p-4">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Skills</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Interests</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Active</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {members.map(member => (
                    <tr key={member.id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{member.name}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{member.skills.join(', ')}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{member.interests.join(', ')}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(member.lastActive).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="analytics">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="p-4">
              <h3 className="text-lg font-semibold mb-4">Skills Distribution</h3>
              <Plot
                data={[{
                  type: 'bar',
                  x: Array.from(new Set(members.flatMap(m => m.skills))),
                  y: Array.from(new Set(members.flatMap(m => m.skills)))
                    .map(skill => 
                      members.filter(m => m.skills.includes(skill)).length
                    ),
                  marker: { color: '#3b82f6' }
                } as Partial<Data>]}
                layout={{
                  margin: { t: 20, r: 20 },
                  height: 300,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  xaxis: { tickangle: -45 }
                } as Partial<Layout>}
                config={{ responsive: true, displayModeBar: false } as Partial<Config>}
              />
            </Card>
            <Card className="p-4">
              <h3 className="text-lg font-semibold mb-4">Interests Distribution</h3>
              <Plot
                data={[{
                  type: 'bar',
                  x: Array.from(new Set(members.flatMap(m => m.interests))),
                  y: Array.from(new Set(members.flatMap(m => m.interests)))
                    .map(interest => 
                      members.filter(m => m.interests.includes(interest)).length
                    ),
                  marker: { color: '#3b82f6' }
                } as Partial<Data>]}
                layout={{
                  margin: { t: 20, r: 20 },
                  height: 300,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  xaxis: { tickangle: -45 }
                } as Partial<Layout>}
                config={{ responsive: true, displayModeBar: false } as Partial<Config>}
              />
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
} 