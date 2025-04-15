import React, { useState, useEffect } from 'react';
import { Card, Select, Button, Tabs } from '@/components/ui';
import dynamic from 'next/dynamic';
import { useSupabase } from '@/lib/supabase/client';

// Dynamically import Plotly for client-side rendering
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

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

interface LeaderDashboardProps {
  groupId: string;
  groupName: string;
}

export default function LeaderDashboard({ groupId, groupName }: LeaderDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedCategories, setSelectedCategories] = useState<string[]>(['skills']);
  const [members, setMembers] = useState<GroupMember[]>([]);
  const [networkData, setNetworkData] = useState<any>(null);
  const { supabase } = useSupabase();

  // Fetch group members and their data
  useEffect(() => {
    const fetchMembers = async () => {
      const { data, error } = await supabase
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
  }, [groupId, supabase]);

  // Update network visualization when categories or members change
  useEffect(() => {
    if (members.length === 0) return;

    // Calculate similarities and create network data
    const nodes = members.map(member => ({
      id: member.id,
      name: member.name,
      size: 10
    }));

    const edges: any[] = [];
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        const similarity = calculateSimilarity(members[i], members[j], selectedCategories);
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
  }, [members, selectedCategories]);

  const calculateSimilarity = (user1: GroupMember, user2: GroupMember, categories: string[]) => {
    let totalSimilarity = 0;
    
    for (const category of categories) {
      const set1 = new Set(user1[category as keyof GroupMember] as string[]);
      const set2 = new Set(user2[category as keyof GroupMember] as string[]);
      
      if (set1.size === 0 && set2.size === 0) continue;
      
      const intersection = new Set([...set1].filter(x => set2.has(x)));
      const union = new Set([...set1, ...set2]);
      
      totalSimilarity += intersection.size / union.size;
    }
    
    return totalSimilarity / categories.length;
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

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">{groupName} Dashboard</h1>
        <Select
          multiple
          value={selectedCategories}
          onChange={(value) => setSelectedCategories(value as string[])}
          options={[
            { label: 'Skills', value: 'skills' },
            { label: 'Interests', value: 'interests' },
            { label: 'Professional Goals', value: 'professionalGoals' },
            { label: 'Values', value: 'values' }
          ]}
          className="w-64"
        />
      </div>

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        items={[
          { label: 'Overview', value: 'overview' },
          { label: 'Network', value: 'network' },
          { label: 'Members', value: 'members' },
          { label: 'Analytics', value: 'analytics' }
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {activeTab === 'overview' && (
          <>
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
          </>
        )}
      </div>

      {activeTab === 'network' && networkData && (
        <Card className="p-4">
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'markers+text',
                x: networkData.nodes.map((node: any) => node.x),
                y: networkData.nodes.map((node: any) => node.y),
                text: networkData.nodes.map((node: any) => node.name),
                marker: {
                  size: networkData.nodes.map((node: any) => node.size),
                  color: 'lightblue'
                }
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: networkData.edges.flatMap((edge: any) => [
                  networkData.nodes.find((n: any) => n.id === edge.source).x,
                  networkData.nodes.find((n: any) => n.id === edge.target).x,
                  null
                ]),
                y: networkData.edges.flatMap((edge: any) => [
                  networkData.nodes.find((n: any) => n.id === edge.source).y,
                  networkData.nodes.find((n: any) => n.id === edge.target).y,
                  null
                ]),
                line: {
                  width: networkData.edges.map((edge: any) => edge.weight * 3)
                }
              }
            ]}
            layout={{
              title: 'Group Network Visualization',
              showlegend: false,
              hovermode: 'closest',
              margin: { b: 20, l: 5, r: 5, t: 40 },
              xaxis: { showgrid: false, zeroline: false, showticklabels: false },
              yaxis: { showgrid: false, zeroline: false, showticklabels: false },
              width: 800,
              height: 600
            }}
            config={{ responsive: true }}
          />
        </Card>
      )}

      {activeTab === 'members' && (
        <Card className="p-4">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="text-left">Name</th>
                <th className="text-left">Skills</th>
                <th className="text-left">Interests</th>
                <th className="text-left">Last Active</th>
              </tr>
            </thead>
            <tbody>
              {members.map(member => (
                <tr key={member.id}>
                  <td>{member.name}</td>
                  <td>{member.skills.join(', ')}</td>
                  <td>{member.interests.join(', ')}</td>
                  <td>{new Date(member.lastActive).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {activeTab === 'analytics' && (
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
                  )
              }]}
              layout={{
                margin: { t: 20 },
                height: 300
              }}
              config={{ responsive: true }}
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
                  )
              }]}
              layout={{
                margin: { t: 20 },
                height: 300
              }}
              config={{ responsive: true }}
            />
          </Card>
        </div>
      )}
    </div>
  );
} 