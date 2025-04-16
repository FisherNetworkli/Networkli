'use client';

import { useState, useEffect } from 'react';
import { User } from '@supabase/supabase-js';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

interface AdminDashboardProps {
  user: User;
}

interface SystemStats {
  totalUsers: number;
  totalGroups: number;
  totalEvents: number;
  activeUsers: number;
  totalBlogPosts: number;
  unreadMessages: number;
  pendingApplications: number;
}

interface UserListItem {
  id: string;
  email: string;
  full_name: string;
  role: string;
  created_at: string;
}

interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  published: boolean;
  author: string;
  date: string;
}

interface ContactMessage {
  id: string;
  name: string;
  email: string;
  subject: string;
  status: 'UNREAD' | 'READ' | 'REPLIED';
  created_at: string;
}

interface JobApplication {
  id: string;
  name: string;
  position: string;
  status: 'PENDING' | 'REVIEWING' | 'ACCEPTED' | 'REJECTED';
  created_at: string;
}

export function AdminDashboard({ user }: AdminDashboardProps) {
  const [stats, setStats] = useState<SystemStats>({
    totalUsers: 0,
    totalGroups: 0,
    totalEvents: 0,
    activeUsers: 0,
    totalBlogPosts: 0,
    unreadMessages: 0,
    pendingApplications: 0,
  });
  const [users, setUsers] = useState<UserListItem[]>([]);
  const [blogPosts, setBlogPosts] = useState<BlogPost[]>([]);
  const [contactMessages, setContactMessages] = useState<ContactMessage[]>([]);
  const [applications, setApplications] = useState<JobApplication[]>([]);
  const [loading, setLoading] = useState(true);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const fetchData = async () => {
      // Fetch system stats
      const [
        { count: userCount },
        // { count: groupCount }, // Commented out - groups table does not exist
        // { count: eventCount }, // Commented out - events table does not exist
        { count: activeCount },
        { count: blogCount },
        { count: unreadCount },
        { count: pendingCount },
      ] = await Promise.all([
        supabase.from('profiles').select('*', { count: 'exact', head: true }),
        // supabase.from('groups').select('*', { count: 'exact', head: true }), // Commented out
        // supabase.from('events').select('*', { count: 'exact', head: true }), // Commented out
        supabase.from('profiles').select('*', { count: 'exact', head: true }), // Assuming active is just total for now
        supabase.from('blog_posts').select('*', { count: 'exact', head: true }),
        supabase.from('contact_submissions')
          .select('*', { count: 'exact', head: true })
          .eq('status', 'UNREAD'),
        supabase.from('job_applications')
          .select('*', { count: 'exact', head: true })
          .eq('status', 'PENDING'),
      ]);

      setStats({
        totalUsers: userCount || 0,
        // totalGroups: groupCount || 0, // Commented out
        // totalEvents: eventCount || 0, // Commented out
        totalGroups: 0, // Set to 0 as placeholder
        totalEvents: 0, // Set to 0 as placeholder
        activeUsers: activeCount || 0,
        totalBlogPosts: blogCount || 0,
        unreadMessages: unreadCount || 0,
        pendingApplications: pendingCount || 0,
      });

      // Fetch recent users
      const { data: usersData } = await supabase
        .from('profiles')
        .select('id, email, full_name, role, created_at')
        .order('created_at', { ascending: false })
        .limit(10);

      if (usersData) {
        setUsers(usersData);
      }

      // Fetch recent blog posts
      const { data: blogData } = await supabase
        .from('blog_posts')
        .select('id, title, excerpt, published, author, date')
        .order('date', { ascending: false })
        .limit(10);

      if (blogData) {
        setBlogPosts(blogData);
      }

      // Fetch recent contact messages
      const { data: messagesData } = await supabase
        .from('contact_submissions')
        .select('id, name, email, subject, status, created_at')
        .order('created_at', { ascending: false })
        .limit(10);

      if (messagesData) {
        setContactMessages(messagesData as ContactMessage[]);
      }

      // Fetch recent job applications
      const { data: applicationsData } = await supabase
        .from('job_applications')
        .select('id, name, position, status, created_at')
        .order('created_at', { ascending: false })
        .limit(10);

      if (applicationsData) {
        setApplications(applicationsData as unknown as JobApplication[]);
      }

      setLoading(false);
    };

    fetchData();
  }, [supabase]);

  const getStatusColor = (status: string) => {
    const colors = {
      UNREAD: 'bg-blue-100 text-blue-800',
      READ: 'bg-gray-100 text-gray-800',
      REPLIED: 'bg-green-100 text-green-800',
      PENDING: 'bg-yellow-100 text-yellow-800',
      REVIEWING: 'bg-blue-100 text-blue-800',
      ACCEPTED: 'bg-green-100 text-green-800',
      REJECTED: 'bg-red-100 text-red-800',
    };
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Admin Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Monitor and manage your platform.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalUsers}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Blog Posts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalBlogPosts}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unread Messages</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.unreadMessages}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Applications</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.pendingApplications}</div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="users" className="space-y-4">
        <TabsList>
          <TabsTrigger value="users">Users</TabsTrigger>
          <TabsTrigger value="blog">Blog Posts</TabsTrigger>
          <TabsTrigger value="messages">Messages</TabsTrigger>
          <TabsTrigger value="applications">Applications</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="users" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Users</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Joined</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {users.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell>{user.full_name}</TableCell>
                      <TableCell>{user.email}</TableCell>
                      <TableCell className="capitalize">{user.role}</TableCell>
                      <TableCell>{new Date(user.created_at).toLocaleDateString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="blog" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Blog Posts</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Title</TableHead>
                    <TableHead>Author</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {blogPosts.map((post) => (
                    <TableRow key={post.id}>
                      <TableCell>{post.title}</TableCell>
                      <TableCell>{post.author}</TableCell>
                      <TableCell>
                        <Badge variant={post.published ? "default" : "secondary"}>
                          {post.published ? 'Published' : 'Draft'}
                        </Badge>
                      </TableCell>
                      <TableCell>{new Date(post.date).toLocaleDateString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="messages" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Messages</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Subject</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {contactMessages.map((message) => (
                    <TableRow key={message.id}>
                      <TableCell>{message.name}</TableCell>
                      <TableCell>{message.subject}</TableCell>
                      <TableCell>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(message.status)}`}>
                          {message.status}
                        </span>
                      </TableCell>
                      <TableCell>{new Date(message.created_at).toLocaleDateString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="applications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Applications</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Applicant</TableHead>
                    <TableHead>Position</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {applications.map((application) => (
                    <TableRow key={application.id}>
                      <TableCell>{application.name}</TableCell>
                      <TableCell>{application.position}</TableCell>
                      <TableCell>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(application.status)}`}>
                          {application.status}
                        </span>
                      </TableCell>
                      <TableCell>{new Date(application.created_at).toLocaleDateString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings">
          <Card>
            <CardHeader>
              <CardTitle>Admin Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <p>System settings and configurations will be available here.</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 