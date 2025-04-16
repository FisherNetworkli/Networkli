import { NextResponse } from 'next/server';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

interface DatabaseInfo {
  total_size: number;
  table_count: number;
  index_count: number;
}

interface StorageBucket {
  name: string;
  size: number;
}

export async function GET() {
  try {
    const supabase = createServerComponentClient({ cookies });
    
    // Check if user is authenticated
    const { data: { session } } = await supabase.auth.getSession();
    
    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }
    
    // Check if user is admin by querying the profiles table
    const { data: profile } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', session.user.id)
      .single();
    
    if (!profile || profile.role !== 'admin') {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Fetch database statistics
    const [
      { data: databaseInfo, error: databaseError },
      { data: storageBuckets, error: storageError },
      { count: apiRequests, error: apiError },
      { count: totalUsers, error: usersError },
      { count: recentUsers, error: recentUsersError },
    ] = await Promise.all([
      // Get database size and statistics
      supabase.rpc('get_database_info'),
      
      // Get storage buckets info
      supabase.storage.listBuckets(),
      
      // Get API request count (using auth.users as a proxy for activity)
      supabase.from('profiles').select('*', { count: 'exact', head: true }),
      
      // Get total user count
      supabase.from('profiles').select('*', { count: 'exact', head: true }),
      
      // Get recent user count (users in the last 30 days)
      supabase.from('profiles')
        .select('*', { count: 'exact', head: true })
        .gt('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
    ]);

    if (databaseError || storageError || apiError || usersError || recentUsersError) {
      const errors = [
        databaseError && `Database: ${databaseError.message}`,
        storageError && `Storage: ${storageError.message}`,
        apiError && `API: ${apiError.message}`,
        usersError && `Users: ${usersError.message}`,
        recentUsersError && `Recent Users: ${recentUsersError.message}`,
      ].filter(Boolean);
      
      console.error('Error fetching system metrics:', errors);
      
      // Even if some metrics fail, return what we have
    }

    // Calculate storage usage
    let totalStorageSize = 0;
    let storageDetails: StorageBucket[] = [];
    
    if (storageBuckets) {
      // For each bucket, we'd ideally get size info
      // This is simplified as direct bucket size is not easily available
      // In a real implementation, you'd track this with a separate function
      storageDetails = storageBuckets.map(bucket => ({
        name: bucket.name,
        // This would need a proper implementation
        size: Math.floor(Math.random() * 1000000) // Placeholder
      }));
      
      totalStorageSize = storageDetails.reduce((sum, bucket) => sum + bucket.size, 0);
    }

    // Database status check (simplified)
    const databaseStatus = databaseError ? 'Error' : 'Healthy';
    
    // Create performance metrics
    const performanceMetrics = {
      responseTime: {
        avg: Math.floor(Math.random() * 100) + 50, // Placeholder 50-150ms
        p95: Math.floor(Math.random() * 200) + 100, // Placeholder 100-300ms
        p99: Math.floor(Math.random() * 300) + 200  // Placeholder 200-500ms
      },
      errorRate: Math.random() * 2, // Placeholder 0-2%
      successRate: 98 + Math.random() * 2 // Placeholder 98-100%
    };

    // Ensure totalUsers has a default value if null
    const userCount = totalUsers || 0;
    const activeUsers = recentUsers || 0;

    return NextResponse.json({
      database: {
        status: databaseStatus,
        size: databaseInfo?.total_size || 0,
        tables: databaseInfo?.table_count || 0,
        indexes: databaseInfo?.index_count || 0
      },
      storage: {
        totalSize: totalStorageSize,
        buckets: storageDetails,
        usagePercentage: Math.min(totalStorageSize / (20 * 1024 * 1024), 100) // Assuming 20MB limit
      },
      api: {
        totalRequests: apiRequests || 0,
        requestsToday: Math.floor((apiRequests || 0) * Math.random() * 0.1), // Placeholder 
        requestLimit: 100000,
        usagePercentage: ((apiRequests || 0) / 100000) * 100
      },
      users: {
        total: userCount,
        active: activeUsers,
        growth: activeUsers && userCount > activeUsers ? (activeUsers / (userCount - activeUsers)) * 100 : 0
      },
      performance: performanceMetrics,
      lastUpdated: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in monitoring API:', error);
    return NextResponse.json({ 
      error: 'Failed to fetch monitoring data', 
      details: error instanceof Error ? error.message : String(error) 
    }, { status: 500 });
  }
} 