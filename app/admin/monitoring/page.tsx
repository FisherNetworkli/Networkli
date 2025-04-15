'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { HomeIcon, ArrowPathIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface MonitoringData {
  database: {
    status: string;
    size: number;
    tables: number;
    indexes: number;
  };
  storage: {
    totalSize: number;
    buckets: Array<{ name: string; size: number }>;
    usagePercentage: number;
  };
  api: {
    totalRequests: number;
    requestsToday: number;
    requestLimit: number;
    usagePercentage: number;
  };
  users: {
    total: number;
    active: number;
    growth: number;
  };
  performance: {
    responseTime: {
      avg: number;
      p95: number;
      p99: number;
    };
    errorRate: number;
    successRate: number;
  };
  lastUpdated: string;
}

export default function MonitoringPage() {
  const [metrics, setMetrics] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchMetrics = async () => {
    try {
      setRefreshing(true);
      const response = await fetch('/api/monitoring');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch monitoring data: ${response.status}`);
      }
      
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching monitoring data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch monitoring data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    
    // Set up auto-refresh every 30 seconds
    const intervalId = setInterval(fetchMetrics, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  // Helper function to format bytes
  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'bg-green-100 text-green-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getUsageColor = (percentage: number) => {
    if (percentage < 50) return 'bg-green-500';
    if (percentage < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="py-6">
      <div className="px-4 sm:px-6 md:px-8">
        {/* Header with back button */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold text-gray-900">System Monitoring</h1>
          <div className="flex space-x-3">
            <button
              onClick={fetchMetrics}
              disabled={refreshing}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue disabled:opacity-50"
            >
              <ArrowPathIcon className={`-ml-0.5 mr-2 h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <Link
              href="/admin"
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark"
            >
              <HomeIcon className="h-4 w-4 mr-2" />
              Back to Dashboard
            </Link>
          </div>
        </div>

        {error && (
          <div className="mb-6 bg-red-50 border-l-4 border-red-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <ExclamationTriangleIcon className="h-5 w-5 text-red-400" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {loading && !metrics ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-connection-blue"></div>
          </div>
        ) : (
          <>
            {/* Last updated info */}
            {metrics && (
              <p className="text-sm text-gray-500 mb-6">
                Last updated: {new Date(metrics.lastUpdated).toLocaleString()}
              </p>
            )}

            {/* Status overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {/* Database status */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 bg-connection-blue/10 rounded-md p-3">
                      <svg className="h-6 w-6 text-connection-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z" />
                        <path strokeLinecap="round" strokeWidth={2} d="M9 3v4M15 3v4M9 13h6M9 17h4" />
                      </svg>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">Database</dt>
                        <dd className="flex items-center">
                          <div className="text-lg font-medium text-gray-900">{metrics?.database.tables || 0} Tables</div>
                          <span className={`ml-2 px-2 py-1 text-xs rounded-full ${getStatusColor(metrics?.database.status || 'Unknown')}`}>
                            {metrics?.database.status || 'Unknown'}
                          </span>
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-4 py-4 sm:px-6">
                  <div className="text-sm">
                    <span className="font-medium text-gray-500">Size:</span>{' '}
                    <span className="font-medium text-gray-900">{formatBytes(metrics?.database.size || 0)}</span>
                  </div>
                </div>
              </div>

              {/* Storage usage */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 bg-connection-blue/10 rounded-md p-3">
                      <svg className="h-6 w-6 text-connection-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                      </svg>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">Storage</dt>
                        <dd>
                          <div className="text-lg font-medium text-gray-900">
                            {formatBytes(metrics?.storage.totalSize || 0)}
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                            <div
                              className={`h-2.5 rounded-full ${getUsageColor(metrics?.storage.usagePercentage || 0)}`}
                              style={{ width: `${metrics?.storage.usagePercentage || 0}%` }}
                            ></div>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {Math.round(metrics?.storage.usagePercentage || 0)}% of quota
                          </div>
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-4 py-4 sm:px-6">
                  <div className="text-sm">
                    <span className="font-medium text-gray-500">Buckets:</span>{' '}
                    <span className="font-medium text-gray-900">{metrics?.storage.buckets.length || 0}</span>
                  </div>
                </div>
              </div>

              {/* API Usage */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 bg-connection-blue/10 rounded-md p-3">
                      <svg className="h-6 w-6 text-connection-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">API Requests</dt>
                        <dd>
                          <div className="text-lg font-medium text-gray-900">
                            {metrics?.api.totalRequests.toLocaleString() || 0}
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                            <div
                              className={`h-2.5 rounded-full ${getUsageColor(metrics?.api.usagePercentage || 0)}`}
                              style={{ width: `${metrics?.api.usagePercentage || 0}%` }}
                            ></div>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {Math.round(metrics?.api.usagePercentage || 0)}% of limit
                          </div>
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-4 py-4 sm:px-6">
                  <div className="text-sm">
                    <span className="font-medium text-gray-500">Today:</span>{' '}
                    <span className="font-medium text-gray-900">{metrics?.api.requestsToday.toLocaleString() || 0}</span>
                  </div>
                </div>
              </div>

              {/* User Stats */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 bg-connection-blue/10 rounded-md p-3">
                      <svg className="h-6 w-6 text-connection-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                      </svg>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">Users</dt>
                        <dd>
                          <div className="text-lg font-medium text-gray-900">
                            {metrics?.users.total.toLocaleString() || 0} Total
                          </div>
                          <div className="text-sm text-gray-500">
                            {metrics?.users.active.toLocaleString() || 0} Active (30d)
                          </div>
                          <div className="text-xs text-connection-blue font-medium mt-1">
                            {(metrics?.users?.growth || 0) > 0 ? '+' : ''}{Math.round(metrics?.users?.growth || 0)}% Growth
                          </div>
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-4 py-4 sm:px-6">
                  <div className="text-sm">
                    <span className="font-medium text-gray-500">Active rate:</span>{' '}
                    <span className="font-medium text-gray-900">
                      {metrics?.users.total ? Math.round((metrics.users.active / metrics.users.total) * 100) : 0}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white shadow overflow-hidden sm:rounded-md mb-8">
              <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Performance Metrics
                </h3>
              </div>
              <div className="px-4 py-5 sm:p-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Response Times */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-500">Response Times</h4>
                    <dl className="mt-3 space-y-2">
                      <div className="flex justify-between">
                        <dt className="text-sm text-gray-500">Average</dt>
                        <dd className="text-sm font-medium text-gray-900">{metrics?.performance.responseTime.avg || 0} ms</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="text-sm text-gray-500">P95</dt>
                        <dd className="text-sm font-medium text-gray-900">{metrics?.performance.responseTime.p95 || 0} ms</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="text-sm text-gray-500">P99</dt>
                        <dd className="text-sm font-medium text-gray-900">{metrics?.performance.responseTime.p99 || 0} ms</dd>
                      </div>
                    </dl>
                  </div>

                  {/* Success Rate */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-500">Success Rate</h4>
                    <div className="mt-4">
                      <div className="relative pt-1">
                        <div className="text-center text-lg font-bold text-gray-900">
                          {metrics ? metrics.performance.successRate.toFixed(2) : 0}%
                        </div>
                        <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
                          <div 
                            style={{ width: `${metrics?.performance.successRate || 0}%` }} 
                            className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-green-500">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Error Rate */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-500">Error Rate</h4>
                    <div className="mt-4">
                      <div className="relative pt-1">
                        <div className="text-center text-lg font-bold text-gray-900">
                          {metrics ? metrics.performance.errorRate.toFixed(2) : 0}%
                        </div>
                        <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
                          <div 
                            style={{ width: `${metrics?.performance.errorRate || 0}%` }} 
                            className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-red-500">
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Storage Details */}
            {metrics?.storage?.buckets && metrics.storage.buckets.length > 0 && (
              <div className="bg-white shadow overflow-hidden sm:rounded-md mb-8">
                <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    Storage Buckets
                  </h3>
                </div>
                <div className="divide-y divide-gray-200">
                  {metrics?.storage?.buckets?.map((bucket) => (
                    <div key={bucket.name} className="px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-connection-blue truncate">{bucket.name}</p>
                        <div className="ml-2 flex-shrink-0 flex">
                          <p className="text-sm text-gray-500">{formatBytes(bucket.size)}</p>
                        </div>
                      </div>
                      <div className="mt-2 w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-connection-blue h-1.5 rounded-full"
                          style={{ width: `${(bucket.size / (metrics?.storage?.totalSize || 1)) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
} 