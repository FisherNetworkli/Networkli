'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  HomeIcon, 
  PlusIcon,
  CalendarIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  PencilIcon,
  TrashIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

interface ScheduledTask {
  id: string;
  name: string;
  description: string;
  frequency: 'hourly' | 'daily' | 'weekly' | 'monthly';
  status: 'active' | 'paused' | 'failed';
  last_run: string | null;
  next_run: string;
  created_at: string;
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<ScheduledTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddTaskModal, setShowAddTaskModal] = useState(false);
  const [editingTask, setEditingTask] = useState<ScheduledTask | null>(null);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const supabase = createClientComponentClient();

  // Generate mock tasks
  const generateMockTasks = (): ScheduledTask[] => {
    const mockTasks: ScheduledTask[] = [
      {
        id: 'task-1',
        name: 'Database Backup',
        description: 'Automated database backup to cloud storage',
        frequency: 'daily',
        status: 'active',
        last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-2',
        name: 'User Activity Report',
        description: 'Generate weekly user activity and engagement report',
        frequency: 'weekly',
        status: 'active',
        last_run: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 4 * 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-3',
        name: 'Storage Cleanup',
        description: 'Clean up temporary files and expired uploads',
        frequency: 'daily',
        status: 'active',
        last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-4',
        name: 'SSL Certificate Check',
        description: 'Check SSL certificate expiration and send notification if needed',
        frequency: 'weekly',
        status: 'active',
        last_run: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-5',
        name: 'Monthly User Analytics',
        description: 'Generate comprehensive monthly analytics report for all users',
        frequency: 'monthly',
        status: 'active',
        last_run: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 15 * 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 120 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-6',
        name: 'Log Rotation',
        description: 'Rotate and archive system logs',
        frequency: 'daily',
        status: 'active',
        last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 75 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-7',
        name: 'Email Deliverability Test',
        description: 'Test email sending capabilities and deliverability',
        frequency: 'weekly',
        status: 'paused',
        last_run: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 100 * 24 * 60 * 60 * 1000).toISOString(), // Far in the future because paused
        created_at: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'task-8',
        name: 'Database Maintenance',
        description: 'Run database vacuum and optimization procedures',
        frequency: 'weekly',
        status: 'failed',
        last_run: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        next_run: new Date(Date.now() + 6 * 24 * 60 * 60 * 1000).toISOString(),
        created_at: new Date(Date.now() - 65 * 24 * 60 * 60 * 1000).toISOString()
      }
    ];
    
    return mockTasks;
  };

  useEffect(() => {
    // In a real app, fetch tasks from database
    const mockTasks = generateMockTasks();
    setTasks(mockTasks);
    setLoading(false);
  }, []);

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Never';
    
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (error) {
      return 'Invalid date';
    }
  };

  const getStatusBadgeClass = (status: ScheduledTask['status']) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getFrequencyIcon = (frequency: ScheduledTask['frequency']) => {
    switch (frequency) {
      case 'hourly':
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
      case 'daily':
        return <CalendarIcon className="h-5 w-5 text-blue-500" />;
      case 'weekly':
        return <CalendarIcon className="h-5 w-5 text-purple-500" />;
      case 'monthly':
        return <CalendarIcon className="h-5 w-5 text-indigo-500" />;
      default:
        return <CalendarIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const handleToggleTaskStatus = (taskId: string) => {
    setTasks(prevTasks => 
      prevTasks.map(task => {
        if (task.id === taskId) {
          const newStatus = task.status === 'active' ? 'paused' : 'active';
          
          // Update next_run based on new status
          let nextRun = task.next_run;
          if (newStatus === 'paused') {
            // If pausing, set next_run far in the future
            const farFuture = new Date();
            farFuture.setFullYear(farFuture.getFullYear() + 10);
            nextRun = farFuture.toISOString();
          } else if (newStatus === 'active') {
            // If activating, set next_run based on frequency
            const now = new Date();
            switch (task.frequency) {
              case 'hourly':
                now.setHours(now.getHours() + 1);
                break;
              case 'daily':
                now.setDate(now.getDate() + 1);
                break;
              case 'weekly':
                now.setDate(now.getDate() + 7);
                break;
              case 'monthly':
                now.setMonth(now.getMonth() + 1);
                break;
            }
            nextRun = now.toISOString();
          }
          
          return { ...task, status: newStatus, next_run: nextRun };
        }
        return task;
      })
    );
    
    // Show notification
    setNotification({
      type: 'success',
      message: `Task status updated successfully`
    });
    
    // Clear notification after 3 seconds
    setTimeout(() => {
      setNotification(null);
    }, 3000);
  };

  const handleRunTaskNow = (taskId: string) => {
    setTasks(prevTasks => 
      prevTasks.map(task => {
        if (task.id === taskId) {
          // Update last_run to now
          const now = new Date().toISOString();
          return { ...task, last_run: now };
        }
        return task;
      })
    );
    
    // Show notification
    setNotification({
      type: 'success',
      message: `Task executed successfully`
    });
    
    // Clear notification after 3 seconds
    setTimeout(() => {
      setNotification(null);
    }, 3000);
  };

  const handleDeleteTask = (taskId: string) => {
    // Filter out the deleted task
    setTasks(prevTasks => prevTasks.filter(task => task.id !== taskId));
    
    // Show notification
    setNotification({
      type: 'success',
      message: `Task deleted successfully`
    });
    
    // Clear notification after 3 seconds
    setTimeout(() => {
      setNotification(null);
    }, 3000);
  };

  return (
    <div className="py-6">
      <div className="px-4 sm:px-6 md:px-8">
        {/* Header with actions */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold text-gray-900">Scheduled Tasks</h1>
          <div className="flex space-x-3">
            <button
              onClick={() => setShowAddTaskModal(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark"
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              Add Task
            </button>
            <Link
              href="/admin"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark"
            >
              <HomeIcon className="h-4 w-4 mr-2" />
              Back to Dashboard
            </Link>
          </div>
        </div>
        
        {/* Notification */}
        {notification && (
          <div className={`mb-6 rounded-md p-4 ${
            notification.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
          }`}>
            <div className="flex">
              <div className="flex-shrink-0">
                {notification.type === 'success' ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-400" />
                ) : (
                  <XCircleIcon className="h-5 w-5 text-red-400" />
                )}
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium">{notification.message}</p>
              </div>
            </div>
          </div>
        )}
        
        {/* Tasks table */}
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-connection-blue"></div>
            </div>
          ) : tasks.length === 0 ? (
            <div className="py-16 text-center">
              <CalendarIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No scheduled tasks</h3>
              <p className="mt-1 text-sm text-gray-500">Get started by creating a new scheduled task.</p>
              <div className="mt-6">
                <button
                  onClick={() => setShowAddTaskModal(true)}
                  className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-connection-blue hover:bg-connection-blue-dark"
                >
                  <PlusIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                  New Task
                </button>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Task
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Frequency
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Run
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Next Run
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {tasks.map((task) => (
                    <tr key={task.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <div className="flex flex-col">
                          <div className="text-sm font-medium text-gray-900">{task.name}</div>
                          <div className="text-sm text-gray-500">{task.description}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {getFrequencyIcon(task.frequency)}
                          <span className="ml-2 text-sm text-gray-900 capitalize">{task.frequency}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusBadgeClass(task.status)}`}>
                          {task.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(task.last_run)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(task.next_run)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end space-x-2">
                          <button
                            onClick={() => handleToggleTaskStatus(task.id)}
                            className="text-connection-blue hover:text-connection-blue-dark"
                            title={task.status === 'active' ? 'Pause task' : 'Activate task'}
                          >
                            {task.status === 'active' ? (
                              <span className="flex items-center">
                                <PauseIcon className="h-5 w-5" aria-hidden="true" />
                              </span>
                            ) : (
                              <span className="flex items-center">
                                <PlayIcon className="h-5 w-5" aria-hidden="true" />
                              </span>
                            )}
                          </button>
                          <button
                            onClick={() => handleRunTaskNow(task.id)}
                            className="text-connection-blue hover:text-connection-blue-dark"
                            title="Run now"
                          >
                            <ArrowPathIcon className="h-5 w-5" aria-hidden="true" />
                          </button>
                          <button
                            onClick={() => setEditingTask(task)}
                            className="text-connection-blue hover:text-connection-blue-dark"
                            title="Edit task"
                          >
                            <PencilIcon className="h-5 w-5" aria-hidden="true" />
                          </button>
                          <button
                            onClick={() => handleDeleteTask(task.id)}
                            className="text-red-600 hover:text-red-900"
                            title="Delete task"
                          >
                            <TrashIcon className="h-5 w-5" aria-hidden="true" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Helper icons for play/pause functionality
const PlayIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
  </svg>
);

const PauseIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" />
  </svg>
); 