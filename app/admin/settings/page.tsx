'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  HomeIcon, 
  ArrowPathIcon,
  Cog6ToothIcon,
  BellIcon,
  ShieldCheckIcon,
  EnvelopeIcon,
  UserGroupIcon,
  DocumentIcon,
  ArrowPathRoundedSquareIcon,
  CloudArrowUpIcon
} from '@heroicons/react/24/outline';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

interface SystemSettings {
  id: string;
  category: 'general' | 'security' | 'email' | 'users' | 'content' | 'api' | 'backups';
  key: string;
  value: string | boolean | number;
  label: string;
  description: string;
  type: 'text' | 'boolean' | 'number' | 'select' | 'email' | 'url' | 'textarea' | 'color';
  options?: string[];
  updated_at: string;
}

export default function SettingsPage() {
  const [categories, setCategories] = useState<string[]>([
    'general', 'security', 'email', 'users', 'content', 'api', 'backups'
  ]);
  const [activeCategory, setActiveCategory] = useState<string>('general');
  const [settings, setSettings] = useState<SystemSettings[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [editedSettings, setEditedSettings] = useState<Record<string, any>>({});
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);
  
  const router = useRouter();
  const supabase = createClientComponentClient();

  // Generate mock system settings
  const generateMockSettings = (): SystemSettings[] => {
    const mockSettings: SystemSettings[] = [
      // General Settings
      {
        id: 'setting-1',
        category: 'general',
        key: 'site_name',
        value: 'Networkli',
        label: 'Site Name',
        description: 'The name of your site displayed in emails and on pages',
        type: 'text',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-2',
        category: 'general',
        key: 'site_description',
        value: 'A professional networking platform',
        label: 'Site Description',
        description: 'A brief description of your site used for SEO',
        type: 'textarea',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-3',
        category: 'general',
        key: 'site_url',
        value: 'https://networkli.com',
        label: 'Site URL',
        description: 'The primary URL of your site',
        type: 'url',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-4',
        category: 'general',
        key: 'primary_color',
        value: '#4f46e5',
        label: 'Primary Color',
        description: 'The main color for your site theme',
        type: 'color',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-5',
        category: 'general',
        key: 'timezone',
        value: 'UTC',
        label: 'Default Timezone',
        description: 'The default timezone for your site',
        type: 'select',
        options: ['UTC', 'America/New_York', 'Europe/London', 'Asia/Tokyo', 'Australia/Sydney'],
        updated_at: new Date().toISOString()
      },
      
      // Security Settings
      {
        id: 'setting-6',
        category: 'security',
        key: 'login_attempts',
        value: 5,
        label: 'Max Login Attempts',
        description: 'Number of failed login attempts before temporary lockout',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-7',
        category: 'security',
        key: 'password_expiry_days',
        value: 90,
        label: 'Password Expiry (Days)',
        description: 'Number of days after which passwords expire',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-8',
        category: 'security',
        key: 'enforce_2fa',
        value: false,
        label: 'Enforce 2FA',
        description: 'Require two-factor authentication for all users',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-9',
        category: 'security',
        key: 'session_timeout',
        value: 60,
        label: 'Session Timeout (Minutes)',
        description: 'Number of minutes of inactivity before session expiry',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-10',
        category: 'security',
        key: 'allowed_ip_ranges',
        value: '*',
        label: 'Allowed IP Ranges',
        description: 'Comma-separated list of allowed IP ranges for admin access',
        type: 'text',
        updated_at: new Date().toISOString()
      },
      
      // Email Settings
      {
        id: 'setting-11',
        category: 'email',
        key: 'smtp_host',
        value: 'smtp.example.com',
        label: 'SMTP Host',
        description: 'The hostname of your SMTP server',
        type: 'text',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-12',
        category: 'email',
        key: 'smtp_port',
        value: 587,
        label: 'SMTP Port',
        description: 'The port of your SMTP server',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-13',
        category: 'email',
        key: 'smtp_username',
        value: 'user@example.com',
        label: 'SMTP Username',
        description: 'The username for SMTP authentication',
        type: 'email',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-14',
        category: 'email',
        key: 'smtp_encryption',
        value: 'tls',
        label: 'SMTP Encryption',
        description: 'The encryption type for SMTP',
        type: 'select',
        options: ['none', 'ssl', 'tls'],
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-15',
        category: 'email',
        key: 'from_email',
        value: 'noreply@networkli.com',
        label: 'From Email',
        description: 'The email address used as sender for system emails',
        type: 'email',
        updated_at: new Date().toISOString()
      },
      
      // User Settings
      {
        id: 'setting-16',
        category: 'users',
        key: 'allow_signups',
        value: true,
        label: 'Allow New Signups',
        description: 'Allow new users to sign up for accounts',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-17',
        category: 'users',
        key: 'default_role',
        value: 'user',
        label: 'Default User Role',
        description: 'The default role assigned to new users',
        type: 'select',
        options: ['user', 'organizer', 'admin'],
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-18',
        category: 'users',
        key: 'require_email_verification',
        value: true,
        label: 'Require Email Verification',
        description: 'Require users to verify their email before accessing the platform',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-19',
        category: 'users',
        key: 'allow_profile_editing',
        value: true,
        label: 'Allow Profile Editing',
        description: 'Allow users to edit their profiles',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      
      // Content Settings
      {
        id: 'setting-20',
        category: 'content',
        key: 'allow_comments',
        value: true,
        label: 'Allow Comments',
        description: 'Allow users to comment on content',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-21',
        category: 'content',
        key: 'moderate_comments',
        value: false,
        label: 'Moderate Comments',
        description: 'Require approval for comments before they are visible',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-22',
        category: 'content',
        key: 'max_file_size',
        value: 5,
        label: 'Max File Size (MB)',
        description: 'Maximum file size for uploads in megabytes',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-23',
        category: 'content',
        key: 'allowed_file_types',
        value: 'jpg,jpeg,png,gif,pdf,doc,docx',
        label: 'Allowed File Types',
        description: 'Comma-separated list of allowed file extensions',
        type: 'text',
        updated_at: new Date().toISOString()
      },
      
      // API Settings
      {
        id: 'setting-24',
        category: 'api',
        key: 'enable_api',
        value: true,
        label: 'Enable API',
        description: 'Allow API access to the platform',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-25',
        category: 'api',
        key: 'api_rate_limit',
        value: 100,
        label: 'API Rate Limit',
        description: 'Maximum number of API requests per minute',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-26',
        category: 'api',
        key: 'api_token_expiry',
        value: 30,
        label: 'API Token Expiry (Days)',
        description: 'Number of days before API tokens expire',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-27',
        category: 'api',
        key: 'api_log_requests',
        value: true,
        label: 'Log API Requests',
        description: 'Log all API requests for auditing purposes',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      
      // Backup Settings
      {
        id: 'setting-28',
        category: 'backups',
        key: 'auto_backup',
        value: true,
        label: 'Automatic Backups',
        description: 'Enable automatic backups of the system',
        type: 'boolean',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-29',
        category: 'backups',
        key: 'backup_frequency',
        value: 'daily',
        label: 'Backup Frequency',
        description: 'How often backups should be created',
        type: 'select',
        options: ['hourly', 'daily', 'weekly', 'monthly'],
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-30',
        category: 'backups',
        key: 'backup_retention',
        value: 30,
        label: 'Backup Retention (Days)',
        description: 'Number of days to keep backups before deleting',
        type: 'number',
        updated_at: new Date().toISOString()
      },
      {
        id: 'setting-31',
        category: 'backups',
        key: 'backup_location',
        value: 's3://networkli-backups',
        label: 'Backup Storage Location',
        description: 'Where to store backups (S3 bucket or path)',
        type: 'text',
        updated_at: new Date().toISOString()
      }
    ];
    
    return mockSettings;
  };

  useEffect(() => {
    // In a real app, fetch settings from database
    const mockSettings = generateMockSettings();
    setSettings(mockSettings);
    
    // Initialize editedSettings with current values
    const initialEdited: Record<string, any> = {};
    mockSettings.forEach(setting => {
      initialEdited[setting.key] = setting.value;
    });
    setEditedSettings(initialEdited);
    
    setLoading(false);
  }, []);

  // Filter settings by active category
  const filteredSettings = settings.filter(setting => setting.category === activeCategory);

  const handleInputChange = (key: string, value: any) => {
    setEditedSettings(prev => ({
      ...prev,
      [key]: value
    }));
    setHasChanges(true);
  };

  const handleSaveSettings = async () => {
    setSaving(true);
    
    try {
      // Simulate API call with timeout
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update settings with edited values
      const updatedSettings = settings.map(setting => {
        if (editedSettings[setting.key] !== undefined) {
          return {
            ...setting,
            value: editedSettings[setting.key],
            updated_at: new Date().toISOString()
          };
        }
        return setting;
      });
      
      setSettings(updatedSettings);
      setHasChanges(false);
      setNotification({
        type: 'success',
        message: 'Settings saved successfully'
      });
      
      // Clear notification after 3 seconds
      setTimeout(() => {
        setNotification(null);
      }, 3000);
    } catch (error) {
      setNotification({
        type: 'error',
        message: 'Failed to save settings'
      });
      
      // Clear notification after 3 seconds
      setTimeout(() => {
        setNotification(null);
      }, 3000);
    } finally {
      setSaving(false);
    }
  };

  const handleResetSettings = () => {
    // Reset edited settings to current values
    const resetEdited: Record<string, any> = {};
    settings.forEach(setting => {
      resetEdited[setting.key] = setting.value;
    });
    setEditedSettings(resetEdited);
    setHasChanges(false);
  };

  const formatDate = (dateString: string) => {
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

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'general':
        return <Cog6ToothIcon className="w-5 h-5" />;
      case 'security':
        return <ShieldCheckIcon className="w-5 h-5" />;
      case 'email':
        return <EnvelopeIcon className="w-5 h-5" />;
      case 'users':
        return <UserGroupIcon className="w-5 h-5" />;
      case 'content':
        return <DocumentIcon className="w-5 h-5" />;
      case 'api':
        return <ArrowPathRoundedSquareIcon className="w-5 h-5" />;
      case 'backups':
        return <CloudArrowUpIcon className="w-5 h-5" />;
      default:
        return <Cog6ToothIcon className="w-5 h-5" />;
    }
  };

  const renderSettingInput = (setting: SystemSettings) => {
    const value = editedSettings[setting.key] !== undefined ? editedSettings[setting.key] : setting.value;
    
    switch (setting.type) {
      case 'text':
      case 'email':
      case 'url':
        return (
          <input
            type={setting.type}
            id={setting.key}
            value={value as string}
            onChange={(e) => handleInputChange(setting.key, e.target.value)}
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
          />
        );
      case 'textarea':
        return (
          <textarea
            id={setting.key}
            value={value as string}
            onChange={(e) => handleInputChange(setting.key, e.target.value)}
            rows={3}
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
          />
        );
      case 'boolean':
        return (
          <div className="flex items-center h-5">
            <input
              type="checkbox"
              id={setting.key}
              checked={value as boolean}
              onChange={(e) => handleInputChange(setting.key, e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-connection-blue focus:ring-connection-blue"
            />
          </div>
        );
      case 'number':
        return (
          <input
            type="number"
            id={setting.key}
            value={value as number}
            onChange={(e) => handleInputChange(setting.key, parseFloat(e.target.value) || 0)}
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
          />
        );
      case 'select':
        return (
          <select
            id={setting.key}
            value={value as string}
            onChange={(e) => handleInputChange(setting.key, e.target.value)}
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
          >
            {setting.options?.map(option => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        );
      case 'color':
        return (
          <div className="flex items-center">
            <input
              type="color"
              id={setting.key}
              value={value as string}
              onChange={(e) => handleInputChange(setting.key, e.target.value)}
              className="h-8 w-8 border-gray-300 rounded-md cursor-pointer"
            />
            <input
              type="text"
              value={value as string}
              onChange={(e) => handleInputChange(setting.key, e.target.value)}
              className="ml-2 block rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold text-gray-900">System Settings</h1>
          <div className="flex space-x-3">
            <Link
              href="/admin"
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
            >
              <HomeIcon className="h-4 w-4 mr-1" />
              Back to Dashboard
            </Link>
          </div>
        </div>

        {/* Notification */}
        {notification && (
          <div className={`rounded-md p-4 mb-6 ${notification.type === 'success' ? 'bg-green-50' : 'bg-red-50'}`}>
            <div className="flex">
              <div className="flex-shrink-0">
                {notification.type === 'success' ? (
                  <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
              <div className="ml-3">
                <p className={`text-sm font-medium ${notification.type === 'success' ? 'text-green-800' : 'text-red-800'}`}>
                  {notification.message}
                </p>
              </div>
            </div>
          </div>
        )}

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-connection-blue"></div>
          </div>
        ) : (
          <div className="flex flex-col md:flex-row">
            {/* Sidebar */}
            <div className="w-full md:w-64 mb-6 md:mb-0 md:mr-8">
              <div className="bg-white shadow rounded-lg overflow-hidden">
                <ul className="divide-y divide-gray-200">
                  {categories.map(category => (
                    <li key={category}>
                      <button
                        onClick={() => setActiveCategory(category)}
                        className={`w-full flex items-center px-4 py-3 hover:bg-gray-50 ${
                          activeCategory === category ? 'bg-gray-50 text-connection-blue' : 'text-gray-700'
                        }`}
                      >
                        <span className={`mr-3 ${activeCategory === category ? 'text-connection-blue' : 'text-gray-400'}`}>
                          {getCategoryIcon(category)}
                        </span>
                        <span className="capitalize font-medium">{category}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Settings Content */}
            <div className="flex-1">
              <div className="bg-white shadow rounded-lg overflow-hidden">
                <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                  <h3 className="text-lg leading-6 font-medium text-gray-900 capitalize">
                    {activeCategory} Settings
                  </h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Configure settings for {activeCategory} functionality
                  </p>
                </div>
                
                <div className="px-4 py-5 sm:p-6">
                  <div className="space-y-6">
                    {filteredSettings.map(setting => (
                      <div key={setting.id} className="sm:grid sm:grid-cols-3 sm:gap-4 sm:items-start">
                        <label
                          htmlFor={setting.key}
                          className="block text-sm font-medium text-gray-700 sm:mt-px sm:pt-2"
                        >
                          {setting.label}
                          <p className="mt-1 text-xs text-gray-500">{setting.description}</p>
                        </label>
                        <div className="mt-1 sm:mt-0 sm:col-span-2">
                          {renderSettingInput(setting)}
                          <p className="mt-1 text-xs text-gray-500">
                            Last updated: {formatDate(setting.updated_at)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="px-4 py-3 bg-gray-50 text-right sm:px-6 flex justify-end space-x-3">
                  <button
                    type="button"
                    onClick={handleResetSettings}
                    disabled={!hasChanges || saving}
                    className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleSaveSettings}
                    disabled={!hasChanges || saving}
                    className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-connection-blue hover:bg-connection-blue-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue disabled:opacity-50"
                  >
                    {saving ? (
                      <>
                        <ArrowPathIcon className="animate-spin -ml-1 mr-2 h-4 w-4" />
                        Saving...
                      </>
                    ) : (
                      'Save Changes'
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 