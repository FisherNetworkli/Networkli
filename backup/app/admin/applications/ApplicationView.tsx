'use client';

import React, { useState } from 'react';

interface Application {
  id: string;
  name: string;
  email: string;
  phone: string | null;
  linkedin: string | null;
  github: string | null;
  portfolio: string | null;
  experience: string;
  availability: string;
  salary: string | null;
  referral: string | null;
  videoUrl: string;
  status: 'PENDING' | 'REVIEWING' | 'ACCEPTED' | 'REJECTED';
  createdAt: Date;
}

interface ApplicationViewProps {
  application: Application;
  onClose: () => void;
  onStatusChange: () => void;
}

export default function ApplicationView({ application, onClose, onStatusChange }: ApplicationViewProps) {
  const [isUpdating, setIsUpdating] = useState(false);

  const handleStatusChange = async (newStatus: Application['status']) => {
    setIsUpdating(true);
    try {
      const response = await fetch(`/api/applications/${application.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: newStatus }),
      });

      if (!response.ok) {
        throw new Error('Failed to update application status');
      }

      onStatusChange();
    } catch (error) {
      console.error('Error updating application status:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full p-6 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{application.name}</h2>
            <p className="text-sm text-gray-500">{application.email}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500"
          >
            <span className="sr-only">Close</span>
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Contact Information</h3>
              <div className="mt-2 space-y-2">
                <p className="text-sm text-gray-900">Phone: {application.phone || 'Not provided'}</p>
                {application.linkedin && (
                  <p className="text-sm">
                    LinkedIn:{' '}
                    <a href={application.linkedin} target="_blank" rel="noopener noreferrer" className="text-connection-blue hover:underline">
                      View Profile
                    </a>
                  </p>
                )}
                {application.github && (
                  <p className="text-sm">
                    GitHub:{' '}
                    <a href={application.github} target="_blank" rel="noopener noreferrer" className="text-connection-blue hover:underline">
                      View Profile
                    </a>
                  </p>
                )}
                {application.portfolio && (
                  <p className="text-sm">
                    Portfolio:{' '}
                    <a href={application.portfolio} target="_blank" rel="noopener noreferrer" className="text-connection-blue hover:underline">
                      View Portfolio
                    </a>
                  </p>
                )}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-500">Application Details</h3>
              <div className="mt-2 space-y-2">
                <p className="text-sm text-gray-900">Experience: {application.experience}</p>
                <p className="text-sm text-gray-900">Availability: {application.availability}</p>
                <p className="text-sm text-gray-900">Salary Expectation: {application.salary || 'Not provided'}</p>
                <p className="text-sm text-gray-900">Referral Source: {application.referral || 'Not provided'}</p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-gray-500">Video Introduction</h3>
            <div className="mt-2">
              <a
                href={application.videoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-connection-blue hover:underline text-sm"
              >
                Watch Video
              </a>
            </div>
          </div>

          <div className="border-t pt-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">
                  Applied on {new Date(application.createdAt).toLocaleString()}
                </span>
                <span
                  className={`px-2 py-1 text-xs font-semibold rounded-full ${
                    application.status === 'PENDING'
                      ? 'bg-yellow-100 text-yellow-800'
                      : application.status === 'REVIEWING'
                      ? 'bg-blue-100 text-blue-800'
                      : application.status === 'ACCEPTED'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {application.status}
                </span>
              </div>

              <div className="flex space-x-2">
                {application.status !== 'REJECTED' && (
                  <button
                    onClick={() => handleStatusChange('REJECTED')}
                    disabled={isUpdating}
                    className="px-3 py-1 text-sm font-medium text-white bg-red-600 rounded hover:bg-red-700 disabled:opacity-50"
                  >
                    Reject
                  </button>
                )}
                {application.status === 'PENDING' && (
                  <button
                    onClick={() => handleStatusChange('REVIEWING')}
                    disabled={isUpdating}
                    className="px-3 py-1 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
                  >
                    Start Review
                  </button>
                )}
                {application.status === 'REVIEWING' && (
                  <button
                    onClick={() => handleStatusChange('ACCEPTED')}
                    disabled={isUpdating}
                    className="px-3 py-1 text-sm font-medium text-white bg-green-600 rounded hover:bg-green-700 disabled:opacity-50"
                  >
                    Accept
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 