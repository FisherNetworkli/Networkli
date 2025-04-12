'use client';

import React, { useState } from 'react';

interface Message {
  id: string;
  name: string;
  email: string;
  subject: string;
  message: string;
  status: 'UNREAD' | 'READ';
  createdAt: Date;
}

interface MessageViewProps {
  message: Message;
  onClose: () => void;
  onStatusChange: () => void;
}

export default function MessageView({ message, onClose, onStatusChange }: MessageViewProps) {
  const [isUpdating, setIsUpdating] = useState(false);

  const handleMarkAsRead = async () => {
    setIsUpdating(true);
    try {
      const response = await fetch(`/api/contact/${message.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: 'READ' }),
      });

      if (!response.ok) {
        throw new Error('Failed to update message status');
      }

      onStatusChange();
    } catch (error) {
      console.error('Error updating message status:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full p-6">
        <div className="flex justify-between items-start mb-4">
          <h2 className="text-xl font-semibold text-gray-900">{message.subject}</h2>
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

        <div className="space-y-4">
          <div>
            <p className="text-sm text-gray-500">From</p>
            <p className="text-sm font-medium text-gray-900">{message.name} ({message.email})</p>
          </div>

          <div>
            <p className="text-sm text-gray-500">Message</p>
            <p className="mt-1 text-sm text-gray-900 whitespace-pre-wrap">{message.message}</p>
          </div>

          <div className="flex items-center justify-between pt-4 border-t">
            <div className="flex items-center space-x-2">
              <span
                className={`px-2 py-1 text-xs font-semibold rounded-full ${
                  message.status === 'UNREAD'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-green-100 text-green-800'
                }`}
              >
                {message.status}
              </span>
              <span className="text-sm text-gray-500">
                {new Date(message.createdAt).toLocaleString()}
              </span>
            </div>

            {message.status === 'UNREAD' && (
              <button
                onClick={handleMarkAsRead}
                disabled={isUpdating}
                className="px-4 py-2 text-sm font-medium text-white bg-connection-blue rounded-md hover:bg-connection-blue-70 disabled:opacity-50"
              >
                {isUpdating ? 'Updating...' : 'Mark as Read'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 