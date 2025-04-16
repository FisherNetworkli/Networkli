'use client';

import { useEffect, useState } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';

interface Message {
  id: string;
  content: string;
  sender_id: string;
  sender_name: string;
  sender_avatar?: string;
  created_at: string;
  read: boolean;
}

interface Conversation {
  id: string;
  name: string;
  type: 'direct' | 'group' | 'event';
  avatar?: string;
  last_message?: string;
  last_message_time?: string;
  unread_count: number;
  participants?: { id: string; name: string; avatar?: string }[];
}

export default function MessagesPage() {
  const [user, setUser] = useState<User | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUser(session.user);
      }
    };

    getUser();
  }, [supabase.auth]);

  useEffect(() => {
    const fetchConversations = async () => {
      setLoading(true);
      
      // In a real app, this would fetch from the database
      // For now, we're using placeholder data
      const mockConversations: Conversation[] = [
        {
          id: '1',
          name: 'Sarah Chen',
          type: 'direct',
          avatar: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3',
          last_message: 'Hi, I saw your profile and I was impressed with your experience in UX design.',
          last_message_time: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
          unread_count: 2,
          participants: [
            { id: '101', name: 'Sarah Chen', avatar: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3' }
          ]
        },
        {
          id: '2',
          name: 'Michael Park',
          type: 'direct',
          avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3',
          last_message: 'Thanks for the connection! Looking forward to collaborating.',
          last_message_time: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          unread_count: 0,
          participants: [
            { id: '102', name: 'Michael Park', avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3' }
          ]
        },
        {
          id: '3',
          name: 'Software Engineers Network',
          type: 'group',
          avatar: 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3',
          last_message: 'Anyone attending the tech conference next month?',
          last_message_time: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
          unread_count: 5,
          participants: [
            { id: '103', name: 'David Rodriguez' },
            { id: '104', name: 'Emily Johnson' },
            { id: '105', name: 'Alex Thompson' },
            { id: '106', name: 'Jessica Wong' }
          ]
        },
        {
          id: '4',
          name: 'Tech Networking Mixer',
          type: 'event',
          avatar: 'https://images.unsplash.com/photo-1511578314322-379afb476865?ixlib=rb-4.0.3',
          last_message: 'Event starts in 3 days! Don\'t forget to bring business cards.',
          last_message_time: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
          unread_count: 1,
          participants: [
            { id: '107', name: 'SF Tech Group' },
            { id: '108', name: 'Laura Kim' },
            { id: '109', name: 'Ryan Garcia' },
            { id: '110', name: 'Sophia Martinez' }
          ]
        },
        {
          id: '5',
          name: 'Women in Tech',
          type: 'group',
          avatar: 'https://images.unsplash.com/photo-1573164574001-518958d9baa2?ixlib=rb-4.0.3',
          last_message: 'Check out this article on promoting diversity in tech leadership.',
          last_message_time: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          unread_count: 0,
          participants: [
            { id: '111', name: 'Sarah Chen' },
            { id: '112', name: 'Jessica Wong' },
            { id: '113', name: 'Emily Johnson' },
            { id: '114', name: 'Laura Kim' }
          ]
        }
      ];
      
      setConversations(mockConversations);
      setLoading(false);
    };

    fetchConversations();
  }, []);

  // Set active conversation and fetch its messages
  const handleSelectConversation = (conversation: Conversation) => {
    setActiveConversation(conversation);
    
    // Simulate fetching messages for this conversation
    const mockMessages: Message[] = [];
    
    // Generate some mock messages
    const messageCount = Math.floor(Math.random() * 10) + 5; // 5-15 messages
    const now = new Date();
    
    for (let i = 0; i < messageCount; i++) {
      const isFromUser = Math.random() > 0.5;
      const timeOffset = (messageCount - i) * (Math.random() * 5 + 1) * 60 * 1000; // Random minutes earlier
      
      mockMessages.push({
        id: `msg-${conversation.id}-${i}`,
        content: isFromUser 
          ? getRandomUserMessage() 
          : getRandomResponseMessage(conversation.type),
        sender_id: isFromUser ? 'current-user' : `other-${i}`,
        sender_name: isFromUser 
          ? 'You' 
          : conversation.type === 'direct' 
            ? conversation.name 
            : conversation.participants?.[Math.floor(Math.random() * (conversation.participants?.length || 1))]?.name || 'Unknown',
        sender_avatar: isFromUser 
          ? undefined 
          : conversation.type === 'direct' 
            ? conversation.avatar 
            : undefined,
        created_at: new Date(now.getTime() - timeOffset).toISOString(),
        read: true
      });
    }
    
    // Sort by timestamp (oldest first)
    mockMessages.sort((a, b) => 
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );
    
    setMessages(mockMessages);
    
    // Mark conversation as read in the list
    setConversations(prevConversations => 
      prevConversations.map(c => 
        c.id === conversation.id 
          ? { ...c, unread_count: 0 } 
          : c
      )
    );
  };

  const handleSendMessage = () => {
    if (!newMessage.trim() || !activeConversation) return;
    
    // Add the new message to the current conversation
    const newMsg: Message = {
      id: `msg-new-${Date.now()}`,
      content: newMessage,
      sender_id: 'current-user',
      sender_name: 'You',
      created_at: new Date().toISOString(),
      read: true
    };
    
    setMessages(prev => [...prev, newMsg]);
    setNewMessage('');
    
    // Update the conversation in the list with the new last message
    setConversations(prevConversations => 
      prevConversations.map(c => 
        c.id === activeConversation.id 
          ? { 
              ...c, 
              last_message: newMessage.substring(0, 60) + (newMessage.length > 60 ? '...' : ''),
              last_message_time: new Date().toISOString()
            } 
          : c
      )
    );
  };

  // Helper functions for generating random message content
  const getRandomUserMessage = () => {
    const messages = [
      "Hi there! How are you doing?",
      "Thanks for the information. That's really helpful.",
      "I was wondering if you'd be interested in collaborating on a project?",
      "Can you tell me more about your experience in this field?",
      "I'm looking forward to the upcoming event!",
      "Do you have any recommendations for resources on this topic?",
      "I've been working on something similar lately.",
      "That's a great point. I hadn't thought about it that way.",
      "When are you free to meet up and discuss this further?",
      "I found an interesting article that might be relevant to our discussion."
    ];
    return messages[Math.floor(Math.random() * messages.length)];
  };

  const getRandomResponseMessage = (conversationType: 'direct' | 'group' | 'event') => {
    const directMessages = [
      "I'm doing well, thanks for asking! How about you?",
      "No problem at all, happy to help!",
      "I'd definitely be interested in collaborating. What did you have in mind?",
      "I've been in this field for about 5 years now, specializing in...",
      "Let me check my calendar and get back to you with some available times.",
      "I'd be happy to share some resources. Are you looking for something specific?",
      "That sounds interesting! I'd love to hear more about what you're working on.",
      "Thanks! I always try to look at problems from different angles.",
      "I'm available next Tuesday afternoon if that works for you?",
      "Thanks for sharing! I'll definitely check it out."
    ];
    
    const groupMessages = [
      "Has anyone here used the new framework that was just released?",
      "I'm looking for recommendations for project management tools.",
      "Welcome to the new members who joined this week!",
      "Did everyone see the announcement about the upcoming meetup?",
      "Can someone help me troubleshoot this issue I'm having?",
      "I just shared a useful resource in the files section.",
      "What do you all think about the recent industry developments?",
      "Is anyone attending the conference next month?",
      "Thanks for all the great discussions this week!",
      "I'm organizing a virtual coffee chat. Let me know if you're interested!"
    ];
    
    const eventMessages = [
      "What time does the event start again?",
      "Is there a dress code for this event?",
      "Can we bring guests to this event?",
      "Where exactly is the venue located?",
      "Will there be food and drinks provided?",
      "Are the presentations going to be recorded?",
      "I'm looking forward to the networking session!",
      "Who's speaking at this event? The agenda looks great.",
      "Will we receive the presentation slides afterward?",
      "Is there parking available at the venue?"
    ];
    
    switch (conversationType) {
      case 'direct':
        return directMessages[Math.floor(Math.random() * directMessages.length)];
      case 'group':
        return groupMessages[Math.floor(Math.random() * groupMessages.length)];
      case 'event':
        return eventMessages[Math.floor(Math.random() * eventMessages.length)];
      default:
        return "Hello there!";
    }
  };

  // Format date for display
  const formatMessageTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      // Today, show time
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      // Yesterday
      return 'Yesterday';
    } else if (diffDays < 7) {
      // This week, show day name
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      // Older, show date
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  // Get icon for conversation type
  const getConversationIcon = (type: 'direct' | 'group' | 'event') => {
    switch (type) {
      case 'direct':
        return (
          <svg className="h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
            <path d="M10 8a3 3 0 100-6 3 3 0 000 6zM3.465 14.493a1.23 1.23 0 00.41 1.412A9.957 9.957 0 0010 18c2.31 0 4.438-.784 6.131-2.1.43-.333.604-.903.408-1.41a7.002 7.002 0 00-13.074.003z" />
          </svg>
        );
      case 'group':
        return (
          <svg className="h-4 w-4 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
            <path d="M7 8a3 3 0 100-6 3 3 0 000 6zM14.5 9a2.5 2.5 0 100-5 2.5 2.5 0 000 5zM1.615 16.428a1.224 1.224 0 01-.569-1.175 6.002 6.002 0 0111.908 0c.058.467-.172.92-.57 1.174A9.953 9.953 0 017 18a9.953 9.953 0 01-5.385-1.572zM14.5 16h-.106c.07-.297.088-.611.048-.933a7.47 7.47 0 00-1.588-3.755 4.502 4.502 0 015.874 2.636.818.818 0 01-.36.98A7.465 7.465 0 0114.5 16z" />
          </svg>
        );
      case 'event':
        return (
          <svg className="h-4 w-4 text-purple-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd" />
          </svg>
        );
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Messages</h1>
      
      <div className="flex flex-col md:flex-row h-[calc(100vh-12rem)] bg-white rounded-lg border shadow-sm overflow-hidden">
        {/* Conversations List */}
        <div className="w-full md:w-1/3 border-r">
          {/* Filter tabs */}
          <div className="flex border-b text-sm">
            <button className="flex-1 py-3 font-medium text-blue-600 border-b-2 border-blue-600">All</button>
            <button className="flex-1 py-3 font-medium text-gray-500 hover:text-gray-700">Direct</button>
            <button className="flex-1 py-3 font-medium text-gray-500 hover:text-gray-700">Groups</button>
            <button className="flex-1 py-3 font-medium text-gray-500 hover:text-gray-700">Events</button>
          </div>
          
          {/* Search */}
          <div className="p-3 border-b">
            <div className="relative">
              <input
                type="text"
                placeholder="Search messages..."
                className="w-full pl-10 pr-4 py-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <svg
                className="absolute left-3 top-2.5 h-5 w-5 text-gray-400"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
          </div>
          
          {/* Conversations */}
          <div className="overflow-y-auto h-[calc(100%-6.5rem)]">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`flex items-start p-3 border-b hover:bg-gray-50 cursor-pointer ${
                  activeConversation?.id === conversation.id ? 'bg-blue-50' : ''
                }`}
                onClick={() => handleSelectConversation(conversation)}
              >
                {/* Avatar or icon */}
                <div className="relative mr-3">
                  {conversation.avatar ? (
                    <img
                      src={conversation.avatar}
                      alt={conversation.name}
                      className="w-12 h-12 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center">
                      <span className="text-lg font-medium text-gray-600">
                        {conversation.name.charAt(0)}
                      </span>
                    </div>
                  )}
                  <div className="absolute -right-1 -bottom-1">
                    {getConversationIcon(conversation.type)}
                  </div>
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-baseline">
                    <h3 className="font-medium truncate">{conversation.name}</h3>
                    <span className="text-xs text-gray-500 whitespace-nowrap ml-2">
                      {conversation.last_message_time && formatMessageTime(conversation.last_message_time)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 truncate">{conversation.last_message}</p>
                </div>
                
                {conversation.unread_count > 0 && (
                  <div className="ml-2 bg-blue-500 text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">
                    {conversation.unread_count}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        {/* Conversation/Chat Area */}
        <div className="w-full md:w-2/3 flex flex-col">
          {activeConversation ? (
            <>
              {/* Conversation Header */}
              <div className="p-3 border-b flex items-center">
                <div className="relative mr-3">
                  {activeConversation.avatar ? (
                    <img
                      src={activeConversation.avatar}
                      alt={activeConversation.name}
                      className="w-10 h-10 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center">
                      <span className="text-lg font-medium text-gray-600">
                        {activeConversation.name.charAt(0)}
                      </span>
                    </div>
                  )}
                  <div className="absolute -right-1 -bottom-1">
                    {getConversationIcon(activeConversation.type)}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium">{activeConversation.name}</h3>
                  <div className="flex items-center text-xs text-gray-500">
                    {activeConversation.type === 'direct' ? (
                      <span>Direct message</span>
                    ) : (
                      <span>
                        {activeConversation.type === 'group' ? 'Group' : 'Event'} · 
                        {activeConversation.participants?.length} participants
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="ml-auto flex space-x-2">
                  <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z" />
                    </svg>
                  </button>
                  <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                    </svg>
                  </button>
                  <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
                    </svg>
                  </button>
                </div>
              </div>
              
              {/* Messages */}
              <div className="flex-1 p-4 overflow-y-auto">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex mb-4 ${
                      message.sender_id === 'current-user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {message.sender_id !== 'current-user' && (
                      <div className="mr-2 flex-shrink-0">
                        {message.sender_avatar ? (
                          <img
                            src={message.sender_avatar}
                            alt={message.sender_name}
                            className="w-8 h-8 rounded-full"
                          />
                        ) : (
                          <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                            <span className="text-sm font-medium text-gray-600">
                              {message.sender_name.charAt(0)}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div
                      className={`max-w-xs md:max-w-md lg:max-w-lg rounded-lg px-4 py-2 ${
                        message.sender_id === 'current-user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {message.sender_id !== 'current-user' && activeConversation?.type !== 'direct' && (
                        <p className="text-xs font-medium mb-1">
                          {message.sender_name}
                        </p>
                      )}
                      <p>{message.content}</p>
                      <p className="text-xs mt-1 text-right">
                        {formatMessageTime(message.created_at)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Message Input */}
              <div className="border-t p-3">
                <div className="flex items-center">
                  <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z" clipRule="evenodd" />
                    </svg>
                  </button>
                  <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8 4a3 3 0 00-3 3v4a5 5 0 0010 0V7a1 1 0 112 0v4a7 7 0 11-14 0V7a5 5 0 0110 0v4a3 3 0 11-6 0V7a1 1 0 012 0v4a1 1 0 102 0V7a3 3 0 00-3-3z" clipRule="evenodd" />
                    </svg>
                  </button>
                  <input
                    type="text"
                    placeholder="Type a message..."
                    className="flex-1 border rounded-md px-4 py-2 mx-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                  />
                  <button
                    className="bg-blue-600 text-white p-2 rounded-full hover:bg-blue-700"
                    onClick={handleSendMessage}
                    disabled={!newMessage.trim()}
                  >
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                    </svg>
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                <svg className="h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-800 mb-2">Your Messages</h3>
              <p className="text-gray-500 max-w-xs">
                Select a conversation from the list to view messages or start a new conversation.
              </p>
              <button className="mt-6 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                New Message
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 