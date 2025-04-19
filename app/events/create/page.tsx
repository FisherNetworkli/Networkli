'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import Link from 'next/link';
import { Button } from '@/components/ui/button';

// List of event formats - extend this list as needed
const eventFormats = [
  'In-Person',
  'Virtual',
  'Hybrid',
  'Webinar',
  'Workshop',
  'Conference',
  'Networking Event',
  'Meetup',
  'Hackathon',
  'Panel Discussion',
  'Training Session',
  'Lecture',
  'Career Fair',
  'Festival',
  'Expo',
  'Round Table',
  'Mastermind',
  'Pitch Event',
  'Trade Show',
  'Seminar',
  'Bootcamp',
  'Retreat',
  'Fundraiser',
  'Concert',
  'Class',
];

export default function CreateEventPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const groupIdParam = searchParams?.get('groupId') ?? null;
  const [groupId, setGroupId] = useState<string | null>(groupIdParam);
  const [groupName, setGroupName] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [userGroups, setUserGroups] = useState<{ id: string; name: string }[]>([]);
  const [authError, setAuthError] = useState('');

  const supabase = createClientComponentClient();

  // Fetch current user ID
  useEffect(() => {
    const getUser = async () => {
      const { data } = await supabase.auth.getUser();
      setUserId(data.user?.id ?? null);
    };
    getUser();
  }, [supabase]);

  // Fetch group info or available groups for organizer
  useEffect(() => {
    if (!userId) return;
    if (groupId) {
      supabase
        .from('groups')
        .select('name,organizer_id')
        .eq('id', groupId)
        .single()
        .then(({ data, error }) => {
          if (error || !data) {
            setAuthError('Invalid group');
          } else if (data.organizer_id !== userId) {
            setAuthError('Not authorized to create event for this group');
          } else {
            setGroupName(data.name);
          }
        });
    } else {
      supabase
        .from('groups')
        .select('id,name')
        .eq('organizer_id', userId)
        .then(({ data, error }) => {
          if (!error && data) setUserGroups(data);
        });
    }
  }, [userId, groupId, supabase]);

  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('');
  const [format, setFormat] = useState(eventFormats[0]);
  const [description, setDescription] = useState('');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [location, setLocation] = useState('');
  const [meetingLink, setMeetingLink] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [tags, setTags] = useState('');
  const [maxAttendees, setMaxAttendees] = useState('');
  const [price, setPrice] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [questions, setQuestions] = useState<string[]>(['']);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleQuestionChange = (index: number, value: string) => {
    const newQs = [...questions];
    newQs[index] = value;
    setQuestions(newQs);
  };
  const addQuestion = () => setQuestions([...questions, '']);
  const removeQuestion = (idx: number) => setQuestions(questions.filter((_, i) => i !== idx));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const tagsArray = tags.split(',').map(t => t.trim()).filter(Boolean);
    const maxNum = maxAttendees ? parseInt(maxAttendees, 10) : null;
    const priceCents = price ? Math.round(parseFloat(price) * 100) : 0;

    try {
      const payload: any = {
        title,
        category,
        format,
        description,
        start_time: startTime ? new Date(startTime).toISOString() : null,
        end_time: endTime ? new Date(endTime).toISOString() : null,
        location,
        meeting_link: meetingLink,
        image_url: imageUrl,
        tags: tagsArray,
        max_attendees: maxNum,
        price: priceCents,
        ...(groupId ? { group_id: groupId } : {}),
        is_private: isPrivate,
        custom_questions: questions,
      };

      const { data, error: supaErr } = await supabase
        .from('events')
        .insert([payload])
        .select('id')
        .single();
      if (supaErr) throw supaErr;
      router.push(`/events/${data.id}`);
    } catch (err: any) {
      console.error('Error creating event:', err);
      setError(err.message || 'Error creating event');
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded shadow">
      <h1 className="text-2xl font-bold mb-4">Create New Event</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && <p className="text-red-500">{error}</p>}
        {authError && <p className="text-red-500">{authError}</p>}

        <div>
          <label className="block text-sm font-medium mb-1">Title</label>
          <input type="text" value={title} onChange={e => setTitle(e.target.value)} required className="w-full border rounded px-3 py-2" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Category</label>
          <input type="text" value={category} onChange={e => setCategory(e.target.value)} className="w-full border rounded px-3 py-2" placeholder="e.g. Technology, Health" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Format</label>
          <select value={format} onChange={e => setFormat(e.target.value)} className="w-full border rounded px-3 py-2">
            {eventFormats.map(fmt => (<option key={fmt} value={fmt}>{fmt}</option>))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Description</label>
          <textarea value={description} onChange={e => setDescription(e.target.value)} required rows={4} className="w-full border rounded px-3 py-2" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Start Date & Time</label>
            <input type="datetime-local" value={startTime} onChange={e => setStartTime(e.target.value)} required className="w-full border rounded px-3 py-2" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">End Date & Time</label>
            <input type="datetime-local" value={endTime} onChange={e => setEndTime(e.target.value)} className="w-full border rounded px-3 py-2" />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Location</label>
          <input type="text" value={location} onChange={e => setLocation(e.target.value)} className="w-full border rounded px-3 py-2" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Meeting Link (if virtual)</label>
          <input type="url" value={meetingLink} onChange={e => setMeetingLink(e.target.value)} className="w-full border rounded px-3 py-2" placeholder="https://example.com/meeting" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Image URL</label>
          <input type="url" value={imageUrl} onChange={e => setImageUrl(e.target.value)} className="w-full border rounded px-3 py-2" placeholder="https://example.com/image.jpg" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Tags (comma separated)</label>
          <input type="text" value={tags} onChange={e => setTags(e.target.value)} className="w-full border rounded px-3 py-2" placeholder="tech, networking" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Max Attendees</label>
          <input type="number" value={maxAttendees} onChange={e => setMaxAttendees(e.target.value)} className="w-full border rounded px-3 py-2" min="1" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Ticket Price (USD)</label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={price}
            onChange={e => setPrice(e.target.value)}
            className="w-full border rounded px-3 py-2"
            placeholder="0.00"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Private Event?</label>
          <input type="checkbox" checked={isPrivate} onChange={e => setIsPrivate(e.target.checked)} className="mr-2" /> Yes
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Custom Questions</label>
          {questions.map((q, idx) => (
            <div key={idx} className="flex items-center mb-2">
              <input type="text" value={q} onChange={e => handleQuestionChange(idx, e.target.value)} className="flex-1 border rounded px-3 py-2" placeholder={`Question #${idx + 1}`} />
              <button type="button" onClick={() => removeQuestion(idx)} className="ml-2 text-red-500">Remove</button>
            </div>
          ))}
          <button type="button" onClick={addQuestion} className="text-blue-600 mt-1">+ Add Question</button>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Group</label>
          {groupId ? (
            <input
              type="text"
              value={groupName || ''}
              disabled
              className="w-full border rounded px-3 py-2 bg-gray-100"
            />
          ) : (
            <select
              value={groupId || ''}
              onChange={e => setGroupId(e.target.value)}
              required
              className="w-full border rounded px-3 py-2"
            >
              <option value="">Select a group</option>
              {userGroups.map(g => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </select>
          )}
        </div>

        <div className="flex space-x-2">
          <Button type="submit" disabled={loading}>{loading ? 'Creating...' : 'Create Event'}</Button>
          <Link href="/dashboard/events" className="ml-2 underline">Cancel</Link>
        </div>
      </form>
    </div>
  );
} 