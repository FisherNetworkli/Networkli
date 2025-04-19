'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import Link from 'next/link';
import { Button } from '@/components/ui/button';

// List of group types - extend this list as needed
const groupTypes = [
  'Book Clubs',
  'Language Exchange Groups',
  'Personal Development Circles',
  'Writing Workshops',
  'Public Speaking Clubs',
  'Meditation and Mindfulness Groups',
  'Career Mentorship Networks',
  'Financial Literacy Groups',
  'Coding and Tech Learning Groups',
  'Entrepreneurship Hubs',
  // TODO: add the remaining group types from your spec
];

export default function CreateGroupPage() {
  const router = useRouter();
  const supabase = createClientComponentClient();

  const [name, setName] = useState('');
  // Main category and subcategory states
  const [categories, setCategories] = useState<{id: string; name: string;}[]>([]);
  const [subcategories, setSubcategories] = useState<{id: string; name: string; category_id: string;}[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [selectedSubcategory, setSelectedSubcategory] = useState<string>('');
  const [description, setDescription] = useState('');
  const [location, setLocation] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [tags, setTags] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [questions, setQuestions] = useState<string[]>(['']);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Load categories and subcategories on mount
  useEffect(() => {
    const load = async () => {
      const { data: cats } = await supabase.from('group_categories').select('id,name');
      setCategories(cats || []);
      const { data: subs } = await supabase.from('group_subcategories').select('id,name,category_id');
      setSubcategories(subs || []);
    };
    load();
  }, [supabase]);

  const handleQuestionChange = (index: number, value: string) => {
    const newQuestions = [...questions];
    newQuestions[index] = value;
    setQuestions(newQuestions);
  };

  const addQuestion = () => setQuestions([...questions, '']);
  const removeQuestion = (index: number) => setQuestions(questions.filter((_, i) => i !== index));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const tagsArray = tags.split(',').map(tag => tag.trim()).filter(Boolean);

    try {
      const { data, error } = await supabase
        .from('groups')
        .insert([{ name, description, location, image_url: imageUrl, subcategory_id: selectedSubcategory }])
        .select('id')
        .single();

      if (error) throw error;
      router.push(`/groups/${data.id}`);
    } catch (err: any) {
      console.error('Error creating group:', err);
      setError(err.message || 'Error creating group');
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded shadow">
      <h1 className="text-2xl font-bold mb-4">Create New Group</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && <p className="text-red-500">{error}</p>}

        <div>
          <label className="block text-sm font-medium mb-1">Group Name</label>
          <input type="text" value={name} onChange={e => setName(e.target.value)} required className="w-full border rounded px-3 py-2" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Main Category</label>
          <select value={selectedCategory} onChange={e => { setSelectedCategory(e.target.value); setSelectedSubcategory(''); }} required className="w-full border rounded px-3 py-2">
            <option value="" disabled>Select a category</option>
            {categories.map(cat => (
              <option key={cat.id} value={cat.id}>{cat.name}</option>
            ))}
          </select>
        </div>

        {selectedCategory && (
          <div>
            <label className="block text-sm font-medium mb-1">Sub-Category</label>
            <select value={selectedSubcategory} onChange={e => setSelectedSubcategory(e.target.value)} required className="w-full border rounded px-3 py-2">
              <option value="" disabled>Select a sub-category</option>
              {subcategories.filter(sub => sub.category_id === selectedCategory).map(sub => (
                <option key={sub.id} value={sub.id}>{sub.name}</option>
              ))}
            </select>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium mb-1">Description</label>
          <textarea value={description} onChange={e => setDescription(e.target.value)} required rows={4} className="w-full border rounded px-3 py-2" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Location</label>
          <input type="text" value={location} onChange={e => setLocation(e.target.value)} className="w-full border rounded px-3 py-2" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Image URL</label>
          <input type="url" value={imageUrl} onChange={e => setImageUrl(e.target.value)} className="w-full border rounded px-3 py-2" placeholder="https://example.com/image.jpg" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Tags (comma separated)</label>
          <input type="text" value={tags} onChange={e => setTags(e.target.value)} className="w-full border rounded px-3 py-2" placeholder="networking, tech, career" />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Private Group?</label>
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

        <div className="flex space-x-2">
          <Button type="submit" disabled={loading}>{loading ? 'Creating...' : 'Create Group'}</Button>
          <Link href="/dashboard/groups" className="ml-2 underline">Cancel</Link>
        </div>
      </form>
    </div>
  );
} 