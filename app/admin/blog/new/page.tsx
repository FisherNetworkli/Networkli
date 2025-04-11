'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function NewBlogPost() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    title: '',
    slug: '',
    excerpt: '',
    content: '',
    category: '',
    author: '',
    image: '',
    readTime: '',
    tags: '',
    published: false
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement blog post creation
    console.log('Creating new post:', formData);
    router.push('/admin/blog');
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value
    }));
  };

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-8">
      <div className="md:flex md:items-center md:justify-between mb-8">
        <div className="min-w-0 flex-1">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:truncate sm:text-3xl sm:tracking-tight">
            Create New Blog Post
          </h2>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6 max-w-3xl">
        <div>
          <label htmlFor="title" className="block text-sm font-medium text-gray-700">
            Title
          </label>
          <input
            type="text"
            name="title"
            id="title"
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.title}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="slug" className="block text-sm font-medium text-gray-700">
            Slug
          </label>
          <input
            type="text"
            name="slug"
            id="slug"
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.slug}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="excerpt" className="block text-sm font-medium text-gray-700">
            Excerpt
          </label>
          <textarea
            name="excerpt"
            id="excerpt"
            rows={3}
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.excerpt}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="content" className="block text-sm font-medium text-gray-700">
            Content
          </label>
          <textarea
            name="content"
            id="content"
            rows={10}
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.content}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="category" className="block text-sm font-medium text-gray-700">
            Category
          </label>
          <input
            type="text"
            name="category"
            id="category"
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.category}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="author" className="block text-sm font-medium text-gray-700">
            Author
          </label>
          <input
            type="text"
            name="author"
            id="author"
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.author}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="image" className="block text-sm font-medium text-gray-700">
            Image URL
          </label>
          <input
            type="text"
            name="image"
            id="image"
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.image}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="readTime" className="block text-sm font-medium text-gray-700">
            Read Time
          </label>
          <input
            type="text"
            name="readTime"
            id="readTime"
            required
            placeholder="e.g., 5 min read"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.readTime}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="tags" className="block text-sm font-medium text-gray-700">
            Tags (comma-separated)
          </label>
          <input
            type="text"
            name="tags"
            id="tags"
            required
            placeholder="e.g., networking, career, technology"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-connection-blue focus:ring-connection-blue sm:text-sm"
            value={formData.tags}
            onChange={handleChange}
          />
        </div>

        <div className="flex items-center">
          <input
            type="checkbox"
            name="published"
            id="published"
            className="h-4 w-4 rounded border-gray-300 text-connection-blue focus:ring-connection-blue"
            checked={formData.published}
            onChange={handleChange}
          />
          <label htmlFor="published" className="ml-2 block text-sm text-gray-900">
            Publish immediately
          </label>
        </div>

        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={() => router.back()}
            className="rounded-md border border-gray-300 bg-white py-2 px-4 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-connection-blue focus:ring-offset-2"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="inline-flex justify-center rounded-md border border-transparent bg-connection-blue py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-connection-blue-dark focus:outline-none focus:ring-2 focus:ring-connection-blue focus:ring-offset-2"
          >
            Create Post
          </button>
        </div>
      </form>
    </div>
  );
} 