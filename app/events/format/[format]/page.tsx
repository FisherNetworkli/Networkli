import { Metadata } from 'next';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import Link from 'next/link';

// Generate dynamic SEO metadata per format
export async function generateMetadata({ params }: { params: { format: string } }): Promise<Metadata> {
  const raw = params.format;
  const human = raw.replace(/-/g, ' ')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
  return {
    title: `${human} Events | Networkli`,
    description: `Discover upcoming ${human.toLowerCase()} events on Networkli to network and connect meaningfully.`,
  };
}

export default async function FormatEventsPage({ params }: { params: { format: string } }) {
  const supabase = createServerComponentClient({ cookies });
  const rawFormat = params.format;
  const humanFormat = rawFormat.replace(/-/g, ' ');

  const { data: events } = await supabase
    .from('events')
    .select('id,title,description,start_time,location,image_url')
    .eq('format', humanFormat)
    .eq('is_private', false)
    .order('start_time', { ascending: true });

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">{humanFormat} Events</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {events?.map(evt => (
          <Link
            href={`/events/${evt.id}`}
            key={evt.id}
            className="block bg-white rounded-lg overflow-hidden shadow hover:shadow-md transition"
          >
            {evt.image_url && (
              <div className="h-48 w-full overflow-hidden">
                <img src={evt.image_url} alt={evt.title} className="w-full h-full object-cover" />
              </div>
            )}
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-1">{evt.title}</h3>
              <p className="text-sm text-gray-600 mb-2">{new Date(evt.start_time).toLocaleDateString()}</p>
              <p className="text-gray-700 line-clamp-3">{evt.description}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
} 