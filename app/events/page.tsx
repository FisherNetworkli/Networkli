export const metadata = {
  title: 'Upcoming Events | Networkli',
  description: 'Discover upcoming public events on Networkli to network and connect meaningfully.',
};

import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import Link from 'next/link';
import PublicTrendingEvents from './components/PublicTrendingEvents';

// Define event type and demo fallback data
interface EventItem {
  id: string;
  title: string;
  description: string;
  start_time: string;
  location: string;
  image_url?: string;
  format: string;
}

const demoEvents: EventItem[] = [
  {
    id: '1',
    title: 'React Summit 2024',
    description: "Join the world's largest React community for two days of talks, workshops, and networking with top React core team members.",
    start_time: '2025-08-15T09:00:00.000Z',
    location: 'San Francisco, CA',
    image_url: 'https://images.unsplash.com/photo-1531058020387-3be344556be6?ixlib=rb-4.0.3',
    format: 'Conference',
  },
  {
    id: '2',
    title: 'AI & ML Hands‑On Workshop',
    description: "A full‑day coding workshop where you'll build and deploy a simple ML model using TensorFlow.js and Next.js.",
    start_time: '2025-07-20T13:30:00.000Z',
    location: 'New York, NY',
    image_url: 'https://images.unsplash.com/photo-1518779578993-ec3579fee39f?ixlib=rb-4.0.3',
    format: 'Workshop',
  },
  {
    id: '3',
    title: 'Startup Pitch Night',
    description: 'Pitch your startup to a panel of angel investors and get live feedback in a fast‑paced, demo‑style event.',
    start_time: '2025-09-05T19:00:00.000Z',
    location: 'Austin, TX',
    image_url: 'https://images.unsplash.com/photo-1551829145-eb2a4bb9d85b?ixlib=rb-4.0.3',
    format: 'Networking',
  },
  {
    id: '4',
    title: 'Women in Tech Meetup',
    description: 'An evening meetup to connect, mentor, and celebrate the achievements of women working in technology.',
    start_time: '2025-07-10T17:00:00.000Z',
    location: 'Seattle, WA',
    image_url: 'https://images.unsplash.com/photo-1589571894960-20bbe2828b12?ixlib=rb-4.0.3',
    format: 'Meetup',
  },
  {
    id: '5',
    title: 'Remote Work Best Practices Webinar',
    description: 'A live webinar covering productivity hacks, tools, and routines that help remote teams thrive.',
    start_time: '2025-07-25T12:00:00.000Z',
    location: 'Online',
    image_url: 'https://images.unsplash.com/photo-1588702547923-7093a6c3ba33?ixlib=rb-4.0.3',
    format: 'Webinar',
  },
];

export default async function PublicEventsPage() {
  const supabase = createServerComponentClient({ cookies });
  const { data: { session } } = await supabase.auth.getSession();
  const isAuthenticated = !!session;

  const { data: events } = await supabase
    .from('events')
    .select('id,title,description,start_time,location,image_url,format')
    .eq('is_private', false)
    .order('start_time', { ascending: true });

  const realEvents = events ?? [];
  const eventList = realEvents.length > 0 ? realEvents : demoEvents;
  const trendingEvents = eventList.slice(0, 5);

  return (
    <div>
      {/* Hero Section matching FeaturesPage */}
      <section
        className="relative pt-24 pb-12 bg-cover bg-center text-white"
        style={{
          backgroundImage: "url('https://ctglknfjoryifmpoynjb.supabase.co/storage/v1/object/sign/images/20250419_0741_Vibrant%20Animated%20Event%20Scene_simple_compose_01js74c273fh2bh9nb85076nwc.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InN0b3JhZ2UtdXJsLXNpZ25pbmcta2V5XzEyZTI0ZDgwLTAxNjItNDVmZS04NWE2LTM0NWE0Mjk5MGJlMiJ9.eyJ1cmwiOiJpbWFnZXMvMjAyNTA0MTlfMDc0MV9WaWJyYW50IEFuaW1hdGVkIEV2ZW50IFNjZW5lX3NpbXBsZV9jb21wb3NlXzAxanM3NGMyNzNmaDJiaDluYjg1MDc2bndjLnBuZyIsImlhdCI6MTc0NTA3MDI2OCwiZXhwIjo0ODY3MTM0MjY4fQ.zP75AZwISwis5HbHt8teiNgiF_TurSt64TKJl1_MwcA')"
        }}
      >
        <div className="absolute inset-0 bg-connection-blue/70" />
        <div className="relative max-w-7xl mx-auto px-4">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">Upcoming Events</h1>
          <p className="text-xl text-gray-100 max-w-3xl mx-auto">
            Discover upcoming public events on Networkli to network and connect meaningfully.
          </p>
          <p className="text-lg text-gray-100 max-w-3xl mx-auto mt-4">
            Our AI-driven matchmaking technology ensures you meet professionals who share your passions and goals. Join virtual and in-person events designed to spark authentic connections and accelerate your career.
          </p>
        </div>
      </section>

      {/* Trending Carousel */}
      <div className="bg-white">
        <PublicTrendingEvents events={trendingEvents} isAuthenticated={isAuthenticated} />
      </div>

      {/* Event Directory */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Event Directory</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {eventList.map(evt => (
              <Link
                key={evt.id}
                href={isAuthenticated ? `/events/${evt.id}` : '/signup'}
                className="block bg-white rounded-lg overflow-hidden shadow hover:shadow-md transition"
              >
                {evt.image_url && (
                  <div className="h-48 overflow-hidden">
                    <img
                      src={evt.image_url}
                      alt={evt.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}
                <div className="p-4">
                  <h3 className="text-xl font-semibold mb-1">{evt.title}</h3>
                  <p className="text-sm text-gray-600 mb-2">
                    {new Date(evt.start_time).toLocaleDateString()} &middot; {evt.location}
                  </p>
                  <p className="text-gray-700 line-clamp-3">{evt.description}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* SEO JSON-LD structured data for event list */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "ItemList",
            itemListElement: eventList.map((evt, idx) => ({
              "@type": "Event",
              position: idx + 1,
              name: evt.title,
              startDate: evt.start_time,
              location: { "@type": "Place", name: evt.location },
              image: evt.image_url,
              description: evt.description,
              url: `https://networkli.com/events/${evt.id}`
            })),
          }),
        }}
      />
    </div>
  );
} 