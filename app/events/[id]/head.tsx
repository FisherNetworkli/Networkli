import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

export default async function Head({ params }: { params: { id: string } }) {
  const supabase = createServerComponentClient({ cookies });
  const { data: event } = await supabase
    .from('events')
    .select('*')
    .eq('id', params.id)
    .single();
  if (!event) {
    return <></>;
  }

  // Breadcrumb structured data
  const breadcrumbList = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Home", item: "https://networkli.com" },
      { "@type": "ListItem", position: 2, name: "Events", item: "https://networkli.com/events" },
      { "@type": "ListItem", position: 3, name: event.title, item: `https://networkli.com/events/${event.id}` }
    ]
  };

  // Event structured data
  const eventData = {
    "@context": "https://schema.org",
    "@type": "Event",
    name: event.title,
    startDate: event.start_time,
    endDate: event.end_time,
    location: { "@type": "Place", name: event.location },
    image: event.image_url,
    description: event.description,
    url: `https://networkli.com/events/${event.id}`,
    offers: {
      "@type": "Offer",
      url: `https://networkli.com/events/${event.id}`,
      price: event.price ?? "0",
      priceCurrency: event.price ? "USD" : undefined,
      availability: event.max_attendees != null
        ? (event.attendee_count < event.max_attendees
            ? "http://schema.org/InStock"
            : "http://schema.org/SoldOut")
        : undefined
    }
  };

  return (
    <>
      <title>{event.title} | Networkli</title>
      <meta name="description" content={event.description} />
      <meta property="og:title" content={event.title} />
      <meta property="og:description" content={event.description} />
      {event.image_url && <meta property="og:image" content={event.image_url} />}
      <meta property="og:url" content={`https://networkli.com/events/${event.id}`} />
      <meta property="og:type" content="website" />
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={event.title} />
      <meta name="twitter:description" content={event.description} />
      {event.image_url && <meta name="twitter:image" content={event.image_url} />}
      <link rel="canonical" href={`https://networkli.com/events/${event.id}`} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbList) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(eventData) }} />
    </>
  );
} 