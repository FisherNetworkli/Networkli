import { NextResponse } from 'next/server';

// Simple ZIP code to city mapping for demo purposes
// In production, this should be replaced with a proper geocoding service
const zipCodeData: Record<string, {
  name: string;  // city name
  adminCode1: string;  // state code
  adminName1: string;  // state name
  adminName2: string;  // county
  postalcode: string;
  lat: string;
  lng: string;
  countryCode: string;
}> = {
  '94105': {
    name: 'San Francisco',
    adminCode1: 'CA',
    adminName1: 'California',
    adminName2: 'San Francisco County',
    postalcode: '94105',
    lat: '37.7858',
    lng: '-122.3968',
    countryCode: 'US'
  },
  '10001': {
    name: 'New York',
    adminCode1: 'NY',
    adminName1: 'New York',
    adminName2: 'New York County',
    postalcode: '10001',
    lat: '40.7506',
    lng: '-73.9971',
    countryCode: 'US'
  },
  '60601': {
    name: 'Chicago',
    adminCode1: 'IL',
    adminName1: 'Illinois',
    adminName2: 'Cook County',
    postalcode: '60601',
    lat: '41.8857',
    lng: '-87.6229',
    countryCode: 'US'
  },
  '80504': {
    name: 'Longmont',
    adminCode1: 'CO',
    adminName1: 'Colorado',
    adminName2: 'Weld County',
    postalcode: '80504',
    lat: '40.1389',
    lng: '-104.9811',
    countryCode: 'US'
  }
};

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const query = searchParams.get('query');

    if (!query) {
      return NextResponse.json(
        { error: 'Query parameter is required' },
        { status: 400 }
      );
    }

    // Use Geonames API with your username
    const geonamesUrl = `http://api.geonames.org/postalCodeLookupJSON?postalcode=${query}&country=US&username=fbeats13`;
    
    const response = await fetch(geonamesUrl);
    const data = await response.json();

    if (!response.ok) {
      console.error('Geonames API error:', data);
      return NextResponse.json(
        { error: 'Failed to fetch location data' },
        { status: 500 }
      );
    }

    if (!data.postalcodes || data.postalcodes.length === 0) {
      return NextResponse.json(
        { error: 'Location not found' },
        { status: 404 }
      );
    }

    const location = data.postalcodes[0];
    
    // Format the response to match our expected structure
    return NextResponse.json({
      name: location.placeName,
      adminCode1: location.adminCode1,
      adminName1: location.adminName1,
      adminName2: location.adminName2 || `${location.placeName} County`,
      postalcode: location.postalcode,
      lat: location.lat,
      lng: location.lng,
      countryCode: location.countryCode
    });

  } catch (error) {
    console.error('Error in cities API:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 