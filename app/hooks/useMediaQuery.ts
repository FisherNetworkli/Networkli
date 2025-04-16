import { useState, useEffect } from 'react';

/**
 * Custom hook for responsive design - detects if the current viewport matches a media query
 * @param query - CSS media query string (e.g. '(min-width: 768px)')
 * @returns boolean indicating if the media query matches
 */
export function useMediaQuery(query: string): boolean {
  // Default to false on the server
  const [matches, setMatches] = useState(false);
  
  useEffect(() => {
    // On client, check if window exists and if the media query matches
    if (typeof window !== 'undefined') {
      const media = window.matchMedia(query);
      
      // Set the initial value
      setMatches(media.matches);
      
      // Define listener function
      const listener = (event: MediaQueryListEvent) => {
        setMatches(event.matches);
      };
      
      // Add the listener
      media.addEventListener('change', listener);
      
      // Clean up listener on unmount
      return () => {
        media.removeEventListener('change', listener);
      };
    }
  }, [query]); // Re-run if query changes
  
  return matches;
} 