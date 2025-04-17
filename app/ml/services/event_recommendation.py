"""
Event Recommendation Service

This module provides algorithms for recommending events to users based on 
their profile, interests, past event attendance, and social connections.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from supabase import Client

logger = logging.getLogger(__name__)

class EventRecommendationService:
    """Service for generating event recommendations."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the event recommendation service.
        
        Args:
            supabase_client: Supabase client for database access
        """
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
    
    async def get_event_recommendations(
        self, 
        user_id: str, 
        limit: int = 10, 
        exclude_registered: bool = True,
        include_reason: bool = True,
        include_past_events: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get event recommendations for a user based on their profile, interests, and past activity.
        
        Args:
            user_id: The ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            exclude_registered: Whether to exclude events the user is already registered for
            include_reason: Whether to include a reason for each recommendation
            include_past_events: Whether to include events that have already happened
            
        Returns:
            List of recommended events with relevance scores and reasons
        """
        try:
            # Step 1: Get the user's profile
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                self.logger.error(f"User profile not found for user_id: {user_id}")
                return []
            
            # Step 2: Get the user's interests
            user_interests = await self._get_user_interests(user_id)
            
            # Step 3: Get the user's past event attendance (for determining preferences)
            user_events = await self._get_user_event_history(user_id)
            
            # Step 4: If excluding registered events, get events the user is already registered for
            excluded_event_ids = set()
            if exclude_registered:
                registered_events = await self._get_registered_events(user_id)
                excluded_event_ids.update(registered_events)
            
            # Step 5: Get candidate events (exclude registered events if requested)
            candidate_events = await self._get_candidate_events(excluded_event_ids, include_past_events)
            
            # Step 6: Get events that user's connections are attending
            connection_events = await self._get_connection_attending_events(user_id)
            
            # Step 7: Calculate relevance scores for each event
            recommendations = []
            for event in candidate_events:
                event_id = event.get('id')
                
                # Calculate relevance and gather matching attributes
                relevance_score, matching_attributes = self._calculate_event_relevance(
                    user_profile=user_profile,
                    user_interests=user_interests,
                    user_events=user_events,
                    event=event,
                    connection_events=connection_events
                )
                
                # Add recommendation with score and matching attributes
                recommendation = {
                    **event,
                    "relevance_score": relevance_score
                }
                
                # Add recommendation reason if requested
                if include_reason:
                    recommendation["reason"] = self._generate_event_recommendation_reason(matching_attributes)
                
                recommendations.append(recommendation)
            
            # Sort by relevance score and return top recommendations
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting event recommendations: {e}", exc_info=True)
            return []
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile from the database."""
        try:
            result = await self.supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            return result.data
        except Exception as e:
            self.logger.error(f"Exception retrieving profile for {user_id}: {e}")
            return None
    
    async def _get_user_interests(self, user_id: str) -> List[str]:
        """Get a user's interests from the database."""
        try:
            result = await self.supabase.table("user_preferences").select("interests").eq("user_id", user_id).single().execute()
            if result.data and "interests" in result.data:
                return result.data["interests"] or []
            return []
        except Exception as e:
            self.logger.error(f"Exception retrieving interests for {user_id}: {e}")
            return []
    
    async def _get_user_event_history(self, user_id: str) -> Dict[str, Any]:
        """Get a user's event attendance history to determine preferences."""
        try:
            # Get events the user has attended
            attendance_result = await self.supabase.table("event_attendance").select("event_id").eq("user_id", user_id).execute()
            
            # Get details for those events to extract categories, types, etc.
            event_ids = [item.get("event_id") for item in attendance_result.data if item.get("event_id")]
            
            if not event_ids:
                return {"attended_events": [], "event_categories": {}, "event_formats": {}}
            
            events_result = await self.supabase.table("events").select("*").in_("id", event_ids).execute()
            
            # Analyze preferences
            event_categories = {}
            event_formats = {}
            
            for event in events_result.data:
                # Count category preferences
                category = event.get("category")
                if category:
                    event_categories[category] = event_categories.get(category, 0) + 1
                
                # Count format preferences (online, in-person, hybrid)
                event_format = event.get("format")
                if event_format:
                    event_formats[event_format] = event_formats.get(event_format, 0) + 1
            
            return {
                "attended_events": event_ids,
                "event_categories": event_categories,
                "event_formats": event_formats
            }
            
        except Exception as e:
            self.logger.error(f"Exception retrieving event history for {user_id}: {e}")
            return {"attended_events": [], "event_categories": {}, "event_formats": {}}
    
    async def _get_registered_events(self, user_id: str) -> Set[str]:
        """Get IDs of events that the user is already registered for."""
        try:
            result = await self.supabase.table("event_attendance").select("event_id").eq("user_id", user_id).execute()
            return {item.get("event_id") for item in result.data if item.get("event_id")}
        except Exception as e:
            self.logger.error(f"Exception retrieving registered events for {user_id}: {e}")
            return set()
    
    async def _get_candidate_events(self, exclude_ids: Set[str], include_past_events: bool) -> List[Dict[str, Any]]:
        """Get candidate events for recommendations, optionally excluding past events."""
        try:
            query = self.supabase.table("events").select("*")
            
            # Filter out events that have already happened
            if not include_past_events:
                now = datetime.now().isoformat()
                query = query.gt("end_time", now)
            
            result = await query.execute()
            
            # Filter out excluded events
            return [event for event in result.data if event.get("id") not in exclude_ids]
        except Exception as e:
            self.logger.error(f"Exception retrieving candidate events: {e}")
            return []
    
    async def _get_connection_attending_events(self, user_id: str) -> Dict[str, int]:
        """Get events that user's connections are attending, with count of connections at each event."""
        try:
            # Get user's connections
            connections_result = await self.supabase.table("connections").select("receiver_id, requester_id").or_(f"requester_id.eq.{user_id},receiver_id.eq.{user_id}").eq("status", "accepted").execute()
            
            if not connections_result.data:
                return {}
            
            # Extract connection IDs
            connection_ids = set()
            for conn in connections_result.data:
                if conn.get("requester_id") == user_id:
                    connection_ids.add(conn.get("receiver_id"))
                else:
                    connection_ids.add(conn.get("requester_id"))
            
            if not connection_ids:
                return {}
            
            # Get events attended by connections
            events_result = await self.supabase.table("event_attendance").select("event_id").in_("user_id", list(connection_ids)).execute()
            
            # Count occurrences of each event
            event_counts = {}
            for item in events_result.data:
                event_id = item.get("event_id")
                if event_id:
                    event_counts[event_id] = event_counts.get(event_id, 0) + 1
            
            return event_counts
        except Exception as e:
            self.logger.error(f"Exception retrieving connection events for {user_id}: {e}")
            return {}
    
    def _calculate_event_relevance(
        self,
        user_profile: Dict[str, Any],
        user_interests: List[str],
        user_events: Dict[str, Any],
        event: Dict[str, Any],
        connection_events: Dict[str, int]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate relevance score between a user and an event.
        
        Returns:
            Tuple of (relevance_score, matching_attributes)
        """
        # Initialize weights for different attributes
        weights = {
            "category_match": 0.25,
            "interest_match": 0.20,
            "connection_match": 0.20,
            "location_match": 0.15,
            "format_match": 0.10,
            "proximity_match": 0.10
        }
        
        # Initialize score and matching attributes
        score = 0.0
        matching_attributes = {}
        
        # 1. Category match (event category vs. user's past attendance)
        event_category = event.get("category", "").lower()
        if event_category and event_category in user_events["event_categories"]:
            category_frequency = user_events["event_categories"][event_category]
            # Higher weight if user frequently attends this category
            score += min(category_frequency * 0.05, weights["category_match"])
            matching_attributes["category"] = event_category
        
        # 2. Interest match (event topics vs user interests)
        event_topics = event.get("topics", [])
        if isinstance(event_topics, str):
            # Handle case where topics might be stored as comma-separated string
            event_topics = [topic.strip() for topic in event_topics.split(",")]
        
        matching_interests = []
        for interest in user_interests:
            if any(topic.lower() == interest.lower() for topic in event_topics):
                matching_interests.append(interest)
        
        if matching_interests:
            interest_score = min(len(matching_interests) * 0.1, weights["interest_match"])
            score += interest_score
            matching_attributes["interests"] = matching_interests
        
        # 3. Connection match (how many connections are attending)
        event_id = event.get("id")
        if event_id in connection_events:
            connection_count = connection_events[event_id]
            # Scale by number of connections, up to max weight
            connection_score = min(connection_count * 0.05, weights["connection_match"])
            score += connection_score
            matching_attributes["connection_count"] = connection_count
        
        # 4. Location match
        user_location = user_profile.get("location", "").lower()
        event_location = event.get("location", "").lower()
        
        if user_location and event_location and (
            user_location == event_location or
            user_location in event_location or
            event_location in user_location
        ):
            score += weights["location_match"]
            matching_attributes["location"] = event_location
        
        # 5. Format match (online, in-person, hybrid)
        event_format = event.get("format", "").lower()
        if event_format and event_format in user_events["event_formats"]:
            format_frequency = user_events["event_formats"][event_format]
            # Higher weight if user frequently attends this format
            score += min(format_frequency * 0.05, weights["format_match"])
            matching_attributes["format"] = event_format
        
        # 6. Time proximity (upcoming events get higher scores)
        now = datetime.now()
        event_start = event.get("start_time")
        if event_start:
            try:
                # Convert to datetime if string
                if isinstance(event_start, str):
                    event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
                
                # Calculate days until event
                days_until = (event_start - now).days
                
                # Higher score for events happening soon (within next 7 days)
                if 0 <= days_until <= 7:
                    proximity_score = weights["proximity_match"] * (1 - (days_until / 7))
                    score += proximity_score
                    matching_attributes["days_until"] = days_until
            except Exception as e:
                self.logger.error(f"Error calculating event proximity: {e}")
        
        return score, matching_attributes
    
    def _generate_event_recommendation_reason(self, matching_attributes: Dict[str, Any]) -> str:
        """Generate a human-readable reason for an event recommendation."""
        if not matching_attributes:
            return "This event might be interesting for your professional development."
        
        reasons = []
        
        # Category match
        if "category" in matching_attributes:
            reasons.append(f"This event is about {matching_attributes['category']}, which you've shown interest in")
        
        # Interest matches
        if "interests" in matching_attributes and matching_attributes["interests"]:
            if len(matching_attributes["interests"]) == 1:
                reasons.append(f"The event covers {matching_attributes['interests'][0]}, which matches your interests")
            else:
                interests_text = ", ".join(matching_attributes["interests"][:2])
                reasons.append(f"The event includes topics like {interests_text} that align with your interests")
        
        # Connection matches
        if "connection_count" in matching_attributes:
            count = matching_attributes["connection_count"]
            if count == 1:
                reasons.append("One of your connections is attending this event")
            else:
                reasons.append(f"{count} of your connections are attending this event")
        
        # Location match
        if "location" in matching_attributes:
            reasons.append(f"This event is located in {matching_attributes['location']}, near you")
        
        # Format match
        if "format" in matching_attributes:
            format_name = matching_attributes["format"]
            if format_name == "online":
                reasons.append("This is an online event, which you've attended in the past")
            elif format_name == "in-person":
                reasons.append("This is an in-person event, which you've attended in the past")
            elif format_name == "hybrid":
                reasons.append("This is a hybrid event, offering both in-person and online attendance")
        
        # Proximity match
        if "days_until" in matching_attributes:
            days = matching_attributes["days_until"]
            if days == 0:
                reasons.append("This event is happening today")
            elif days == 1:
                reasons.append("This event is happening tomorrow")
            else:
                reasons.append(f"This event is happening in {days} days")
        
        # Build the final reason string
        if reasons:
            return reasons[0] + (f". {reasons[1]}" if len(reasons) > 1 else "")
        else:
            return "This event might be interesting for your professional development."

# Factory function for consistent initialization
def create_event_recommendation_service(supabase_client: Client) -> EventRecommendationService:
    """Create and return an EventRecommendationService instance."""
    return EventRecommendationService(supabase_client) 