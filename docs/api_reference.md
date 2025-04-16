# Networkli API Reference

This document provides a comprehensive overview of all API endpoints available in the Networkli platform.

## Authentication Endpoints

### User Authentication
- `POST /auth/login` - Authenticate a user
- `POST /auth/register` - Register a new user
- `POST /auth/refresh` - Refresh authentication token
- `POST /auth/logout` - Logout a user

## User Profile Endpoints

### Profile Management
- `GET /profiles/{profile_id}` - Get a user profile by ID
  - Returns full or limited profile view based on user permissions
  - Tracks profile views for analytics

- `PATCH /profiles/{profile_id}` - Update a user profile
  - Updates profile info and regenerates embeddings for matching

- `GET /profiles/mock/{profile_id}` - Get mock profile data (development only)

### Profile Views
- `POST /profiles/{profile_id}/record-view` - Record a profile view
  - Tracks who viewed a profile, when, and from where
  - Stores metadata including source and referrer

- `GET /dashboard/profile-views` - Get history of profile views
  - Premium feature that shows who viewed your profile
  - Returns viewer details and view timestamps

## Dashboard Endpoints

### User Dashboard
- `GET /dashboard` - Get dashboard data for the current user
  - Returns profile stats, network activity, and recommendations
  - Aggregates profile views and other interaction metrics

### Organizer Dashboard
- `GET /organizer/dashboard` - Get dashboard data for organizers
  - Returns event stats and attendance metrics
  - Access restricted to users with organizer role

## Recommendation Endpoints

### Connection Recommendations
- `GET /recommendations/{user_id}` - Get recommendations for a specific user
  - Returns personalized connection recommendations
  - Uses attribute matching algorithm with synergy scores

### Recommendation Tracking
- `POST /recommendations/profile/click` - Track when a user clicks on a recommendation
  - Stores metadata about the recommendation source and ranking
  - Used for ML algorithm improvement

- `POST /recommendations/profile/view` - Track when a user views a profile from recommendations
  - Similar to profile view tracking but with recommendation context

## Interaction Endpoints

### Search
- `GET /search` - Search for users, groups, or events
  - Supports text search and filtering
  - Logs search queries for analytics

### Interaction Logging
- `POST /interactions/log` - Log a user interaction
  - Generic endpoint for tracking various user actions
  - Supports multiple interaction types

- `POST /interactions/recommendation-click` - Log recommendation clicks
  - Tracks when users click on recommended connections

- `POST /interactions/event` - Log event interactions
  - Tracks RSVPs, attendance, and other event-related actions

- `POST /interactions/group` - Log group interactions
  - Tracks join requests, posts, and other group-related actions

## Analytics Endpoints

### User Analytics
- `GET /analytics/interactions` - Get aggregate interaction statistics
  - Provides overview of platform-wide interaction patterns
  - Access restricted to admin users

- `GET /analytics/user/{user_id}/interactions` - Get interaction stats for a specific user
  - Provides personalized analytics for user activity
  - Shows engagement metrics and trends

### Batch Analytics
- `POST /analytics/batch` - Get analytics for multiple users
  - Supports filtering by date range and interaction types
  - Useful for organizers to track engagement

## Network Endpoints

### Connections
- `GET /connections` - Get user connections
  - Returns a list of established connections
  - Includes connection metadata

- `POST /connections` - Create a new connection request
  - Initiates the connection process between users

- `PATCH /connections/{connection_id}` - Update connection status
  - Accept or reject connection requests

## Premium Features

### Premium Access
- `GET /premium-features` - Get available premium features
  - Lists features available to premium users
  - Checks user subscription status

## Utility Endpoints

### Health Check
- `GET /health` - Check API health status
  - Monitors connectivity to database and services
  - Returns degraded status if services are partially available

### Mock Data
- `GET /mock/{endpoint}` - Get mock data for development
  - Simulates API responses for frontend development
  - Only available in development environment

## Database Tables

The API interacts with the following key tables:

### User Data
- `profiles` - User profile information and settings
- `connections` - Connections between users
- `profile_views` - Record of profile view events

### Interaction Data
- `interaction_history` - Comprehensive log of user interactions
- `search_history` - Record of search queries

### Organization Data
- `groups` - Information about networking groups
- `events` - Information about networking events
- `event_attendees` - Record of event attendance

### Recommendation Data
- `recommendations` - Generated user recommendations
- `recommendation_clicks` - Record of clicks on recommendations

## Implementation Status

Currently, we have implemented:
- Authentication endpoints
- Basic profile management
- Profile view tracking
- Dashboard data retrieval
- Interaction logging
- Initial recommendation endpoints
- Simple search functionality

In development:
- Advanced recommendation features
- Swipe-based matching interface
- Analytics dashboards
- ML-powered matching 