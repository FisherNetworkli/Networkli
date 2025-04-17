# Recommendation Algorithms

This document outlines the recommendation algorithms used in Networkli to suggest users, groups, and events based on user preferences and behaviors.

## User Recommendations

### Simple Recommendation Algorithm

The simple recommendation algorithm uses attribute matching to find users with similar profiles:

1. **Attribute Matching**: Compares user profiles based on:
   - Skills
   - Interests
   - Industry
   - Location

2. **Scoring Method**:
   - Each shared attribute contributes to the match score
   - Weighted by importance (skills and interests typically weighted higher than location)
   - Normalized to a scale of 0-1 for consistent comparison

3. **Advantages**:
   - Simple to implement and understand
   - Fast execution
   - Transparent matching reasons

4. **Limitations**:
   - Doesn't capture complex relationships
   - May miss non-obvious matches
   - Limited to explicit attributes

### ML-Based Recommendation Algorithm

The ML-based algorithm uses a Graph Neural Network (GNN) to capture complex relationships:

1. **Graph Representation**:
   - Users, groups, and events represented as nodes
   - Connections, memberships, and attendances as edges
   - Node features include profile attributes

2. **Training Process**:
   - Trained on user interaction data
   - Learns to predict successful connections
   - Optimized for engagement metrics

3. **Inference**:
   - Generates embeddings for each user
   - Calculates similarity between embeddings
   - Ranks potential connections

4. **Advantages**:
   - Captures implicit relationships
   - Learns from user behavior
   - Can discover non-obvious matches

5. **Limitations**:
   - Requires sufficient training data
   - More complex to implement and maintain
   - Less interpretable matching reasons

## Group and Event Aligned Member Recommendations

The Group and Event Aligned Member algorithm identifies potential connections among members of the same group or attendees of the same event:

1. **Membership Verification**:
   - Confirms users are members of the same group or event
   - Excludes already connected members

2. **Attribute Comparison**:
   - Identifies shared skills and interests
   - Calculates a match score based on overlap
   - Generates human-readable match reasons

3. **Scoring Method**:
   - Match score = (shared skills + shared interests) / max possible matches
   - Normalized to 0-1 scale
   - Sorted by descending match score

4. **Match Reasons**:
   - Generates personalized explanations for matches
   - Highlights shared skills and interests
   - Fallback to group/event membership if no specific matches

5. **Implementation**:
   - API endpoint: `/api/recommendations/[entityType]/[entityId]/members`
   - Parameters: userId, limit
   - Returns ranked recommendations with match reasons

## Implementation Details

### Backend Components

1. **API Endpoints**:
   - `/api/recommendations` - General user recommendations
   - `/api/recommendations/[entityType]/[entityId]/members` - Group/event member recommendations

2. **Database Queries**:
   - Profiles table for user attributes
   - Connections table to exclude existing connections
   - Group members and event attendance for membership verification

3. **Scoring and Ranking**:
   - Calculated server-side
   - Sorted by match score
   - Paginated for performance

### Frontend Components

1. **RecommendationsList Component**:
   - Displays general recommendations

2. **AlignedMembersList Component**:
   - Displays group or event specific recommendations
   - Parameters: userId, entityType, entityId
   - Features:
     - Loading states
     - Error handling
     - Pagination support
     - Connection request functionality

## Evaluation Metrics

The recommendation algorithms are evaluated using the following metrics:

1. **Precision**: Percentage of recommendations that result in connections
2. **Recall**: Percentage of potential connections captured by recommendations
3. **Diversity**: Variety in recommendation attributes
4. **Explanation Quality**: Usefulness of match reasons
5. **User Engagement**: Click-through and connection request rates

## Future Improvements

1. **Hybrid Approach**:
   - Combine simple and ML-based algorithms
   - Weighted ensemble for better performance

2. **Contextual Recommendations**:
   - Consider user activity context
   - Time-based recommendations

3. **Feedback Loop**:
   - Incorporate user feedback on recommendations
   - A/B testing of algorithm variations

4. **Advanced Features**:
   - Collaborative filtering components
   - Interest clustering
   - Activity-based recommendations 