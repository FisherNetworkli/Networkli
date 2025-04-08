import React, { useState, useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Button } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import SwipeCard from '../components/SwipeCard';
import MatchNotification from '../components/MatchNotification';
import { supabase } from '../lib/supabase/client';
import { 
  getPotentialConnections, 
  recordSwipe, 
  subscribeToNewMatches,
  PotentialConnection 
} from '../lib/supabase/connections';

export default function SwipeScreen() {
  const [profiles, setProfiles] = useState<PotentialConnection[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [showMatch, setShowMatch] = useState(false);
  const [matchedProfile, setMatchedProfile] = useState<{
    name: string;
    title: string;
    avatar: string;
  } | null>(null);

  useEffect(() => {
    loadProfiles();
    
    // Subscribe to new matches
    const subscription = subscribeToNewMatches((match) => {
      // Determine which user is the match (not the current user)
      const currentUserId = supabase.auth.getUser()?.data.user?.id;
      const matchedUser = match.user1.id === currentUserId ? match.user2 : match.user1;
      
      setMatchedProfile({
        name: matchedUser.name,
        title: matchedUser.title,
        avatar: matchedUser.avatar,
      });
      setShowMatch(true);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const loadProfiles = async () => {
    try {
      setIsLoading(true);
      const potentialConnections = await getPotentialConnections(10);
      setProfiles(potentialConnections);
      setCurrentIndex(0);
    } catch (error) {
      console.error('Error loading profiles:', error);
      // Handle error appropriately
    } finally {
      setIsLoading(false);
    }
  };

  const handleSwipeRight = async (profileId: string) => {
    try {
      await recordSwipe(profileId, 'right');
      moveToNextProfile();
    } catch (error) {
      console.error('Error recording right swipe:', error);
      // Handle error appropriately
    }
  };

  const handleSwipeLeft = async (profileId: string) => {
    try {
      await recordSwipe(profileId, 'left');
      moveToNextProfile();
    } catch (error) {
      console.error('Error recording left swipe:', error);
      // Handle error appropriately
    }
  };

  const moveToNextProfile = () => {
    setCurrentIndex(prevIndex => prevIndex + 1);
  };

  const handleMessage = () => {
    // Navigate to chat screen
    // navigation.navigate('Chat', { profile: matchedProfile });
    setShowMatch(false);
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.emptyStateContainer}>
          <Text style={styles.emptyStateTitle}>Loading profiles...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (currentIndex >= profiles.length) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.emptyStateContainer}>
          <Text style={styles.emptyStateTitle}>No More Profiles</Text>
          <Text style={styles.emptyStateText}>
            You've viewed all available profiles for now.
          </Text>
          <Button
            mode="contained"
            onPress={loadProfiles}
            style={styles.resetButton}
            buttonColor="#F15B27"
          >
            Refresh Profiles
          </Button>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {showMatch && matchedProfile && (
        <MatchNotification
          matchedProfile={matchedProfile}
          onClose={() => setShowMatch(false)}
          onMessage={handleMessage}
        />
      )}
      {profiles.map((profile, index) => {
        if (index < currentIndex) return null;
        if (index === currentIndex) {
          return (
            <SwipeCard
              key={profile.id}
              profile={profile}
              onSwipeRight={() => handleSwipeRight(profile.id)}
              onSwipeLeft={() => handleSwipeLeft(profile.id)}
            />
          );
        }
        // Show next card peeking from behind
        return (
          <View
            key={profile.id}
            style={[styles.container, { transform: [{ scale: 0.95 }] }]}
          >
            <SwipeCard
              profile={profile}
              onSwipeRight={() => handleSwipeRight(profile.id)}
              onSwipeLeft={() => handleSwipeLeft(profile.id)}
            />
          </View>
        );
      })}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#3659A8',
  },
  emptyStateContainer: {
    alignItems: 'center',
    padding: 32,
  },
  emptyStateTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 24,
  },
  resetButton: {
    paddingHorizontal: 32,
  },
}); 