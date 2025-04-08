import React, { useEffect, useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { Text, Card, Button, Chip, Divider } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { api } from '../services/api';

interface Recommendation {
  id: string;
  name: string;
  title?: string;
  company?: string;
  match_score: number;
  match_reasons: string[];
}

interface Event {
  id: string;
  title: string;
  description: string;
  date: string;
  format: string;
  topics: string[];
  match_score: number;
  match_reasons: string[];
}

interface Group {
  id: string;
  name: string;
  description: string;
  member_count: number;
  focus_areas: string[];
  match_score: number;
  match_reasons: string[];
}

const RecommendationsScreen = () => {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [matches, setMatches] = useState<Recommendation[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);

  const fetchRecommendations = async () => {
    try {
      const [matchesData, eventsData, groupsData] = await Promise.all([
        api.getRecommendations(),
        api.getRecommendedEvents(),
        api.getRecommendedGroups(),
      ]);

      setMatches(matchesData);
      setEvents(eventsData);
      setGroups(groupsData);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    fetchRecommendations();
  }, []);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Professional Matches */}
        <View style={styles.section}>
          <Text variant="titleLarge" style={styles.sectionTitle}>
            Recommended Connections
          </Text>
          {matches.map((match) => (
            <Card key={match.id} style={styles.card}>
              <Card.Content>
                <Text variant="titleMedium">{match.name}</Text>
                {match.title && (
                  <Text variant="bodyMedium">{match.title}</Text>
                )}
                {match.company && (
                  <Text variant="bodyMedium">{match.company}</Text>
                )}
                <View style={styles.matchScore}>
                  <Text variant="bodySmall">
                    Match Score: {Math.round(match.match_score * 100)}%
                  </Text>
                </View>
                <View style={styles.reasons}>
                  {match.match_reasons.map((reason, index) => (
                    <Chip key={index} style={styles.reasonChip}>
                      {reason}
                    </Chip>
                  ))}
                </View>
              </Card.Content>
              <Card.Actions>
                <Button mode="contained">Connect</Button>
                <Button>View Profile</Button>
              </Card.Actions>
            </Card>
          ))}
        </View>

        <Divider style={styles.divider} />

        {/* Professional Events */}
        <View style={styles.section}>
          <Text variant="titleLarge" style={styles.sectionTitle}>
            Recommended Events
          </Text>
          {events.map((event) => (
            <Card key={event.id} style={styles.card}>
              <Card.Content>
                <Text variant="titleMedium">{event.title}</Text>
                <Text variant="bodyMedium">{event.description}</Text>
                <Text variant="bodySmall">Date: {event.date}</Text>
                <Text variant="bodySmall">Format: {event.format}</Text>
                <View style={styles.topics}>
                  {event.topics.map((topic, index) => (
                    <Chip key={index} style={styles.topicChip}>
                      {topic}
                    </Chip>
                  ))}
                </View>
                <View style={styles.reasons}>
                  {event.match_reasons.map((reason, index) => (
                    <Chip key={index} style={styles.reasonChip}>
                      {reason}
                    </Chip>
                  ))}
                </View>
              </Card.Content>
              <Card.Actions>
                <Button mode="contained">Register</Button>
                <Button>Learn More</Button>
              </Card.Actions>
            </Card>
          ))}
        </View>

        <Divider style={styles.divider} />

        {/* Professional Groups */}
        <View style={styles.section}>
          <Text variant="titleLarge" style={styles.sectionTitle}>
            Recommended Groups
          </Text>
          {groups.map((group) => (
            <Card key={group.id} style={styles.card}>
              <Card.Content>
                <Text variant="titleMedium">{group.name}</Text>
                <Text variant="bodyMedium">{group.description}</Text>
                <Text variant="bodySmall">
                  {group.member_count} members
                </Text>
                <View style={styles.topics}>
                  {group.focus_areas.map((area, index) => (
                    <Chip key={index} style={styles.topicChip}>
                      {area}
                    </Chip>
                  ))}
                </View>
                <View style={styles.reasons}>
                  {group.match_reasons.map((reason, index) => (
                    <Chip key={index} style={styles.reasonChip}>
                      {reason}
                    </Chip>
                  ))}
                </View>
              </Card.Content>
              <Card.Actions>
                <Button mode="contained">Join Group</Button>
                <Button>Learn More</Button>
              </Card.Actions>
            </Card>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollView: {
    flex: 1,
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    marginBottom: 16,
  },
  card: {
    marginBottom: 16,
    backgroundColor: '#FFFFFF',
  },
  matchScore: {
    marginTop: 8,
  },
  reasons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  reasonChip: {
    margin: 4,
  },
  topics: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  topicChip: {
    margin: 4,
    backgroundColor: '#E1F5FE',
  },
  divider: {
    marginVertical: 16,
  },
});

export default RecommendationsScreen; 