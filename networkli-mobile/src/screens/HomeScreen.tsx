import React from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  RefreshControl,
} from 'react-native';
import { Text, Card, Button, Avatar, useTheme } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';

// Sample test data
const TEST_EVENTS = [
  {
    id: '1',
    title: 'Tech Networking Mixer',
    date: '2024-04-15T18:00:00',
    location: 'San Francisco, CA',
    format: 'in_person',
    attendees: 42,
    maxAttendees: 100,
    description: 'Join us for an evening of networking with tech professionals.',
  },
  {
    id: '2',
    title: 'Virtual Coffee Chat',
    date: '2024-04-20T09:00:00',
    format: 'virtual',
    attendees: 12,
    maxAttendees: 20,
    description: 'Start your morning with meaningful connections over coffee.',
  },
];

const TEST_CONNECTIONS = [
  {
    id: '1',
    name: 'Sarah Chen',
    title: 'Senior Product Manager at Apple',
    mutual_connections: 15,
    avatar: 'SC',
    skills: ['Product Strategy', 'UX Design', 'Agile'],
  },
  {
    id: '2',
    name: 'Marcus Johnson',
    title: 'Software Engineer at Google',
    mutual_connections: 8,
    avatar: 'MJ',
    skills: ['React Native', 'TypeScript', 'AWS'],
  },
  {
    id: '3',
    name: 'Priya Patel',
    title: 'Data Scientist at Meta',
    mutual_connections: 23,
    avatar: 'PP',
    skills: ['Machine Learning', 'Python', 'Data Analytics'],
  },
];

export default function HomeScreen() {
  const navigation = useNavigation();
  const theme = useTheme();
  const [refreshing, setRefreshing] = React.useState(false);
  const [events, setEvents] = React.useState(TEST_EVENTS);
  const [recommendations, setRecommendations] = React.useState(TEST_CONNECTIONS);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate API refresh
    setTimeout(() => {
      setEvents(TEST_EVENTS);
      setRecommendations(TEST_CONNECTIONS);
      setRefreshing(false);
    }, 1000);
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  return (
    <SafeAreaView style={styles.container} edges={['left', 'right']}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl 
            refreshing={refreshing} 
            onRefresh={onRefresh}
            tintColor="#FFFFFF"
          />
        }
      >
        <View style={styles.header}>
          <Text style={styles.title}>Welcome to networkli</Text>
          <Text style={styles.subtitle}>Your Community Awaits</Text>
        </View>
        
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Upcoming Events</Text>
          {events.length === 0 ? (
            <Card style={styles.card}>
              <Card.Content>
                <Text variant="titleLarge" style={styles.cardTitle}>No Events Yet</Text>
                <Text variant="bodyMedium" style={styles.cardText}>
                  Stay tuned for upcoming networking events!
                </Text>
              </Card.Content>
            </Card>
          ) : (
            events.map((event) => (
              <Card key={event.id} style={styles.card}>
                <Card.Content>
                  <View style={styles.eventHeader}>
                    <View style={styles.eventInfo}>
                      <Text variant="titleLarge" style={styles.cardTitle}>{event.title}</Text>
                      <Text variant="bodyMedium" style={styles.cardText}>
                        {formatDate(event.date)}
                      </Text>
                      {event.location && (
                        <Text variant="bodyMedium" style={styles.cardText}>
                          📍 {event.location}
                        </Text>
                      )}
                    </View>
                    <View style={[
                      styles.formatBadge,
                      event.format === 'virtual' ? styles.virtualBadge : styles.inPersonBadge
                    ]}>
                      <Text style={styles.formatText}>
                        {event.format === 'virtual' ? 'Virtual' : 'In Person'}
                      </Text>
                    </View>
                  </View>
                  <Text variant="bodyMedium" style={[styles.cardText, styles.description]}>
                    {event.description}
                  </Text>
                  <View style={styles.attendeeInfo}>
                    <Text variant="bodySmall" style={styles.cardText}>
                      {event.attendees}/{event.maxAttendees} attending
                    </Text>
                  </View>
                </Card.Content>
                <Card.Actions>
                  <Button 
                    mode="contained" 
                    buttonColor="#F15B27"
                    textColor="white"
                  >
                    Join Event
                  </Button>
                </Card.Actions>
              </Card>
            ))
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recommended Connections</Text>
          {recommendations.length === 0 ? (
            <Card style={styles.card}>
              <Card.Content>
                <Text variant="titleLarge" style={styles.cardTitle}>Finding Connections</Text>
                <Text variant="bodyMedium" style={styles.cardText}>
                  We're discovering the best connections for you...
                </Text>
              </Card.Content>
            </Card>
          ) : (
            recommendations.map((recommendation) => (
              <Card
                key={recommendation.id}
                style={styles.card}
              >
                <Card.Content>
                  <View style={styles.connectionHeader}>
                    <Avatar.Text 
                      size={50} 
                      label={recommendation.avatar}
                      style={[styles.avatar, { backgroundColor: '#F15B27' }]}
                      color="white"
                    />
                    <View style={styles.connectionInfo}>
                      <Text variant="titleMedium" style={styles.cardTitle}>
                        {recommendation.name}
                      </Text>
                      <Text variant="bodyMedium" style={styles.cardText}>
                        {recommendation.title}
                      </Text>
                      <Text variant="bodySmall" style={[styles.cardText, styles.mutualText]}>
                        🤝 {recommendation.mutual_connections} mutual connections
                      </Text>
                    </View>
                  </View>
                  <View style={styles.skillsContainer}>
                    {recommendation.skills.map((skill, index) => (
                      <View key={index} style={styles.skillBadge}>
                        <Text style={styles.skillText}>{skill}</Text>
                      </View>
                    ))}
                  </View>
                </Card.Content>
                <Card.Actions>
                  <Button 
                    mode="contained"
                    buttonColor="#F15B27"
                    textColor="white"
                  >
                    Connect
                  </Button>
                  <Button 
                    mode="outlined"
                    textColor="#F15B27"
                    style={styles.outlinedButton}
                  >
                    View Profile
                  </Button>
                </Card.Actions>
              </Card>
            ))
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#3659A8', // Connection Blue
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 32,
    fontWeight: '900',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: '300',
    color: '#F15B27', // Networkli Orange
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  card: {
    backgroundColor: '#FFFFFF',
    marginBottom: 16,
    borderRadius: 12,
  },
  cardTitle: {
    color: '#3659A8',
    fontWeight: '700',
  },
  cardText: {
    color: '#666666',
    marginTop: 4,
  },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  eventInfo: {
    flex: 1,
    marginRight: 12,
  },
  formatBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  virtualBadge: {
    backgroundColor: '#E1EFFE',
  },
  inPersonBadge: {
    backgroundColor: '#DEF7EC',
  },
  formatText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#3659A8',
  },
  description: {
    marginTop: 12,
    marginBottom: 8,
  },
  attendeeInfo: {
    marginTop: 8,
  },
  connectionHeader: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  avatar: {
    marginRight: 12,
  },
  connectionInfo: {
    flex: 1,
  },
  mutualText: {
    marginTop: 4,
    color: '#3659A8',
  },
  skillsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  skillBadge: {
    backgroundColor: '#FEF3C7',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
  },
  skillText: {
    color: '#92400E',
    fontSize: 12,
    fontWeight: '500',
  },
  outlinedButton: {
    borderColor: '#F15B27',
    marginLeft: 8,
  },
}); 