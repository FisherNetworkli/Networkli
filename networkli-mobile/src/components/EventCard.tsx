import React from 'react'
import { View, Text, Image, StyleSheet } from 'react-native'
import { Database } from '../lib/database.types'

type Event = Database['public']['Tables']['events']['Row'] & {
  topics: string[]
  required_skills: string[]
  match_score?: number
}

export default function EventCard({ event }: { event: Event }) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    })
  }

  return (
    <View style={styles.container}>
      {event.image_url && (
        <View style={styles.imageContainer}>
          <Image
            source={{ uri: event.image_url }}
            style={styles.image}
          />
          {event.match_score && (
            <View style={styles.matchScore}>
              <Text style={styles.matchScoreText}>
                {Math.round(event.match_score * 100)}% Match
              </Text>
            </View>
          )}
        </View>
      )}
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>{event.title}</Text>
          <View style={[
            styles.formatBadge,
            event.format === 'in_person' ? styles.inPersonBadge :
            event.format === 'virtual' ? styles.virtualBadge :
            styles.hybridBadge
          ]}>
            <Text style={[
              styles.formatText,
              event.format === 'in_person' ? styles.inPersonText :
              event.format === 'virtual' ? styles.virtualText :
              styles.hybridText
            ]}>
              {event.format.replace('_', ' ')}
            </Text>
          </View>
        </View>

        <View style={styles.details}>
          <Text style={styles.dateText}>{formatDate(event.date)}</Text>
          {event.location && (
            <Text style={styles.locationText}>{event.location}</Text>
          )}
          {event.max_attendees && (
            <Text style={styles.attendeesText}>
              Max {event.max_attendees} attendees
            </Text>
          )}
        </View>

        {event.description && (
          <Text style={styles.description}>{event.description}</Text>
        )}

        {event.topics && event.topics.length > 0 && (
          <View style={styles.tagsContainer}>
            {event.topics.map((topic) => (
              <View key={topic} style={styles.topicBadge}>
                <Text style={styles.topicText}>{topic}</Text>
              </View>
            ))}
          </View>
        )}

        {event.required_skills && event.required_skills.length > 0 && (
          <View style={styles.tagsContainer}>
            {event.required_skills.map((skill) => (
              <View key={skill} style={styles.skillBadge}>
                <Text style={styles.skillText}>{skill}</Text>
              </View>
            ))}
          </View>
        )}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    overflow: 'hidden',
  },
  imageContainer: {
    height: 200,
    width: '100%',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  matchScore: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: '#4F46E5',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
  },
  matchScoreText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  content: {
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
    flex: 1,
    marginRight: 12,
  },
  formatBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
  },
  inPersonBadge: {
    backgroundColor: '#DEF7EC',
  },
  virtualBadge: {
    backgroundColor: '#E1EFFE',
  },
  hybridBadge: {
    backgroundColor: '#F3E8FF',
  },
  formatText: {
    fontSize: 14,
    fontWeight: '500',
  },
  inPersonText: {
    color: '#046C4E',
  },
  virtualText: {
    color: '#1E429F',
  },
  hybridText: {
    color: '#6B46C1',
  },
  details: {
    marginTop: 12,
  },
  dateText: {
    fontSize: 16,
    color: '#4B5563',
    marginBottom: 4,
  },
  locationText: {
    fontSize: 16,
    color: '#4B5563',
    marginBottom: 4,
  },
  attendeesText: {
    fontSize: 16,
    color: '#4B5563',
  },
  description: {
    marginTop: 12,
    fontSize: 16,
    color: '#4B5563',
    lineHeight: 24,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 12,
    gap: 8,
  },
  topicBadge: {
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
  },
  topicText: {
    color: '#4F46E5',
    fontSize: 14,
  },
  skillBadge: {
    backgroundColor: '#FEF3C7',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
  },
  skillText: {
    color: '#92400E',
    fontSize: 14,
  },
}) 