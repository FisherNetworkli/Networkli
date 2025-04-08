import React from 'react'
import { View, Text, Image, StyleSheet } from 'react-native'
import { Database } from '../lib/database.types'

type Profile = Database['public']['Tables']['profiles']['Row']

export default function ProfileCard({ profile }: { profile: Profile }) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        {profile.avatar_url && (
          <View style={styles.avatarContainer}>
            <Image
              source={{ uri: profile.avatar_url }}
              style={styles.avatar}
            />
          </View>
        )}
      </View>
      <View style={styles.content}>
        <Text style={styles.name}>{profile.name || 'Anonymous'}</Text>
        {profile.title && (
          <Text style={styles.title}>{profile.title}</Text>
        )}
        {profile.company && (
          <Text style={styles.company}>{profile.company}</Text>
        )}
        {profile.industry && (
          <View style={styles.industryContainer}>
            <Text style={styles.industry}>{profile.industry}</Text>
          </View>
        )}
        {profile.bio && (
          <Text style={styles.bio}>{profile.bio}</Text>
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
  header: {
    height: 120,
    backgroundColor: '#4F46E5',
  },
  avatarContainer: {
    position: 'absolute',
    bottom: -48,
    left: 16,
    borderWidth: 4,
    borderColor: 'white',
    borderRadius: 48,
  },
  avatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
  },
  content: {
    paddingTop: 56,
    paddingBottom: 16,
    paddingHorizontal: 16,
  },
  name: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
  },
  title: {
    fontSize: 16,
    color: '#4B5563',
    marginTop: 4,
  },
  company: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 2,
  },
  industryContainer: {
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  industry: {
    color: '#4F46E5',
    fontSize: 14,
  },
  bio: {
    marginTop: 16,
    color: '#4B5563',
    fontSize: 14,
    lineHeight: 20,
  },
}) 