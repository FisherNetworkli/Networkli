import React, { useEffect } from 'react';
import { View, StyleSheet, Animated, Dimensions } from 'react-native';
import { Text, Button } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const SCREEN_HEIGHT = Dimensions.get('window').height;

interface MatchNotificationProps {
  matchedProfile: {
    name: string;
    title: string;
    avatar: string;
  };
  onClose: () => void;
  onMessage: () => void;
}

export default function MatchNotification({ 
  matchedProfile, 
  onClose, 
  onMessage 
}: MatchNotificationProps) {
  const slideAnim = React.useRef(new Animated.Value(SCREEN_HEIGHT)).current;
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const handleClose = () => {
    Animated.parallel([
      Animated.timing(slideAnim, {
        toValue: SCREEN_HEIGHT,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start(() => {
      onClose();
    });
  };

  return (
    <Animated.View 
      style={[
        styles.container,
        {
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }],
        },
      ]}
    >
      <View style={styles.content}>
        <MaterialCommunityIcons 
          name="lightning-bolt" 
          size={60} 
          color="#F15B27" 
        />
        <Text style={styles.title}>It's a Match!</Text>
        <Text style={styles.subtitle}>
          You and {matchedProfile.name} have connected
        </Text>
        <View style={styles.profileInfo}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{matchedProfile.avatar}</Text>
          </View>
          <View style={styles.profileText}>
            <Text style={styles.name}>{matchedProfile.name}</Text>
            <Text style={styles.position}>{matchedProfile.title}</Text>
          </View>
        </View>
        <View style={styles.buttons}>
          <Button
            mode="contained"
            onPress={onMessage}
            style={styles.messageButton}
            buttonColor="#F15B27"
          >
            Send Message
          </Button>
          <Button
            mode="outlined"
            onPress={handleClose}
            style={styles.keepButton}
            textColor="#FFFFFF"
          >
            Keep Browsing
          </Button>
        </View>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(54, 89, 168, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  content: {
    alignItems: 'center',
    padding: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginTop: 16,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 32,
  },
  profileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 32,
  },
  avatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#F15B27',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  avatarText: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: 'bold',
  },
  profileText: {
    flex: 1,
  },
  name: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  position: {
    color: '#FFFFFF',
    fontSize: 14,
    opacity: 0.8,
  },
  buttons: {
    width: '100%',
    gap: 12,
  },
  messageButton: {
    width: '100%',
  },
  keepButton: {
    width: '100%',
    borderColor: '#FFFFFF',
  },
}); 