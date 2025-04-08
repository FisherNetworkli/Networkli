"""Bumble API interface for Networkli."""
import httpx
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BumbleInterface:
    """Interface for interacting with Bumble's API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Bumble interface.
        
        Args:
            api_key: Optional API key for Bumble
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://bumble.com/api/v2"
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    def _load_api_key(self) -> str:
        """Load API key from environment or config file."""
        try:
            from dotenv import load_dotenv
            import os
            
            load_dotenv()
            api_key = os.getenv("BUMBLE_API_KEY")
            if api_key:
                return api_key
            
            # Try loading from config file
            config_path = Path("config/bumble_config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    return config.get("api_key", "")
                    
        except Exception as e:
            logger.error(f"Failed to load Bumble API key: {e}")
        
        return ""
    
    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user's Bumble profile.
        
        Args:
            user_id: Bumble user ID
            
        Returns:
            Dictionary containing profile information
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/users/{user_id}/profile"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get Bumble profile: {e}")
            return {}
    
    async def get_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get a user's Bumble connections.
        
        Args:
            user_id: Bumble user ID
            
        Returns:
            List of connection dictionaries
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/users/{user_id}/connections"
            )
            response.raise_for_status()
            return response.json().get("connections", [])
        except Exception as e:
            logger.error(f"Failed to get Bumble connections: {e}")
            return []
    
    async def get_professional_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user's professional information from Bumble.
        
        Args:
            user_id: Bumble user ID
            
        Returns:
            Dictionary containing professional information
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/users/{user_id}/professional"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get professional info: {e}")
            return {}
    
    async def sync_data(self, user_id: str) -> Dict[str, Any]:
        """
        Sync all relevant data for a user from Bumble.
        
        Args:
            user_id: Bumble user ID
            
        Returns:
            Dictionary containing all synced data
        """
        profile = await self.get_profile(user_id)
        connections = await self.get_connections(user_id)
        professional = await self.get_professional_info(user_id)
        
        return {
            "profile": profile,
            "connections": connections,
            "professional": professional,
            "synced_at": datetime.utcnow().isoformat()
        }
    
    def extract_skills(self, professional_info: Dict[str, Any]) -> List[str]:
        """
        Extract skills from professional information.
        
        Args:
            professional_info: Professional information dictionary
            
        Returns:
            List of skills
        """
        skills = set()
        
        # Extract from work experience
        for job in professional_info.get("experience", []):
            skills.update(job.get("skills", []))
        
        # Extract from education
        for edu in professional_info.get("education", []):
            skills.update(edu.get("fields_of_study", []))
        
        return list(skills)
    
    def extract_interests(self, profile: Dict[str, Any]) -> List[str]:
        """
        Extract professional interests from profile.
        
        Args:
            profile: Profile dictionary
            
        Returns:
            List of interests
        """
        interests = set()
        
        # Extract from interests section
        interests.update(profile.get("interests", []))
        
        # Extract from about section (if tagged)
        about = profile.get("about", "")
        if about:
            # Extract hashtags as interests
            interests.update(
                tag.strip("#")
                for tag in about.split()
                if tag.startswith("#")
            )
        
        return list(interests)
    
    def format_for_networkli(self, 
                           user_id: str,
                           synced_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Bumble data for Networkli's system.
        
        Args:
            user_id: User ID
            synced_data: Data synced from Bumble
            
        Returns:
            Formatted data for Networkli
        """
        profile = synced_data["profile"]
        professional = synced_data["professional"]
        
        return {
            "user_id": user_id,
            "source": "bumble",
            "name": profile.get("name", ""),
            "skills": self.extract_skills(professional),
            "interests": self.extract_interests(profile),
            "experience": professional.get("experience", []),
            "education": professional.get("education", []),
            "connections": [
                conn["user_id"]
                for conn in synced_data["connections"]
            ],
            "last_synced": synced_data["synced_at"]
        } 