from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseModel

class JobMatcher(BaseModel):
    """Model for matching users with job opportunities based on skills and preferences."""
    
    def __init__(self):
        super().__init__(model_name="job_matcher", version="1.0.0")
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.job_vectors = None
        self.job_data = None
        
    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Train the job matching model.
        
        Args:
            data: DataFrame containing job data with columns:
                - title: Job title
                - description: Job description
                - required_skills: List of required skills
                - preferred_skills: List of preferred skills
                - industry: Industry category
                - location: Job location
        """
        # Combine job details for better matching
        job_texts = (
            data['title'] + ' ' + 
            data['description'] + ' ' + 
            data['required_skills'].apply(lambda x: ' '.join(x)) + ' ' +
            data['preferred_skills'].apply(lambda x: ' '.join(x))
        )
        
        # Create TF-IDF vectors
        self.job_vectors = self.vectorizer.fit_transform(job_texts)
        self.job_data = data
        
        # Update metadata
        self.update_metadata({
            "num_jobs": len(self.job_data),
            "vector_dimensions": self.job_vectors.shape[1]
        })
    
    def predict(self, 
                user_skills: List[str], 
                user_preferences: Dict[str, str],
                top_k: int = 5) -> List[Dict[str, float]]:
        """
        Find matching jobs based on user skills and preferences.
        
        Args:
            user_skills: List of user's skills
            user_preferences: Dictionary containing user preferences:
                - industry: Preferred industry
                - location: Preferred location
                - remote: Whether remote work is preferred
            top_k: Number of matching jobs to return
            
        Returns:
            List of dictionaries containing job details and match scores
        """
        if not self.job_vectors or self.job_data is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Convert user skills to TF-IDF vector
        skill_text = ' '.join(user_skills)
        skill_vector = self.vectorizer.transform([skill_text])
        
        # Calculate skill-based similarity scores
        similarities = cosine_similarity(skill_vector, self.job_vectors).flatten()
        
        # Apply preference filters
        filtered_indices = []
        for i, job in self.job_data.iterrows():
            # Check industry match
            if user_preferences.get('industry') and job['industry'] != user_preferences['industry']:
                continue
                
            # Check location match
            if user_preferences.get('location') and job['location'] != user_preferences['location']:
                continue
                
            # Check remote preference
            if user_preferences.get('remote') and not job.get('remote_available', False):
                continue
                
            filtered_indices.append(i)
        
        if not filtered_indices:
            return []
            
        # Get top k matching jobs from filtered indices
        filtered_similarities = similarities[filtered_indices]
        top_k_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
        
        return [
            {
                "job_id": str(self.job_data.iloc[filtered_indices[i]].name),
                "title": self.job_data.iloc[filtered_indices[i]]['title'],
                "company": self.job_data.iloc[filtered_indices[i]]['company'],
                "match_score": float(filtered_similarities[i]),
                "required_skills_match": len(
                    set(user_skills) & set(self.job_data.iloc[filtered_indices[i]]['required_skills'])
                ) / len(self.job_data.iloc[filtered_indices[i]]['required_skills'])
            }
            for i in top_k_indices
        ]
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            data: DataFrame containing test data with columns:
                - user_skills: List of user skills
                - user_preferences: Dictionary of user preferences
                - applied_jobs: List of jobs the user actually applied to
                
        Returns:
            Dictionary containing evaluation metrics
        """
        total_recommendations = 0
        relevant_recommendations = 0
        
        for _, row in data.iterrows():
            user_skills = row['user_skills']
            user_preferences = row['user_preferences']
            applied_jobs = set(row['applied_jobs'])
            
            recommendations = self.predict(user_skills, user_preferences, top_k=10)
            recommended_job_ids = {r['job_id'] for r in recommendations}
            
            total_recommendations += len(recommendations)
            relevant_recommendations += len(recommended_job_ids & applied_jobs)
        
        precision = relevant_recommendations / total_recommendations if total_recommendations > 0 else 0.0
        
        return {
            "precision": precision,
            "total_recommendations": total_recommendations,
            "relevant_recommendations": relevant_recommendations
        } 