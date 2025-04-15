from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseModel

class SkillMatcher(BaseModel):
    """Model for matching users based on skills and interests."""
    
    def __init__(self):
        super().__init__(model_name="skill_matcher", version="1.0.0")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.skill_vectors = None
        self.skill_names = None
        
    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Train the skill matching model.
        
        Args:
            data: DataFrame containing skill data with columns:
                - skill_name: Name of the skill
                - description: Description of the skill
                - category: Category of the skill
        """
        # Combine skill name and description for better matching
        skill_texts = data['skill_name'] + ' ' + data['description']
        
        # Create TF-IDF vectors
        self.skill_vectors = self.vectorizer.fit_transform(skill_texts)
        self.skill_names = data['skill_name'].tolist()
        
        # Update metadata
        self.update_metadata({
            "num_skills": len(self.skill_names),
            "vector_dimensions": self.skill_vectors.shape[1]
        })
    
    def predict(self, skills: List[str], top_k: int = 5) -> List[Dict[str, float]]:
        """
        Find similar skills based on input skills.
        
        Args:
            skills: List of skill names to match against
            top_k: Number of similar skills to return
            
        Returns:
            List of dictionaries containing skill names and similarity scores
        """
        if not self.skill_vectors or not self.skill_names:
            raise ValueError("Model must be trained before making predictions")
            
        # Convert input skills to TF-IDF vectors
        skill_text = ' '.join(skills)
        skill_vector = self.vectorizer.transform([skill_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(skill_vector, self.skill_vectors).flatten()
        
        # Get top k similar skills
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            {
                "skill": self.skill_names[i],
                "similarity": float(similarities[i])
            }
            for i in top_indices
        ]
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            data: DataFrame containing test data with columns:
                - input_skills: List of input skills
                - expected_matches: List of expected matching skills
                
        Returns:
            Dictionary containing evaluation metrics
        """
        # This is a simplified evaluation - in practice, you'd want more sophisticated metrics
        total_matches = 0
        correct_matches = 0
        
        for _, row in data.iterrows():
            input_skills = row['input_skills']
            expected_matches = set(row['expected_matches'])
            
            predictions = self.predict(input_skills, top_k=len(expected_matches))
            predicted_skills = {p['skill'] for p in predictions}
            
            total_matches += len(expected_matches)
            correct_matches += len(expected_matches.intersection(predicted_skills))
        
        accuracy = correct_matches / total_matches if total_matches > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "total_matches": total_matches,
            "correct_matches": correct_matches
        } 