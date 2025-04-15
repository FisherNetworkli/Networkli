from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModel

class FeatureEngineer:
    """Feature engineering pipeline for user and job data."""
    
    def __init__(self,
                 text_model_name: str = "bert-base-uncased",
                 max_text_length: int = 512,
                 n_components: int = 128):
        self.text_model_name = text_model_name
        self.max_text_length = max_text_length
        self.n_components = n_components
        
        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.label_encoders = {}
        
    def process_text(self, texts: List[str]) -> torch.Tensor:
        """
        Process text data using BERT.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of text embeddings
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.text_model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        return embeddings
    
    def process_skills(self, skills: List[List[str]]) -> torch.Tensor:
        """
        Process skill data.
        
        Args:
            skills: List of skill lists
            
        Returns:
            Tensor of skill embeddings
        """
        # Flatten skills and create TF-IDF vectors
        flat_skills = [" ".join(skill_list) for skill_list in skills]
        skill_vectors = self.tfidf.fit_transform(flat_skills)
        
        # Convert to dense tensor
        return torch.from_numpy(skill_vectors.toarray()).float()
    
    def process_categorical(self,
                          data: Dict[str, List[Union[str, int]]],
                          columns: Optional[List[str]] = None) -> torch.Tensor:
        """
        Process categorical data.
        
        Args:
            data: Dictionary of categorical features
            columns: List of columns to process (defaults to all)
            
        Returns:
            Tensor of encoded categorical features
        """
        columns = columns or list(data.keys())
        encoded_features = []
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(data[col])
            
            encoded = self.label_encoders[col].transform(data[col])
            encoded_features.append(encoded)
        
        return torch.from_numpy(np.column_stack(encoded_features)).float()
    
    def process_numerical(self,
                         data: Dict[str, List[float]],
                         columns: Optional[List[str]] = None) -> torch.Tensor:
        """
        Process numerical data.
        
        Args:
            data: Dictionary of numerical features
            columns: List of columns to process (defaults to all)
            
        Returns:
            Tensor of scaled numerical features
        """
        columns = columns or list(data.keys())
        features = np.column_stack([data[col] for col in columns])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        return torch.from_numpy(scaled_features).float()
    
    def reduce_dimensions(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reduce feature dimensions using PCA.
        
        Args:
            features: Input feature tensor
            
        Returns:
            Tensor of reduced features
        """
        # Convert to numpy for PCA
        features_np = features.numpy()
        
        # Apply PCA
        reduced_features = self.pca.fit_transform(features_np)
        
        return torch.from_numpy(reduced_features).float()
    
    def create_graph_features(self,
                            nodes: Dict[str, torch.Tensor],
                            edges: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Create features for graph neural network.
        
        Args:
            nodes: Dictionary of node features
            edges: Optional dictionary of edge features
            
        Returns:
            Dictionary of graph features
        """
        # Combine node features
        node_features = torch.cat(list(nodes.values()), dim=1)
        
        # Create edge features if provided
        if edges is not None:
            edge_features = torch.cat(list(edges.values()), dim=1)
        else:
            edge_features = None
        
        return {
            'node_features': node_features,
            'edge_features': edge_features
        }
    
    def process_user_data(self,
                         user_data: Dict[str, List[Union[str, float, List[str]]]]) -> Dict[str, torch.Tensor]:
        """
        Process user profile data.
        
        Args:
            user_data: Dictionary containing user features
            
        Returns:
            Dictionary of processed features
        """
        features = {}
        
        # Process text fields
        if 'bio' in user_data:
            features['bio'] = self.process_text(user_data['bio'])
        
        # Process skills
        if 'skills' in user_data:
            features['skills'] = self.process_skills(user_data['skills'])
        
        # Process categorical features
        categorical_data = {
            k: v for k, v in user_data.items()
            if k in ['industry', 'role', 'location']
        }
        if categorical_data:
            features['categorical'] = self.process_categorical(categorical_data)
        
        # Process numerical features
        numerical_data = {
            k: v for k, v in user_data.items()
            if k in ['experience_years', 'education_level']
        }
        if numerical_data:
            features['numerical'] = self.process_numerical(numerical_data)
        
        # Combine all features
        combined_features = torch.cat(list(features.values()), dim=1)
        
        # Reduce dimensions
        reduced_features = self.reduce_dimensions(combined_features)
        
        return {
            'features': reduced_features,
            'original_features': features
        }
    
    def process_job_data(self,
                        job_data: Dict[str, List[Union[str, float, List[str]]]]) -> Dict[str, torch.Tensor]:
        """
        Process job posting data.
        
        Args:
            job_data: Dictionary containing job features
            
        Returns:
            Dictionary of processed features
        """
        features = {}
        
        # Process text fields
        if 'description' in job_data:
            features['description'] = self.process_text(job_data['description'])
        
        # Process required skills
        if 'required_skills' in job_data:
            features['required_skills'] = self.process_skills(job_data['required_skills'])
        
        # Process categorical features
        categorical_data = {
            k: v for k, v in job_data.items()
            if k in ['industry', 'job_type', 'location']
        }
        if categorical_data:
            features['categorical'] = self.process_categorical(categorical_data)
        
        # Process numerical features
        numerical_data = {
            k: v for k, v in job_data.items()
            if k in ['experience_required', 'salary_range']
        }
        if numerical_data:
            features['numerical'] = self.process_numerical(numerical_data)
        
        # Combine all features
        combined_features = torch.cat(list(features.values()), dim=1)
        
        # Reduce dimensions
        reduced_features = self.reduce_dimensions(combined_features)
        
        return {
            'features': reduced_features,
            'original_features': features
        } 