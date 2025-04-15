from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .base_model import BaseModel

class ContentRecommender(BaseModel):
    """Content recommendation model using transformer-based embeddings and attention."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize the content recommender.
        
        Args:
            model_name: Name of the pre-trained transformer model
            embedding_dim: Dimension of content embeddings
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Initialize transformer model for content encoding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Content embedding layers
        self.content_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # User interest embedding
        self.interest_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention for content-user matching
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final scoring layer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Store metadata
        self.metadata.update({
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "dropout": dropout
        })
        
    def encode_content(self, content: str) -> torch.Tensor:
        """Encode content text into embeddings.
        
        Args:
            content: Content text to encode
            
        Returns:
            Content embeddings
        """
        # Tokenize and encode content
        inputs = self.tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.transformer(**inputs)
            content_emb = outputs.last_hidden_state.mean(dim=1)
        
        # Project to hidden dimension
        content_emb = self.content_encoder(content_emb)
        return content_emb
    
    def encode_interests(self, interests: List[str]) -> torch.Tensor:
        """Encode user interests into embeddings.
        
        Args:
            interests: List of interest texts
            
        Returns:
            Interest embeddings
        """
        # Encode each interest
        interest_embs = []
        for interest in interests:
            inputs = self.tokenizer(
                interest,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.transformer(**inputs)
                interest_emb = outputs.last_hidden_state.mean(dim=1)
            interest_embs.append(interest_emb)
        
        # Combine interest embeddings
        interest_emb = torch.stack(interest_embs).mean(dim=0)
        interest_emb = self.interest_encoder(interest_emb)
        return interest_emb
    
    def forward(
        self,
        content: str,
        interests: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            content: Content text
            interests: List of user interests
            
        Returns:
            Tuple of (content embeddings, relevance score)
        """
        # Encode content and interests
        content_emb = self.encode_content(content)
        interest_emb = self.encode_interests(interests)
        
        # Apply attention between content and interests
        attn_output, _ = self.attention(
            content_emb.unsqueeze(0),
            interest_emb.unsqueeze(0),
            interest_emb.unsqueeze(0)
        )
        
        # Concatenate and score
        combined = torch.cat([content_emb, attn_output.squeeze(0)], dim=-1)
        score = self.scorer(combined)
        
        return content_emb, score
    
    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> Dict:
        """Train the content recommender.
        
        Args:
            train_data: List of training examples with content and interests
            val_data: Optional validation data
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training metrics
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            total_loss = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                # Process batch
                batch_scores = []
                batch_labels = []
                
                for example in batch:
                    _, score = self.forward(
                        example["content"],
                        example["interests"]
                    )
                    batch_scores.append(score)
                    batch_labels.append(example["label"])
                
                # Compute loss
                scores = torch.cat(batch_scores)
                labels = torch.tensor(batch_labels, dtype=torch.float32)
                loss = criterion(scores, labels)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Record metrics
            avg_loss = total_loss / (len(train_data) / batch_size)
            metrics["train_loss"].append(avg_loss)
            
            # Validation
            if val_data:
                val_loss, val_acc = self.evaluate(val_data)
                metrics["val_loss"].append(val_loss)
                metrics["val_accuracy"].append(val_acc)
                
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {avg_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {avg_loss:.4f}")
        
        return metrics
    
    def predict(
        self,
        content: str,
        interests: List[str],
        threshold: float = 0.5
    ) -> Tuple[float, bool]:
        """Predict content relevance for a user.
        
        Args:
            content: Content text
            interests: List of user interests
            threshold: Relevance threshold
            
        Returns:
            Tuple of (relevance score, is_relevant)
        """
        self.eval()
        with torch.no_grad():
            _, score = self.forward(content, interests)
            score = score.item()
            is_relevant = score >= threshold
        return score, is_relevant
    
    def evaluate(
        self,
        eval_data: List[Dict]
    ) -> Tuple[float, float]:
        """Evaluate the model on test data.
        
        Args:
            eval_data: List of evaluation examples
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.eval()
        criterion = nn.BCELoss()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for example in eval_data:
                _, score = self.forward(
                    example["content"],
                    example["interests"]
                )
                
                # Compute loss
                label = torch.tensor(example["label"], dtype=torch.float32)
                loss = criterion(score, label)
                total_loss += loss.item()
                
                # Compute accuracy
                pred = (score >= 0.5).float()
                correct += (pred == label).item()
        
        avg_loss = total_loss / len(eval_data)
        accuracy = correct / len(eval_data)
        
        return avg_loss, accuracy 