# Machine Learning Models for Networkli

This directory contains the machine learning models used for the Networkli recommendation system.

## Model Structure

- `saved_models/`: Serialized model files that are loaded by the prediction service
   - `gnn_model.pt`: Graph Neural Network model for network-based recommendations
   - `network_model.pt`: Simpler network model as fallback

## Training

Models are trained using the scripts in the `app/ml/training` directory. See the training README for details on how to train and update these models.

## Model Descriptions

### GNN Model

The Graph Neural Network model captures user relationships and preferences by modeling users as nodes in a graph. Features:

- Node embeddings that represent users in a low-dimensional space
- Edge predictions for likelihood of professional connections
- Multi-scale feature extraction for capturing different aspects of user profiles

### Network Model

The Network Model is a simpler fallback that doesn't require graph structure. Features:

- Direct encoding of user profile attributes
- Similarity-based matching using neural networks
- Efficient computation for large user bases

## Using These Models

These models are automatically loaded by the `PredictionService` class in the recommendation system. To use them:

```python
from app.ml.services.prediction_service import PredictionService

# Initialize with a Supabase client
prediction_service = PredictionService(supabase_client)

# Get recommendations
recommendations = await prediction_service.get_recommendations(user_id="user123")
``` 