"""
Machine learning utility functions.
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Set

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    
    # Normalize vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
    
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    
    # Calculate cosine similarity
    similarity = np.dot(a_normalized, b_normalized)
    
    # Ensure value is in the valid range [0, 1]
    return max(0.0, min(1.0, float(similarity)))

def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """
    Calculate Jaccard similarity (intersection over union) between two sets.
    
    Args:
        set_a: First set
        set_b: Second set
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def interests_similarity(interests_a: List[str], interests_b: List[str]) -> Tuple[float, List[str]]:
    """
    Calculate similarity between two lists of interests.
    
    Args:
        interests_a: First list of interests
        interests_b: Second list of interests
        
    Returns:
        Tuple of (similarity_score, matching_interests)
    """
    if not interests_a or not interests_b:
        return 0.0, []
    
    # Normalize interests (lowercase)
    set_a = {interest.lower() for interest in interests_a}
    set_b = {interest.lower() for interest in interests_b}
    
    # Find matching interests
    matching = set_a.intersection(set_b)
    matching_interests = list(matching)
    
    # Calculate similarity score
    similarity = jaccard_similarity(set_a, set_b)
    
    # Apply weighted scoring based on number of matches
    if len(matching) >= 3:
        similarity = min(1.0, similarity * 1.5)  # Boost for many matches
    
    return similarity, matching_interests

def skills_similarity(skills_a: List[str], skills_b: List[str]) -> Tuple[float, List[str]]:
    """
    Calculate similarity between two lists of skills.
    
    Args:
        skills_a: First list of skills
        skills_b: Second list of skills
        
    Returns:
        Tuple of (similarity_score, matching_skills)
    """
    if not skills_a or not skills_b:
        return 0.0, []
    
    # Normalize skills (lowercase)
    set_a = {skill.lower() for skill in skills_a}
    set_b = {skill.lower() for skill in skills_b}
    
    # Find exact matching skills
    matching = set_a.intersection(set_b)
    
    # Find partial matches (e.g., "Python" and "Python Programming")
    partial_matches = set()
    for skill_a in set_a:
        if skill_a in matching:
            continue  # Already counted as an exact match
        
        for skill_b in set_b:
            if skill_b in matching or skill_b in partial_matches:
                continue  # Already matched
                
            # Check for partial matches
            if (skill_a in skill_b or skill_b in skill_a) and min(len(skill_a), len(skill_b)) > 3:
                partial_matches.add(skill_a)
                break
    
    # Combine matches and convert to original case from the first list
    matching_skills = []
    for skill in skills_a:
        if skill.lower() in matching or skill.lower() in partial_matches:
            matching_skills.append(skill)
    
    # Calculate similarity score
    exact_score = len(matching) / max(len(set_a), len(set_b))
    partial_score = len(partial_matches) / max(len(set_a), len(set_b)) * 0.5  # Partial matches count less
    
    similarity = exact_score + partial_score
    similarity = min(1.0, similarity)  # Cap at 1.0
    
    return similarity, matching_skills

def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove special chars)."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text 