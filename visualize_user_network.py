import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict

class UserNetwork:
    def __init__(self):
        self.G = nx.Graph()
        
    def add_user(self, user_id: str, name: str, skills: List[str], 
                 interests: List[str], goals: List[str], values: List[str]):
        """Add a user to the network with their attributes"""
        self.G.add_node(user_id, 
                       name=name,
                       skills=skills,
                       interests=interests,
                       goals=goals,
                       values=values)
    
    def calculate_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users based on their attributes"""
        node1 = self.G.nodes[user1]
        node2 = self.G.nodes[user2]
        
        # Calculate Jaccard similarity for each attribute type
        def jaccard_similarity(set1, set2):
            if not set1 and not set2:
                return 0
            return len(set1.intersection(set2)) / len(set1.union(set2))
        
        # Convert lists to sets for similarity calculation
        skills_sim = jaccard_similarity(set(node1['skills']), set(node2['skills']))
        interests_sim = jaccard_similarity(set(node1['interests']), set(node2['interests']))
        goals_sim = jaccard_similarity(set(node1['goals']), set(node2['goals']))
        values_sim = jaccard_similarity(set(node1['values']), set(node2['values']))
        
        # Weighted average of similarities
        weights = {'skills': 0.3, 'interests': 0.3, 'goals': 0.2, 'values': 0.2}
        total_similarity = (
            skills_sim * weights['skills'] +
            interests_sim * weights['interests'] +
            goals_sim * weights['goals'] +
            values_sim * weights['values']
        )
        
        return total_similarity

    def build_network(self, similarity_threshold: float = 0.3):
        """Build edges between users based on similarity"""
        users = list(self.G.nodes())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                similarity = self.calculate_similarity(users[i], users[j])
                if similarity >= similarity_threshold:
                    self.G.add_edge(users[i], users[j], weight=similarity)

    def visualize(self):
        """Visualize the network with user attributes"""
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(self.G, seed=42)
        
        # Draw edges with weights
        edge_weights = [self.G[u][v]['weight'] * 2 for u, v in self.G.edges()]
        nx.draw_networkx_edges(self.G, pos, width=edge_weights, alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.8)
        
        # Add labels
        labels = {node: self.G.nodes[node]['name'] for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10)
        
        plt.title("User Network Based on Profile Similarity", pad=20, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Example usage with sample data
def main():
    # Create sample users based on your data structure
    network = UserNetwork()
    
    # Add some sample users with varying attributes
    network.add_user(
        "user1",
        "Dan",
        skills=["Python", "JavaScript", "React"],
        interests=["AI", "Web Development"],
        goals=["Full Stack Developer"],
        values=["Innovation", "Learning"]
    )
    
    network.add_user(
        "user2",
        "Alex",
        skills=["Python", "Machine Learning", "Data Science"],
        interests=["AI", "Data Analysis"],
        goals=["Data Scientist"],
        values=["Innovation", "Research"]
    )
    
    network.add_user(
        "user3",
        "Sarah",
        skills=["JavaScript", "React", "Node.js"],
        interests=["Web Development", "UI/UX"],
        goals=["Frontend Developer"],
        values=["Creativity", "User Experience"]
    )
    
    network.add_user(
        "user4",
        "Mike",
        skills=["Python", "Java", "DevOps"],
        interests=["Cloud Computing", "System Architecture"],
        goals=["DevOps Engineer"],
        values=["Efficiency", "Reliability"]
    )
    
    network.add_user(
        "user5",
        "Emma",
        skills=["Python", "Data Science", "Machine Learning"],
        interests=["AI", "Research"],
        goals=["Research Scientist"],
        values=["Innovation", "Discovery"]
    )
    
    # Build the network
    network.build_network(similarity_threshold=0.2)
    
    # Visualize
    network.visualize()
    
    # Print network statistics
    print("\nNetwork Statistics:")
    print(f"Number of users: {network.G.number_of_nodes()}")
    print(f"Number of connections: {network.G.number_of_edges()}")
    print("\nUser Connections:")
    for node in network.G.nodes():
        connections = list(network.G.neighbors(node))
        print(f"{network.G.nodes[node]['name']} is connected to: {', '.join([network.G.nodes[n]['name'] for n in connections])}")

if __name__ == "__main__":
    main() 