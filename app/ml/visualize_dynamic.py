import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Optional
import pandas as pd

class DynamicNetworkVisualizer:
    def __init__(self):
        self.G = nx.Graph()
        self.pos = None
        self.fig = None
        
    def add_user(self, user_id: str, data: Dict):
        """Add a user with their attributes to the network"""
        self.G.add_node(user_id, **data)
    
    def calculate_similarity_by_category(self, user1: str, user2: str, categories: List[str]) -> float:
        """Calculate similarity between users based on specified categories"""
        similarities = []
        for category in categories:
            set1 = set(self.G.nodes[user1].get(category, []))
            set2 = set(self.G.nodes[user2].get(category, []))
            if set1 or set2:
                sim = len(set1.intersection(set2)) / len(set1.union(set2)) if set1 or set2 else 0
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0
    
    def update_edges(self, categories: List[str], threshold: float = 0.2):
        """Update network edges based on selected categories"""
        self.G.clear_edges()
        nodes = list(self.G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self.calculate_similarity_by_category(nodes[i], nodes[j], categories)
                if similarity >= threshold:
                    self.G.add_edge(nodes[i], nodes[j], weight=similarity)
    
    def create_interactive_visualization(self, categories: List[str] = None):
        """Create an interactive visualization with Plotly"""
        if categories is None:
            categories = ['skills', 'interests', 'professionalGoals', 'values']
            
        if not self.pos:
            self.pos = nx.spring_layout(self.G, seed=42)
            
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in self.G.edges(data=True):
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
            
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in self.G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text with user details
            user_data = self.G.nodes[node]
            hover_text = f"Name: {user_data.get('name', 'N/A')}<br>"
            for category in categories:
                items = user_data.get(category, [])
                if items:
                    hover_text += f"{category}: {', '.join(items)}<br>"
            node_text.append(hover_text)
            
            # Node size based on number of connections
            node_size.append(10 + len(list(self.G.neighbors(node))) * 5)
            
        # Create the figure
        self.fig = go.Figure()
        
        # Add edges
        self.fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))
        
        # Add nodes
        self.fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color='lightblue',
                line_width=2,
                line_color='white'
            ),
            text=[self.G.nodes[node].get('name', '') for node in self.G.nodes()],
            textposition="top center",
            name='Users'
        ))
        
        # Update layout
        self.fig.update_layout(
            title=f"Network Analysis by {', '.join(categories)}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play Animation",
                            method="animate",
                            args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 300}}]
                        )
                    ]
                )
            ]
        )
        
        return self.fig
    
    def save_html(self, filename: str = "network_visualization.html"):
        """Save the visualization as an interactive HTML file"""
        if self.fig:
            self.fig.write_html(filename)
            
def main():
    # Create visualizer
    visualizer = DynamicNetworkVisualizer()
    
    # Add sample users (replace with real data from your database)
    sample_users = [
        {
            "id": "user1",
            "name": "Dan",
            "skills": ["Python", "JavaScript", "React"],
            "interests": ["AI", "Web Development"],
            "professionalGoals": ["Full Stack Developer"],
            "values": ["Innovation", "Learning"]
        },
        # Add more users from your database
    ]
    
    for user in sample_users:
        user_id = user.pop("id")
        visualizer.add_user(user_id, user)
    
    # Create visualizations for different category combinations
    category_combinations = [
        ["skills"],
        ["interests"],
        ["professionalGoals"],
        ["values"],
        ["skills", "interests"],
        ["skills", "interests", "professionalGoals", "values"]
    ]
    
    for categories in category_combinations:
        visualizer.update_edges(categories)
        fig = visualizer.create_interactive_visualization(categories)
        visualizer.save_html(f"network_by_{'_'.join(categories)}.html")

if __name__ == "__main__":
    main() 