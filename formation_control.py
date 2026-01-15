# Formation Control Module - Artificial Potential Fields (APF)
# Implements formation control using artificial potential fields to maintain a symmetric geometric formation around the leader robot.


import numpy as np
from typing import List, Tuple

# FormationController class to manage formation potential and gradients
class FormationController:
    
    def __init__(self, num_robots: int, formation_radius: float, 
                 adjacency: np.ndarray):
        # Initialize the formation controller.
        self.num_robots = num_robots
        self.formation_radius = formation_radius
        self.adjacency = adjacency
        self.desired_distances = self._compute_desired_distances()
    
    # Compute desired distances between connected robots
    def _compute_desired_distances(self) -> np.ndarray:
        n = self.num_robots
        distances = np.zeros((n, n))
        
        # Distance from leader (0) to outer robots
        for i in range(1, n):
            if self.adjacency[0, i]:
                distances[0, i] = self.formation_radius
                distances[i, 0] = self.formation_radius
        
        # Distance between adjacent outer robots (regular polygon)
        num_outer = n - 1
        if num_outer > 1:
            side_length = 2 * self.formation_radius * np.sin(np.pi / num_outer)
            for i in range(1, n):
                for j in range(1, n):
                    if i != j and self.adjacency[i, j]:
                        distances[i, j] = side_length
        
        return distances
    
    def compute_formation_potential(self, positions: np.ndarray) -> float:
        # Compute the total formation potential energy.
        
        n = self.num_robots
        potential = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.adjacency[i, j]:
                    d_actual = np.linalg.norm(positions[i] - positions[j])
                    d_desired = self.desired_distances[i, j]
                    potential += 0.5 * (d_actual - d_desired) ** 2
        
        return potential
    
    # Compute the gradient of formation potential for a specific robot.
    # The gradient points in the direction of increasing potential,
    # so robots should move in the negative gradient direction.
    def compute_formation_gradient(self, positions: np.ndarray, 
                                   robot_id: int) -> np.ndarray:
        
        gradient = np.zeros(2)
        
        for j in range(self.num_robots):
            if self.adjacency[robot_id, j]:
                diff = positions[robot_id] - positions[j]
                d_actual = np.linalg.norm(diff)
                
                if d_actual > 1e-6:  # Avoid division by zero
                    d_desired = self.desired_distances[robot_id, j]
                    # Gradient of 0.5 * (d - d_desired)^2
                    gradient += (d_actual - d_desired) * (diff / d_actual)
        
        return gradient
    
    # Compute formation gradients for all robots.
    def compute_all_formation_gradients(self, positions: np.ndarray) -> np.ndarray:
       
        gradients = np.zeros((self.num_robots, 2))
        for i in range(self.num_robots):
            gradients[i] = self.compute_formation_gradient(positions, i)
        return gradients
    
    # Compute the RMS formation error.
    def get_formation_error(self, positions: np.ndarray) -> float:
        errors = []
        n = self.num_robots
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.adjacency[i, j]:
                    d_actual = np.linalg.norm(positions[i] - positions[j])
                    d_desired = self.desired_distances[i, j]
                    errors.append((d_actual - d_desired) ** 2)
        
        if len(errors) == 0:
            return 0.0
        
        return np.sqrt(np.mean(errors))
