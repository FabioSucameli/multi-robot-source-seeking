
# Robot Module - Mobile robots with concentration sensors
# Implements the robot team with position tracking and concentration measurement.

import numpy as np
from typing import List, Optional
from concentration_field import ConcentrationField

# Robot class representing a mobile robot with a concentration sensor
class Robot:
    
    # Initialize a robot
    # robot_id: Unique identifier for the robot
    # initial_position: Starting position [x, y]
    # is_leader: Whether this robot is the formation leader
    def __init__(self, robot_id: int, initial_position: np.ndarray, is_leader: bool = False):
        
        self.id = robot_id
        self.position = np.array(initial_position, dtype=float)
        self.is_leader = is_leader
        self.last_measurement = 0.0
        self.velocity = np.zeros(2)
        
        # History for visualization
        self.position_history = [self.position.copy()]
    
    # Measure concentration at current position
    def measure_concentration(self, field: ConcentrationField) -> float:
        
        self.last_measurement = field.get_concentration(self.position)
        return self.last_measurement
    
    # Update robot position
    def update_position(self, new_position: np.ndarray):
        
        self.velocity = new_position - self.position
        self.position = np.array(new_position, dtype=float)
        self.position_history.append(self.position.copy())
    
    def get_position(self) -> np.ndarray:
        # Return current position
        return self.position.copy()
    
    def get_measurement(self) -> float:
        # Return last concentration measurement
        return self.last_measurement

# RobotTeam class to manage multiple robots in formation
# Structure: One leader at center, others in polygon around
class RobotTeam:
    # Initialize the robot team
    # num_robots: Total number of robots (including leader)
    # initial_center: Initial position of the leader
    # formation_radius: Distance from leader to outer robots
    def __init__(self, num_robots: int, initial_center: np.ndarray, 
                 formation_radius: float = 5.0):
        self.num_robots = num_robots
        self.formation_radius = formation_radius
        self.robots: List[Robot] = []
        
        # Create leader robot at center
        leader = Robot(0, initial_center, is_leader=True)
        self.robots.append(leader)
        
        # Create outer robots in a regular polygon
        num_outer = num_robots - 1
        for i in range(num_outer):
            angle = 2 * np.pi * i / num_outer
            offset = formation_radius * np.array([np.cos(angle), np.sin(angle)])
            position = initial_center + offset
            robot = Robot(i + 1, position, is_leader=False)
            self.robots.append(robot)
        
        # Build adjacency graph
        self._build_adjacency_graph()
    
    # Build the formation graph
    def _build_adjacency_graph(self):
        
        n = self.num_robots
        self.adjacency = np.zeros((n, n), dtype=bool)
        
        # Leader (index 0) connected to all outer robots
        for i in range(1, n):
            self.adjacency[0, i] = True
            self.adjacency[i, 0] = True
        
        # Outer robots connected to adjacent neighbors
        num_outer = n - 1
        for i in range(1, n):
            # Previous neighbor (circular)
            prev = ((i - 2) % num_outer) + 1
            # Next neighbor (circular)
            next_idx = (i % num_outer) + 1
            
            self.adjacency[i, prev] = True
            self.adjacency[i, next_idx] = True
    
    # Get neighbors of a robot
    def get_neighbors(self, robot_id: int) -> List[int]:
        return [j for j in range(self.num_robots) if self.adjacency[robot_id, j]]
    
    # Have all robots measure concentration
    def collect_measurements(self, field: ConcentrationField) -> np.ndarray:
        measurements = np.array([r.measure_concentration(field) for r in self.robots])
        return measurements
    
    # Get positions of all robots
    def get_all_positions(self) -> np.ndarray:
        return np.array([r.get_position() for r in self.robots])
    
    def get_leader(self) -> Robot:
        # Return the leader robot
        return self.robots[0]
    
    def get_formation_center(self) -> np.ndarray:
        # Return the centroid of the formation
        positions = self.get_all_positions()
        return np.mean(positions, axis=0)
    
    # Get desired distances between connected robots
    def get_desired_distances(self) -> dict:
        distances = {}
        n = self.num_robots
        
        # Distance from leader to outer robots
        for i in range(1, n):
            distances[(0, i)] = self.formation_radius
            distances[(i, 0)] = self.formation_radius
        
        # Distance between adjacent outer robots
        num_outer = n - 1
        if num_outer > 1:
            # Side length of regular polygon
            side_length = 2 * self.formation_radius * np.sin(np.pi / num_outer)
            for i in range(1, n):
                prev = ((i - 2) % num_outer) + 1
                next_idx = (i % num_outer) + 1
                distances[(i, prev)] = side_length
                distances[(i, next_idx)] = side_length
        
        return distances
