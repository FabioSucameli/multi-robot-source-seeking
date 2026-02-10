
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
    # measurement_noise_std: Standard deviation of Gaussian measurement noise
    def __init__(self, robot_id: int, initial_position: np.ndarray, is_leader: bool = False,
                 measurement_noise_std: float = 0.0):
        
        self.id = robot_id
        self.position = np.array(initial_position, dtype=float)
        self.is_leader = is_leader
        self.last_measurement = 0.0
        self.measurement_noise_std = measurement_noise_std
        
        # History for visualization
        self.position_history = [self.position.copy()]
    
    # Measure concentration at current position (with optional Gaussian noise)
    def measure_concentration(self, field: ConcentrationField) -> float:
        
        true_concentration = field.get_concentration(self.position)
        # Add Gaussian measurement noise
        if self.measurement_noise_std > 0:
            noise = np.random.normal(0, self.measurement_noise_std)
            self.last_measurement = max(0.0, true_concentration + noise)  # Concentration >= 0
        else:
            self.last_measurement = true_concentration
        return self.last_measurement
    
    # Update robot position
    def update_position(self, new_position: np.ndarray):
        
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
    # formation_noise_std: Standard deviation of initial position perturbation
    # measurement_noise_std: Standard deviation of Gaussian measurement noise
    def __init__(self, num_robots: int, initial_center: np.ndarray, 
                 formation_radius: float = 5.0,
                 formation_noise_std: float = 0.0,
                 measurement_noise_std: float = 0.0):
        self.num_robots = num_robots
        self.formation_radius = formation_radius
        self.formation_noise_std = formation_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.robots: List[Robot] = []
        
        # Create leader robot at center (with optional position noise)
        leader_position = initial_center.copy()
        if formation_noise_std > 0:
            leader_position += np.random.normal(0, formation_noise_std, 2)
        leader = Robot(0, leader_position, is_leader=True, 
                      measurement_noise_std=measurement_noise_std)
        self.robots.append(leader)
        
        # Create outer robots in a regular polygon (with optional position noise)
        num_outer = num_robots - 1
        for i in range(num_outer):
            angle = 2 * np.pi * i / num_outer
            offset = formation_radius * np.array([np.cos(angle), np.sin(angle)])
            position = initial_center + offset
            # Add formation noise (initial disturbance)
            if formation_noise_std > 0:
                position += np.random.normal(0, formation_noise_std, 2)
            robot = Robot(i + 1, position, is_leader=False,
                         measurement_noise_std=measurement_noise_std)
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
    
