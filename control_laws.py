
# Control Laws Module
# Implements the control laws for robot motion:
# - Formation control for non-leader robots
# - Formation control + gradient following for leader robot


import numpy as np
from typing import Tuple

from robot import RobotTeam
from formation_control import FormationController
from gradient_estimation import GradientEstimator

# Main controller for multi-robot source seeking.
class SourceSeekingController:
    
    def __init__(self, 
                 robot_team: RobotTeam,
                 formation_controller: FormationController,
                 gradient_estimator: GradientEstimator,
                 sampling_time: float = 0.1,
                 formation_gain: float = 1.0,
                 gradient_gain: float = 0.5,
                 max_velocity: float = 2.0):
        
        self.robot_team = robot_team
        self.formation_controller = formation_controller
        self.gradient_estimator = gradient_estimator
        
        self.Ts = sampling_time
        self.K_for = formation_gain
        self.K_grad = gradient_gain
        self.max_velocity = max_velocity
        
        # Logging
        self.gradient_history = []
        self.formation_error_history = []
    
    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        # Clip velocity to maximum magnitude.
        norm = np.linalg.norm(velocity)
        if norm > self.max_velocity:
            return velocity * (self.max_velocity / norm)
        return velocity
    
    # Compute control inputs for all robots.
    def compute_control(self, positions: np.ndarray,
                       measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        n = len(positions)
        new_positions = np.zeros_like(positions)
        
        # Compute formation gradients for all robots
        formation_gradients = self.formation_controller.compute_all_formation_gradients(positions)
        
        # Estimate concentration gradient
        conc_gradient = self.gradient_estimator.estimate_gradient(positions, measurements)
        self.gradient_history.append(conc_gradient.copy())
        
        # Normalize gradient direction for consistent motion toward source
        grad_norm = np.linalg.norm(conc_gradient)
        if grad_norm > 1e-8:
            conc_gradient_normalized = conc_gradient / grad_norm
        else:
            conc_gradient_normalized = np.zeros(2)
        
        # Log formation error
        formation_error = self.formation_controller.get_formation_error(positions)
        self.formation_error_history.append(formation_error)
        
        # Update each robot
        for i in range(n):
            robot = self.robot_team.robots[i]
            
            # Formation control term (move against gradient of formation potential)
            formation_term = -self.K_for * formation_gradients[i]
            
            if robot.is_leader:
                # Leader also follows concentration gradient (using normalized direction)
                gradient_term = self.K_grad * conc_gradient_normalized
                velocity = formation_term + gradient_term
            else:
                # Non-leader: only formation control
                velocity = formation_term
            
            # Clip velocity to prevent instability
            velocity = self._clip_velocity(velocity)
            
            # Compute new position
            new_positions[i] = positions[i] + self.Ts * velocity
        
        return new_positions, conc_gradient
    
    # Perform one control step.
    def step(self, field) -> Tuple[np.ndarray, float]:
        
        # Get current positions
        positions = self.robot_team.get_all_positions()
        
        # Collect measurements
        measurements = self.robot_team.collect_measurements(field)
        
        # Compute control
        new_positions, gradient = self.compute_control(positions, measurements)
        
        # Update robot positions
        for i, robot in enumerate(self.robot_team.robots):
            robot.update_position(new_positions[i])
        
        # Compute distance to source
        leader_pos = self.robot_team.get_leader().get_position()
        source_pos = field.get_source_position()
        distance = np.linalg.norm(leader_pos - source_pos)
        
        return gradient, distance
    
    def get_convergence_metrics(self) -> dict:
        # Get metrics about the control performance.
        
        return {
            'formation_errors': np.array(self.formation_error_history),
            'gradient_norms': np.array([np.linalg.norm(g) for g in self.gradient_history])
        }

# Adaptive controller that adjusts gains based on situation.
# - Increases gradient gain when close to source
# - Decreases formation gain when gradient is strong
class AdaptiveController(SourceSeekingController):
    
    def __init__(self, *args, 
                 min_gradient_gain: float = 0.2,
                 max_gradient_gain: float = 2.0,
                 max_velocity: float = 2.0,
                 **kwargs):
        super().__init__(*args, max_velocity=max_velocity, **kwargs)
        self.min_gradient_gain = min_gradient_gain
        self.max_gradient_gain = max_gradient_gain
        self.base_gradient_gain = self.K_grad
    
    def compute_control(self, positions: np.ndarray,
                       measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Compute control with adaptive gains.
        # Adapt gradient gain based on concentration level
        max_conc = np.max(measurements)
        avg_conc = np.mean(measurements)
        
        # Higher concentration = higher gradient gain
        conc_factor = min(max_conc / (avg_conc + 1e-6), 3.0)
        self.K_grad = np.clip(
            self.base_gradient_gain * conc_factor,
            self.min_gradient_gain,
            self.max_gradient_gain
        )
        
        return super().compute_control(positions, measurements)
