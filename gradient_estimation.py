# Gradient Estimation Module

import numpy as np
from typing import Tuple

# Gradient estimator using least squares fitting.
class GradientEstimator:
    # Initialize the estimator
    def __init__(self, regularization: float = 1e-6):
        
        self.regularization = regularization
        self.last_estimate = np.zeros(2)
    
    # Estimate gradient from positions and measurements
    # The first position (index 0) is assumed to be the leader (p0).
    def estimate_gradient(self, positions: np.ndarray, 
                         measurements: np.ndarray) -> np.ndarray:
        n = len(positions)
        
        if n < 3:
            # Need at least 3 points (1 leader + 2 followers) for 2D gradient
            return self.last_estimate
        
        # Leader position and measurement
        p0 = positions[0]
        c0 = measurements[0]
        
        # Build matrix M ∈ R^{N×2} where N = n-1 (number of followers)
        # M[i] = (p_{i+1} - p0)^⊤
        M = positions[1:] - p0  # Shape: (N, 2)
        
        # Build vector v ∈ R^N
        # v[i] = c(p_{i+1}) - c(p0)
        v = measurements[1:] - c0  # Shape: (N,)
        
        # Solve least squares: min ||M * g - v||^2
        # Solution: ĝ = (M^⊤ M)^{-1} M^⊤ v
        MTM = M.T @ M  # Shape: (2, 2)
        MTM += self.regularization * np.eye(2)  # Regularization for numerical stability
        MTv = M.T @ v  # Shape: (2,)
        
        try:
            gradient = np.linalg.solve(MTM, MTv)
        except np.linalg.LinAlgError:
            # If solve fails, return last estimate
            gradient = self.last_estimate
        
        self.last_estimate = gradient
        return gradient
    
    # Estimate normalized gradient direction.
    def estimate_gradient_normalized(self, positions: np.ndarray,
                                     measurements: np.ndarray) -> np.ndarray:
        
        gradient = self.estimate_gradient(positions, measurements)
        norm = np.linalg.norm(gradient)
        
        if norm > 1e-8:
            return gradient / norm
        return np.zeros(2)

# Weighted gradient estimator using concentration-based weights.
class WeightedGradientEstimator(GradientEstimator):
    
    def __init__(self, regularization: float = 1e-6,
                 weight_exponent: float = 0.5):
        # Initialize the weighted gradient estimator.
        # regularization: Small value for numerical stability
        # weight_exponent: Exponent for concentration-based weighting

        super().__init__(regularization)
        self.weight_exponent = weight_exponent
    
    # Estimate gradient with concentration-weighted least squares.
    def estimate_gradient(self, positions: np.ndarray,
                         measurements: np.ndarray) -> np.ndarray:
        n = len(positions)
        
        if n < 3:
            return self.last_estimate
        
        # Leader position and measurement
        p0 = positions[0]
        c0 = measurements[0]
        
        # Build matrix M ∈ R^{N×2} (follower positions relative to leader)
        M = positions[1:] - p0  # Shape: (N, 2)
        
        # Build vector v ∈ R^N (concentration differences)
        v = measurements[1:] - c0  # Shape: (N,)
        
        # Compute weights based on follower concentrations
        follower_conc = measurements[1:]
        weights = np.power(np.maximum(follower_conc, 1e-6), self.weight_exponent)
        weights = weights / np.sum(weights)  # Normalize
        
        # Build diagonal weight matrix W
        W = np.diag(weights)
        
        # Weighted least squares: ĝ = (M^⊤ W M)^{-1} M^⊤ W v
        MTWM = M.T @ W @ M  # Shape: (2, 2)
        MTWM += self.regularization * np.eye(2)  # Regularization
        MTWv = M.T @ W @ v  # Shape: (2,)
        
        try:
            gradient = np.linalg.solve(MTWM, MTWv)
        except np.linalg.LinAlgError:
            gradient = self.last_estimate
        
        self.last_estimate = gradient
        return gradient
