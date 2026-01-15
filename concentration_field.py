# Concentration Field Module - 2D Gaussian Plume Model

# Implements the concentration field model where concentration is maximum
# at the source and decreases with distance following a Gaussian distribution.

import numpy as np

# 2D Gaussian Plume Model for concentration field.
class ConcentrationField:
    
    def __init__(self, source_position: np.ndarray, 
                 covariance: np.ndarray = None,
                 amplitude: float = 100.0):
        
        # Initialize the concentration field.
        self.x0 = np.array(source_position, dtype=float)
        self.P = covariance if covariance is not None else np.eye(2) * 50.0
        self.P_inv = np.linalg.inv(self.P)
        self.amplitude = amplitude
    
    # Get concentration at a given position.
    def get_concentration(self, position: np.ndarray) -> float:
        
        x = np.array(position, dtype=float)
        diff = x - self.x0
        exponent = -0.5 * diff @ self.P_inv @ diff
        return self.amplitude * np.exp(exponent)
    
    # Get the analytical gradient of concentration at a position.
    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        
        x = np.array(position, dtype=float)
        c = self.get_concentration(x)
        diff = x - self.x0
        return -c * self.P_inv @ diff
    
    def get_source_position(self) -> np.ndarray:
        # Return the source position (for visualization only).
        return self.x0.copy()

# Create a concentration field with elliptical shape.
def create_elliptical_field(source_position: np.ndarray,
                            sigma_x: float = 10.0,
                            sigma_y: float = 5.0,
                            rotation: float = 0.0,
                            amplitude: float = 100.0) -> ConcentrationField:
    # Rotation matrix
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    
    # Covariance matrix
    D = np.diag([sigma_x**2, sigma_y**2])
    P = R @ D @ R.T
    
    return ConcentrationField(source_position, P, amplitude)
