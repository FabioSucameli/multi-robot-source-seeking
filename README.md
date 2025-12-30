# Multi-Robot Source Seeking with Formation Control

## Overview

This project simulates a **source seeking problem** where a team of mobile robots equipped with concentration sensors works cooperatively to locate an unknown source. The robots maintain a geometric formation and use distributed measurements to estimate the concentration gradient, progressively converging toward the source.

The implementation uses a simplified but modular architecture that can be extended to more complex models.

---

## Problem Description

### 1. Concentration Model (Gaussian Plume Model)

The concentration field is modeled using a **2D Gaussian Plume Model**:

- **Space**: 2D environment, $x \in \mathbb{R}^2$
- **Source**: Located at unknown position $x_0 \in \mathbb{R}^2$
- **Concentration function**: 
  $$c(x) = g(x; x_0, P)$$
  where:
  - $x_0$: source position
  - $P$: covariance matrix determining the Gaussian shape (elliptical distribution)

The concentration is maximum at the source and decreases with distance.

### 2. Robot Team and Sensors

The system consists of **N mobile robots** with the following characteristics:

- Each robot knows its own position
- Each robot has a concentration sensor measuring $c(x)$ at its location
- Robots do not know the source position $x_0$
- Robots can communicate to share concentration measurements

### 3. Formation Control (Artificial Potential Fields)

To estimate the concentration gradient direction, robots maintain a **fixed symmetric geometric formation** using **Artificial Potential Fields (APF)**:

**Formation Structure**:
- One **central robot (leader)**
- $N-1$ robots arranged on a **regular polygon** around the leader (e.g., hexagon)
- **Communication graph**:
  - Leader connected to all outer robots
  - Each outer robot connected to adjacent neighbors

Each robot maintains:
- Fixed distance from the leader
- Correct distance from adjacent robots

Formation control minimizes a formation potential $J_{\text{for}}(p)$ depending on all robot positions.

### 4. Gradient Estimation

At each time step:
1. All robots measure local concentration
2. Measurements are shared across the team
3. A **gradient estimate** $\hat{g}_t$ is computed from distributed measurements

This estimate indicates the direction of maximum concentration increase.

The gradient is estimated by locally approximating the concentration field and solving a least squares problem based on concentration measurements collected at different spatial locations. This approach provides a robust gradient estimate and benefits from well-distributed robot configurations around the central agent.

### 5. Control Laws

**Non-leader robots**:
$$p_i(t+1) = p_i(t) - T_s K_{\text{for}} \frac{\partial J_{\text{for}}}{\partial p_i}$$

where:
- $T_s$: sampling time
- $K_{\text{for}}$: formation control gain
- Anti-gradient term maintains formation

**Leader robot**:
$$p_1(t+1) = p_1(t) - T_s K_{\text{for}} \frac{\partial J_{\text{for}}}{\partial p_1} + T_s K_{\text{grad}} \hat{g}_t$$

The additional term:
- Drives the leader toward maximum concentration increase
- Guides the formation toward the source

### 6. System Behavior

1. Formation is maintained through APF
2. Distributed measurements enable gradient estimation
3. Leader follows the estimated gradient
4. Other robots follow the leader while maintaining formation
5. The entire group progressively converges to the source


