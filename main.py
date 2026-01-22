# Main entry point for running the source seeking simulation with
# formation control and distributed gradient estimation.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

from concentration_field import ConcentrationField, create_elliptical_field
from robot import RobotTeam
from formation_control import FormationController
from gradient_estimation import GradientEstimator, WeightedGradientEstimator
from control_laws import SourceSeekingController, AdaptiveController


def run_simulation(
    num_robots: int = 7,
    num_steps: int = 300,
    source_position: tuple = (50.0, 50.0),
    initial_position: tuple = (10.0, 10.0),
    formation_radius: float = 5.0,
    sampling_time: float = 0.1,
    formation_gain: float = 2.0,
    gradient_gain: float = 1.0,
    max_velocity: float = 2.0,
    use_adaptive: bool = False,
    visualize: bool = True,
    animate: bool = False,
    save_gif: bool = False,
    gif_filename: str = 'simulation.gif',
    animation_speed: int = 1,
    early_stopping: bool = True,
    concentration_threshold: float = 90.0,
    gradient_norm_threshold: float = 0.5,
    stability_window: int = 30
):
    # Run the simulation.
    print("=" * 60)
    print("Multi-Robot Source Seeking Simulation")
    print("=" * 60)
    
    # Create concentration field
    field = create_elliptical_field(
        source_position=np.array(source_position),
        sigma_x=15.0,
        sigma_y=10.0,
        rotation=np.pi / 6,
        amplitude=100.0
    )
    
    print(f"\nSource position: {source_position} ")
    print(f"Initial position: {initial_position}")
    print(f"Number of robots: {num_robots}")
    print(f"Formation radius: {formation_radius}")
    if early_stopping:
        print(f"Early stopping: enabled (conc > {concentration_threshold}, |grad| < {gradient_norm_threshold}, window = {stability_window})")
    
    # Create robot team
    team = RobotTeam(
        num_robots=num_robots,
        initial_center=np.array(initial_position),
        formation_radius=formation_radius
    )
    
    # Create formation controller
    formation_ctrl = FormationController(
        num_robots=num_robots,
        formation_radius=formation_radius,
        adjacency=team.adjacency
    )
    
    # Create gradient estimator
    gradient_est = GradientEstimator()
    
    # Create main controller
    if use_adaptive:
        controller = AdaptiveController(
            robot_team=team,
            formation_controller=formation_ctrl,
            gradient_estimator=gradient_est,
            sampling_time=sampling_time,
            formation_gain=formation_gain,
            gradient_gain=gradient_gain,
            max_velocity=max_velocity
        )
        print("Using: Adaptive Controller")
    else:
        controller = SourceSeekingController(
            robot_team=team,
            formation_controller=formation_ctrl,
            gradient_estimator=gradient_est,
            sampling_time=sampling_time,
            formation_gain=formation_gain,
            gradient_gain=gradient_gain,
            max_velocity=max_velocity
        )
        print("Using: Standard Controller")
    
    # Run simulation
    print("\nRunning simulation...")
    
    
    distances = []
    concentrations = []
    gradient_norms = []
    formation_errors = []
    converged = False
    final_step = num_steps
    
    for step in range(num_steps):
        gradient, distance = controller.step(field)
        distances.append(distance)
        
        leader_conc = team.get_leader().get_measurement()
        concentrations.append(leader_conc)
        
        gradient_norm = np.linalg.norm(gradient)
        gradient_norms.append(gradient_norm)
        
        # Get current formation error
        current_formation_error = controller.formation_error_history[-1]
        formation_errors.append(current_formation_error)
        
        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Step: {step:5d},  Distance to source: {distance:6.2f},  Formation Error: {current_formation_error:7.4f},  Concentration: {leader_conc:5.2f}")
        
        # Early stopping check based on observable quantities
        # High concentration (close to maximum)
        # Small gradient norm (at/near peak)
        # Stability over recent window
        if early_stopping and step >= stability_window:
            recent_concentrations = concentrations[-stability_window:]
            recent_gradient_norms = gradient_norms[-stability_window:]
            
            avg_concentration = np.mean(recent_concentrations)
            avg_gradient_norm = np.mean(recent_gradient_norms)
            concentration_stable = np.std(recent_concentrations) < 1.0  # Low variance
            
            if (avg_concentration > concentration_threshold and 
                avg_gradient_norm < gradient_norm_threshold and
                concentration_stable):
                converged = True
                final_step = step + 1
                print(f"\n" + "*" * 60)
                print(f"  SUCCESS! Source reached at step {step}  ")
                print(f"  Avg Concentration: {avg_concentration:.3f} (threshold: {concentration_threshold})")
                print(f"  Avg |Gradient|: {avg_gradient_norm:.6f} (threshold: {gradient_norm_threshold})")
                print("*" * 60)
                break
    
    # Final results
    final_distance = distances[-1]
    final_formation_error = formation_errors[-1]
    avg_formation_error = np.mean(formation_errors)
    max_formation_error = np.max(formation_errors)
    initial_distance = np.linalg.norm(
        np.array(initial_position) - np.array(source_position)
    )
    improvement = (1 - final_distance / initial_distance) * 100
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Converged:                  {'YES' if converged else 'NO (max steps reached)'}")
    print(f"  Total steps:                {final_step} / {num_steps}")
    print(f"  Initial distance to source: {initial_distance:.2f}")
    print(f"  Final distance to source:   {final_distance:.2f}")
    print(f"  Improvement:                {improvement:.1f}%")
    print(f"  Final concentration:        {concentrations[-1]:.2f}")
    print(f"  Final formation error:      {final_formation_error:.4f}")
    print(f"  Average formation error:    {avg_formation_error:.4f}")
    print(f"  Max formation error:        {max_formation_error:.4f}")
    print("=" * 60)
    
    results = {
        'distances': np.array(distances),
        'concentrations': np.array(concentrations),
        'team': team,
        'field': field,
        'controller': controller,
        'final_distance': final_distance,
        'improvement': improvement,
        'converged': converged,
        'final_step': final_step
    }
    
    # Visualization
    if visualize:
        visualize_results(results, source_position, initial_position)
    
    if animate or save_gif:
        create_animation(results, source_position, save_gif=save_gif, gif_filename=gif_filename, animation_speed=animation_speed)
    
    return results


def visualize_results(results: dict, source_pos: tuple, initial_pos: tuple):
    # Create visualization of simulation results.
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    team = results['team']
    field = results['field']
    distances = results['distances']
    concentrations = results['concentrations']
    
    # Plot 1: Robot trajectories with concentration field
    ax1 = axes[0, 0]
    
    # Create concentration field heatmap
    x_range = np.linspace(-10, 70, 100)
    y_range = np.linspace(-10, 70, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = field.get_concentration(np.array([X[i, j], Y[i, j]]))
    
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='hot', alpha=0.7)
    plt.colorbar(contour, ax=ax1, label='Concentration')
    
    # Plot robot trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(team.robots)))
    for i, robot in enumerate(team.robots):
        history = np.array(robot.position_history)
        if robot.is_leader:
            ax1.plot(history[:, 0], history[:, 1], 'b-', linewidth=2, 
                    label='Leader trajectory', zorder=5)
            ax1.scatter(history[-1, 0], history[-1, 1], c='blue', s=150, 
                       marker='*', edgecolors='white', linewidth=2, zorder=6)
        else:
            ax1.plot(history[:, 0], history[:, 1], '-', color=colors[i], 
                    alpha=0.5, linewidth=1)
            ax1.scatter(history[-1, 0], history[-1, 1], c=[colors[i]], s=60, 
                       marker='o', edgecolors='white', zorder=5)
    
    # Mark source and start
    ax1.scatter(*source_pos, c='red', s=300, marker='X', 
               edgecolors='white', linewidth=2, label='Source', zorder=7)
    ax1.scatter(*initial_pos, c='green', s=150, marker='s', 
               edgecolors='white', linewidth=2, label='Start', zorder=7)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Robot Trajectories and Concentration Field')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance to source over time 
    ax2 = axes[0, 1]
    steps = np.arange(len(distances))
    ax2.plot(steps, distances, 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Source')
    ax2.fill_between(steps, distances, alpha=0.3)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Distance to Source')
    ax2.set_title('Convergence to Source')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Concentration measured by leader
    ax3 = axes[1, 0]
    ax3.plot(steps, concentrations, 'r-', linewidth=2)
    ax3.fill_between(steps, concentrations, alpha=0.3, color='red')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Concentration at Leader Position')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Formation error over time 
    ax4 = axes[1, 1]
    metrics = results['controller'].get_convergence_metrics()
    formation_errors = metrics['formation_errors']
    ax4.plot(steps, formation_errors, 'g-', linewidth=2)
    ax4.fill_between(steps, formation_errors, alpha=0.3, color='green')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Formation Error (RMS)')
    ax4.set_title('Formation Maintenance Error')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'simulation_results.png'")
    plt.show()

# Create an animation of the simulation.
def create_animation(results: dict, source_pos: tuple, save_gif: bool = False, gif_filename: str = 'simulation.gif', animation_speed: int = 1):
    
    # animation_speed: Multiplier for animation speed (1 = normal, 2 = 2x faster, etc.)
    # Higher values skip more frames, making the animation faster visually.
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    team = results['team']
    field = results['field']
    final_step = results.get('final_step', len(results['distances']))
    converged = results.get('converged', False)
    
    # Create concentration field heatmap
    x_range = np.linspace(-10, 70, 80)
    y_range = np.linspace(-10, 70, 80)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = field.get_concentration(np.array([X[i, j], Y[i, j]]))
    
    ax.contourf(X, Y, Z, levels=30, cmap='hot', alpha=0.6)
    ax.scatter(*source_pos, c='red', s=300, marker='X', 
              edgecolors='white', linewidth=2, zorder=10)
    
    # Get all trajectory data
    all_histories = [np.array(robot.position_history) for robot in team.robots]
    
    # Use final_step as the number of frames (stops at convergence)
    total_steps = min(final_step, len(all_histories[0]))
    
    # Apply animation_speed: select every Nth frame
    frame_indices = list(range(0, total_steps, animation_speed))
    # Always include the last frame
    if frame_indices[-1] != total_steps - 1:
        frame_indices.append(total_steps - 1)
    num_frames = len(frame_indices)
    
    # Initialize plot elements
    scatter_leader = ax.scatter([], [], c='blue', s=200, marker='*', 
                               edgecolors='white', linewidth=2, zorder=8)
    scatter_followers = ax.scatter([], [], c='cyan', s=100, marker='o',
                                  edgecolors='white', linewidth=1, zorder=7)
    trajectory_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=5)
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-10, 70)
    ax.set_ylim(-10, 70)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Multi-Robot Source Seeking')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    def init():
        scatter_leader.set_offsets(np.empty((0, 2)))
        scatter_followers.set_offsets(np.empty((0, 2)))
        trajectory_line.set_data([], [])
        status_text.set_text('')
        return scatter_leader, scatter_followers, trajectory_line, status_text
    
    def update(frame_idx):
        # Get actual step index from frame_indices
        actual_step = frame_indices[frame_idx]
        
        # Leader position
        leader_pos = all_histories[0][actual_step:actual_step+1]
        scatter_leader.set_offsets(leader_pos)
        
        # Follower positions
        follower_pos = np.array([h[actual_step] for h in all_histories[1:]])
        scatter_followers.set_offsets(follower_pos)
        
        # Leader trajectory
        traj = all_histories[0][:actual_step+1]
        trajectory_line.set_data(traj[:, 0], traj[:, 1])
        
        # Update title and status
        if frame_idx == num_frames - 1 and converged:
            ax.set_title(f'Multi-Robot Source Seeking (Step {actual_step}) - CONVERGED!')
            status_text.set_text(f'Source reached!\nDistance: {results["final_distance"]:.2f}')
            status_text.set_color('green')
        else:
            ax.set_title(f'Multi-Robot Source Seeking (Step {actual_step})')
            dist = results['distances'][actual_step] if actual_step < len(results['distances']) else results['final_distance']
            status_text.set_text(f'Distance: {dist:.2f}')
            status_text.set_color('black')
        
        return scatter_leader, scatter_followers, trajectory_line, status_text
    
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames,
                        interval=30, blit=True)  # Reduced interval for smoother playback
    
    if save_gif:
        print(f"\nSaving animation to '{gif_filename}'...")
        # Use pillow writer for GIF
        anim.save(gif_filename, writer='pillow', fps=20)
        print(f"Animation saved to '{gif_filename}' ({num_frames} frames)")
        if converged:
            print(f"GIF stops at convergence (step {final_step})")
    else:
        plt.show()
        print("\nAnimation displayed.")
    
    plt.close(fig)


def main():

    results = run_simulation(
        num_robots=7,          # 1 leader + 6 outer robots (hexagon)
        num_steps=15000,
        source_position=(50.0, 50.0),
        initial_position=(10.0, 10.0),
        formation_radius=6.0,  # Larger formation for better gradient estimation
        sampling_time=0.2,     # Time step
        formation_gain=1.5,    # Formation control gain
        gradient_gain=1.0,     # Gradient following gain
        max_velocity=0.5,     # Max velocity (higher limit)
        use_adaptive=True,
        visualize=True,
        animate=True,          # Set to True for animation
        save_gif=False,        # Set to True to save animation as GIF
        gif_filename='simulation.gif',  # Output filename for GIF
        animation_speed=50,     # Animation speed multiplier (1=normal, 3=3x faster)
        early_stopping=True,
        concentration_threshold=99.0,  # High concentration indicates near source
        gradient_norm_threshold=0.1,   # Small gradient indicates at peak
        stability_window=100            # Check stability over last N steps
    )
    
    return results


if __name__ == "__main__":
    main()
