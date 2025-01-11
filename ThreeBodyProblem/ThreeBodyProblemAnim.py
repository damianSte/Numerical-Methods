import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend

# Constants
G = 1  # Gravitational constant
m1 = m2 = m3 = 1  # Masses of the three bodies

# Define the three-body system of equations
def three_body_system(t, y):
    r1, r2, r3 = y[:2], y[2:4], y[4:6]  # Positions of three bodies
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]  # Velocities of three bodies

    def acceleration_between(ri, rj, mj):
        r = np.linalg.norm(rj - ri)  # Distance between particles
        return G * mj * (rj - ri) / r**3  # Gravitational acceleration

    # Accelerations
    a1 = acceleration_between(r1, r2, m2) + acceleration_between(r1, r3, m3)
    a2 = acceleration_between(r2, r1, m1) + acceleration_between(r2, r3, m3)
    a3 = acceleration_between(r3, r1, m1) + acceleration_between(r3, r2, m2)

    # Return derivatives: [dx/dt, dv/dt] for all three bodies
    return [*v1, *v2, *v3, *a1, *a2, *a3]

# Initial conditions: positions and velocities
y0 = [-0.97, 0.243, 0, 0, 0.97, -0.243,  # Positions: r1, r2, r3
      0.466, 0.432, -0.932, -0.864, 0.466, 0.432]  # Velocities: v1, v2, v3

# Time span for the simulation
t_end = 20
t_eval = np.linspace(0, t_end, 1000)

# Solve the system
solution = solve_ivp(three_body_system, [0, t_end], y0,
                     t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

# Extract positions of the three bodies
r1 = solution.y[:2]  # x, y for body 1
r2 = solution.y[2:4]  # x, y for body 2
r3 = solution.y[4:6]  # x, y for body 3

# Animation Function
def animate_three_body():
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 2)  # Adjust the limits as needed
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)

    # Initialize particles and trails
    colors = ['red', 'blue', 'green']
    particles = [ax.plot([], [], 'o', color=c, markersize=10, label=f'Body {i+1}')[0]
                 for i, c in enumerate(colors)]
    trails = [ax.plot([], [], '-', color=c, alpha=0.3, lw=1.5)[0] for c in colors]

    ax.legend()
    plt.title('Three-Body Problem')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    def init():
        """Initialize animation."""
        for particle, trail in zip(particles, trails):
            particle.set_data([], [])
            trail.set_data([], [])
        return particles + trails

    def update(frame):
        trail_length = 150
        """Update positions and trails for each frame."""
        for i, (particle, trail) in enumerate(zip(particles, trails)):
            # Update particle position
            particle.set_data([solution.y[2 * i, frame]], [solution.y[2 * i + 1, frame]])

            # Update trail
            start = max(0, frame - trail_length)
            trail.set_data(solution.y[2 * i, start:frame + 1], solution.y[2 * i + 1, start:frame + 1])
        return particles + trails

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True,
                         interval=20, repeat=True)

    plt.show()
    return anim  # Keep reference to avoid garbage collection

# Run the animation
if __name__ == "__main__":
    animate_three_body()
