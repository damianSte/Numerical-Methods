import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

def particle_acceleration(r):
    return (48 / r ** 13) - (24 / r ** 7)

def particle_system(t, y):
    number_of_particles = len(y) // 4
    dydt = np.zeros_like(y)

    #Boundries
    box_limit = 4

    for i in range(number_of_particles):
        # Positions and velocities
        x_i, y_i, vx_i, vy_i = y[4 * i: 4 * i + 4]
        ax_i, ay_i = 0, 0


        #Box limits
        if x_i <= -box_limit or x_i >= box_limit:
            vx_i = -vx_i
        if y_i <= -box_limit or y_i >= box_limit:
            vy_i = -vy_i

        # Acceleration/ Movement
        for j in range(number_of_particles):
            if i != j:
                x_j, y_j, vx_j, vy_j = y[4 * j: 4 * j + 4]
                r = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                force = particle_acceleration(r)
                ax_i += force * (x_i - x_j) / r
                ay_i += force * (y_i - y_j) / r

        dydt[4 * i: 4 * i + 4] = [vx_i, vy_i, ax_i, ay_i]

    return dydt

number_of_particles = 6
y0 = []


for _ in range(number_of_particles):
    x, y = np.random.uniform(-1, 1, 2)
    vx, vy = np.random.uniform(-1, 1, 2)
    y0.extend([x, y, vx, vy])

y0 = np.array(y0)

stop = 20
t_eval = np.linspace(0, stop, 2000)

solution = solve_ivp(particle_system, [0, stop], y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

# Animation function
def animate_particles():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    particles = [ax.plot([], [], 'o', color=c, markersize=10, label=f'Particle {i + 1}')[0]
                 for i, c in enumerate(colors[:number_of_particles])]
    trails = [ax.plot([], [], '-', color=c, alpha=0.3, lw=1.5)[0] for c in colors[:number_of_particles]]

    ax.legend()
    plt.title('Particle Collisions Simulation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    def init():
        for particle, trail in zip(particles, trails):
            particle.set_data([], [])
            trail.set_data([], [])
        return particles + trails

    def update(frame):
        trail_length = 150
        for i, (particle, trail) in enumerate(zip(particles, trails)):
            particle.set_data([solution.y[4 * i, frame]], [solution.y[4 * i + 1, frame]])

            start = max(0, frame - trail_length)
            trail.set_data(solution.y[4 * i, start:frame + 1], solution.y[4 * i + 1, start:frame + 1])
        return particles + trails

    anim = FuncAnimation(fig, update, frames=len(solution.t), init_func=init, blit=True,
                         interval=20, repeat=True)

    plt.show()
    return anim

# Run the animation
if __name__ == "__main__":
    animate_particles()
