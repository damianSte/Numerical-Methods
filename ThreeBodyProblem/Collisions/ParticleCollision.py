import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('TkAgg')

#Define masses of particles
m1 = m2 = 1.0

#Initial conditions
y0 = [1.0, 0.0, -1.0, 0.0,
      -1.0, 0.0, 1.0, 0.0]

stop = 20
t_eval = np.linspace(0, stop, 2000)

def lenard_jones_potential(x):
     return 4 * (1 / x**12 - 1 / x**6)

def U(x):
    return (48 / x ** 13) - (24 / x ** 7)

def particle_system(t,y):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    force = U(r)
    ax1 = force * (x2 - x1) / r
    ay1 = force * (y2 - y1) / r
    # ax2 = -ax1
    # ay2 = -ay1
    ax2 = force * (x1 - x2)
    ay2 = force * (y1 - y2)

    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]


solution = solve_ivp(particle_system, [0, stop], y0,
                     t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)


def animate_particles():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)

    colors = ['red', 'blue']
    particles = [ax.plot([], [], 'o', color=c, markersize=10, label=f'Particle {i+1}')[0]
                 for i, c in enumerate(colors)]
    trails = [ax.plot([], [], '-', color=c, alpha=0.3, lw=1.5)[0] for c in colors]

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
        """Update positions and trails for each frame."""
        for i, (particle, trail) in enumerate(zip(particles, trails)):
            particle.set_data([solution.y[2 * i, frame]], [solution.y[2 * i + 1, frame]])

            start = max(0, frame - trail_length)
            trail.set_data(solution.y[2 * i, start:frame + 1], solution.y[2 * i + 1, start:frame + 1])
        return particles + trails

    anim = FuncAnimation(fig, update, frames=len(solution.t), init_func=init, blit=True,
                         interval=20, repeat=True)

    print("solution.t.shape:", solution.t.shape)

    plt.show()
    return anim


if __name__ == "__main__":
    animate_particles()






