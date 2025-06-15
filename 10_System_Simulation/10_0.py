import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
# ODE hello world
# --- Parameters ---
g = 9.81
L = 1.0
m = 1.0  # Mass (affects lambda magnitude, not direction visualization)


# --- ODE for standard pendulum angle ---
def pendulum_ode(t, state):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]


# --- Time span and initial conditions ---
t_span = (0, 10)
dt = 0.05
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
theta0 = np.pi / 2  # Initial angle (horizontal)
omega0 = 0.0  # Initial angular velocity
initial_state = [theta0, omega0]

# --- Solve ODE ---
sol = solve_ivp(pendulum_ode, t_span, initial_state, t_eval=t_eval, dense_output=True)
theta = sol.y[0]
omega = sol.y[1]

# --- Convert to Cartesian coordinates ---
x = L * np.sin(theta)
y = -L * np.cos(theta)  # Negative because y=0 is pivot, positive y is up

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L * 1.5, L * 1.5)
ax.set_ylim(-L * 1.5, L * 1.5)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Cartesian Pendulum: Constraint Force and Gradient Direction')
ax.grid(True)

# Draw the circular path
path = plt.Circle((0, 0), L, color='grey', fill=False, ls='--')
ax.add_patch(path)

# Initialize plot elements
pivot, = ax.plot([0], [0], 'ko', markersize=5)
bob, = ax.plot([], [], 'o', color='blue', markersize=10, label='Bob')
rod, = ax.plot([], [], '-', color='black', lw=1.5, label='Rod')
# Use quiver for vectors: quiver(x_origin, y_origin, x_component, y_component, ...)
gradient_vec = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='red', width=0.008,
                         label=r'$\nabla C = (2x, 2y)$ (Gradient)')
constraint_force_dir_vec = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='green', width=0.005,
                                     label='Constraint Force Dir.')


# --- Animation Update Function ---
def update(frame):
    current_x = x[frame]
    current_y = y[frame]

    # Update bob position
    bob.set_data([current_x], [current_y])

    # Update rod position
    rod.set_data([0, current_x], [0, current_y])

    # Update Gradient Vector (origin at bob, points radially out)
    grad_x = 2 * current_x
    grad_y = 2 * current_y
    # Scale for visibility
    scale_factor_grad = 0.2
    gradient_vec.set_offsets([current_x, current_y])
    gradient_vec.set_UVC(grad_x * scale_factor_grad, grad_y * scale_factor_grad)

    # Update Constraint Force Direction Vector (origin at bob, points radially in/out)
    # Let's show tension (pointing inwards) as an example
    force_dir_x = -current_x
    force_dir_y = -current_y
    # Scale for visibility
    scale_factor_force = 0.3
    constraint_force_dir_vec.set_offsets([current_x, current_y])
    constraint_force_dir_vec.set_UVC(force_dir_x * scale_factor_force, force_dir_y * scale_factor_force)

    return bob, rod, gradient_vec, constraint_force_dir_vec


# --- Create and Show Animation ---
ax.legend(loc='upper right')
ani = animation.FuncAnimation(fig, update, frames=len(t_eval),
                              interval=int(dt * 1000), blit=True, repeat=False)

plt.show()

# To save the animation (optional, requires ffmpeg or similar):
# print("Saving animation...")
# ani.save('cartesian_pendulum_constraint.gif', writer='pillow', fps=1/dt)
# print("Done.")

a = 1
