import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from scipy.stats import chi2

# Simulation parameters
np.random.seed(50)
num_steps = 100

# Landmark positions (cameras)
landmarks = np.array([[0, 0], [10, 0], [5, 10]])

# Sensor errors (Dr, Dtheta) for each camera
camera_errors = [
    (0.1, np.deg2rad(2)),  # Camera 1
    (0.2, np.deg2rad(3)),  # Camera 2
    (0.15, np.deg2rad(1.5))  # Camera 3
]

# Robot's sensor bearing error
robot_phi_error = np.deg2rad(2)

# Initialize ground truth and estimated positions
true_positions = []
true_orientations = []
est_positions = []
est_orientations = []
est_covariances = []  # Store covariance matrices

for step in range(num_steps):
    # Ground truth position and orientation
    if step == 0:
        x, y = np.random.uniform(2, 8), np.random.uniform(2, 8)
        theta = np.random.uniform(-np.pi, np.pi)
    else:
        x += np.random.uniform(-1, 1)
        y += np.random.uniform(-1, 1)
        theta += np.random.uniform(-np.pi/5, np.pi/5)

    # Store ground truth
    true_positions.append((x, y))
    true_orientations.append(theta)

    # --- Step 1: Camera measurements ---
    sensor_positions = []
    sensor_covariances = []
    for i, (lx, ly) in enumerate(landmarks):
        dx = x - lx
        dy = y - ly
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        # Add noise
        dr, dtheta = camera_errors[i]
        r_meas = r + np.random.normal(0, dr)
        theta_meas = angle + np.random.normal(0, dtheta)

        # Convert to global coordinates
        est_x = lx + r_meas * np.cos(theta_meas)
        est_y = ly + r_meas * np.sin(theta_meas)
        sensor_positions.append((est_x, est_y))

        # Covariance matrix via Jacobian
        J = np.array([
            [np.cos(theta_meas), -r_meas * np.sin(theta_meas)],
            [np.sin(theta_meas),  r_meas * np.cos(theta_meas)]
        ])
        cov = J @ np.diag([dr**2, dtheta**2]) @ J.T
        sensor_covariances.append(cov)

    # Weighted fusion of position estimates
    Sigma_inv_sum = np.zeros((2, 2))
    weighted_pos_sum = np.zeros(2)
    for i in range(3):
        cov_inv = np.linalg.inv(sensor_covariances[i])
        Sigma_inv_sum += cov_inv
        weighted_pos_sum += cov_inv @ np.array(sensor_positions[i])

    Sigma_pos = np.linalg.inv(Sigma_inv_sum)
    est_pos = Sigma_pos @ weighted_pos_sum
    est_positions.append(est_pos)
    est_covariances.append(Sigma_pos)  # Store estimated covariance matrix

    # --- Step 2: Estimate orientation using robot's own camera ---
    orientation_estimates = []
    weights = []
    for i, (lx, ly) in enumerate(landmarks):
        dx = lx - est_pos[0]
        dy = ly - est_pos[1]
        theta_global = np.arctan2(dy, dx)

        # Robot would measure relative bearing
        dx_true = lx - x
        dy_true = ly - y
        phi = np.arctan2(dy_true, dx_true) - theta
        phi += np.random.normal(0, robot_phi_error)

        psi_i = theta_global - phi
        psi_i = np.arctan2(np.sin(psi_i), np.cos(psi_i))

        # Error propagation
        d_sq = dx**2 + dy**2
        var_theta_global = ((dx**2 * Sigma_pos[1,1] + dy**2 * Sigma_pos[0,0] -
                            2 * dx * dy * Sigma_pos[0,1]) / d_sq**2)
        var_psi_i = var_theta_global + robot_phi_error**2
        weight = 1 / var_psi_i

        orientation_estimates.append(psi_i)
        weights.append(weight)

    weights = np.array(weights)
    orientation_estimates = np.array(orientation_estimates)
    sin_sum = np.sum(weights * np.sin(orientation_estimates))
    cos_sum = np.sum(weights * np.cos(orientation_estimates))
    psi_est = np.arctan2(sin_sum, cos_sum)
    est_orientations.append(psi_est)

# Convert to arrays
true_positions = np.array(true_positions)
est_positions = np.array(est_positions)
true_orientations = np.unwrap(np.array(true_orientations))
est_orientations = np.unwrap(np.array(est_orientations))

# Extract covariance elements for DataFrame
cov_xx = [cov[0, 0] for cov in est_covariances]
cov_xy = [cov[0, 1] for cov in est_covariances]
cov_yx = [cov[1, 0] for cov in est_covariances]
cov_yy = [cov[1, 1] for cov in est_covariances]

# Compute RMSE
pos_rmse = np.sqrt(np.mean((true_positions - est_positions)**2))
angle_rmse = np.sqrt(np.mean((true_orientations - est_orientations)**2))

# Store results in a pandas DataFrame
df = pd.DataFrame({
    'True_X': true_positions[:, 0],
    'True_Y': true_positions[:, 1],
    'True_Theta_rad': true_orientations,
    'Est_X': est_positions[:, 0],
    'Est_Y': est_positions[:, 1],
    'Est_Theta_rad': est_orientations,
    'Cov_XX': cov_xx,
    'Cov_XY': cov_xy,
    'Cov_YX': cov_yx,
    'Cov_YY': cov_yy
})

# Export to CSV file
csv_filename = "simulation_results.csv"
df.to_csv(csv_filename, index=False)

###########################################################################

# Compute estimation error
error_x = df['Est_X'] - df['True_X']
error_y = df['Est_Y'] - df['True_Y']

# Create error vectors and covariance matrices for each time step
error_vectors = np.vstack((error_x, error_y)).T
cov_matrices = np.array([[ [row['Cov_XX'], row['Cov_XY']], 
                           [row['Cov_YX'], row['Cov_YY']] ] for _, row in df.iterrows()])

# Calculate NEES
nees_values = np.array([e.T @ np.linalg.inv(P) @ e for e, P in zip(error_vectors, cov_matrices)])

# Calculate ANEES
anees = np.mean(nees_values)

# Chi-squared consistency bounds
dof = 2
confidence_level = 0.95
n = len(nees_values)
lower_bound = chi2.ppf((1 - confidence_level) / 2, dof * n) / n
upper_bound = chi2.ppf(1 - (1 - confidence_level) / 2, dof * n) / n

# Print results
consistency = None
print(f"Average NEES (ANEES): {anees:.4f}")
print(f"95% Confidence Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
if lower_bound <= anees <= upper_bound:
    print("Estimates are statistically consistent.")
    consistency = "Estimates are statistically consistent."
else:
    print("Estimates are NOT statistically consistent.")
    consistency = "Estimates are NOT statistically consistent."




##########################################################################



# Animation
fig, ax = plt.subplots(figsize=(20, 20))

x_min = min(np.min(true_positions[:, 0]), np.min(landmarks[:, 0])) - 2
x_max = max(np.max(true_positions[:, 0]), np.max(landmarks[:, 0])) + 2
y_min = min(np.min(true_positions[:, 1]), np.min(landmarks[:, 1])) - 2
y_max = max(np.max(true_positions[:, 1]), np.max(landmarks[:, 1])) + 2

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title(
    f"Animated Robot Pose Estimation\n"
    f"Position RMSE: {pos_rmse:.3f}, Orientation RMSE: {np.rad2deg(angle_rmse):.2f}°\n"
    f"Average NEES (ANEES): {anees:.4f}\n"
    f"95% Confidence Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]\n"
    f"{consistency}"   
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
ax.set_aspect('equal')

ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', marker='X', s=100, label='Landmarks')

true_path_line, = ax.plot([], [], 'g-', label='True Position')
est_path_line, = ax.plot([], [], 'r--', label='Estimated Position')
true_arrow = ax.quiver([], [], [], [], color='green', scale=10, width=0.005, label='True Orientation')
est_arrow = ax.quiver([], [], [], [], color='red', scale=10, width=0.005, label='Estimated Orientation')

ax.legend()

def update(frame):
    true_path_line.set_data(true_positions[:frame+1, 0], true_positions[:frame+1, 1])
    est_path_line.set_data(est_positions[:frame+1, 0], est_positions[:frame+1, 1])
    true_arrow.set_offsets(true_positions[frame])
    est_arrow.set_offsets(est_positions[frame])
    true_u = np.cos(true_orientations[frame])
    true_v = np.sin(true_orientations[frame])
    est_u = np.cos(est_orientations[frame])
    est_v = np.sin(est_orientations[frame])
    true_arrow.set_UVC(true_u, true_v)
    est_arrow.set_UVC(est_u, est_v)
    return true_path_line, est_path_line, true_arrow, est_arrow

ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=300, blit=False)
plt.show()


##########################################################################
'''
# Static Plot for Project Report
plt.figure(figsize=(10, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'g.-', label='True Position')
plt.plot(est_positions[:, 0], est_positions[:, 1], 'r.-', label='Estimated Position')
plt.quiver(est_positions[:, 0], est_positions[:, 1], np.cos(est_orientations), np.sin(est_orientations), 
           color='r', width=0.005, scale=10, label='Estimated Orientation')
plt.quiver(true_positions[:, 0], true_positions[:, 1], np.cos(true_orientations), np.sin(true_orientations), 
           color='g', width=0.005, scale=10, label='True Orientation')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='b', marker='X', s=100, label='Landmarks')
plt.title(f"Robot Position and Orientation Estimation\nPosition RMSE: {pos_rmse:.3f}, Orientation RMSE: {np.rad2deg(angle_rmse):.2f}°")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
##########################################################################

ani.save("robot_pose_estimation.gif", writer='pillow', fps=3)



