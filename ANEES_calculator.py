# Create a complete Python script as a single file with all steps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Load the CSV file
df = pd.read_csv("simulation_results.csv")

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
print(f"Average NEES (ANEES): {anees:.4f}")
print(f"95% Confidence Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
if lower_bound <= anees <= upper_bound:
    print("Estimates are statistically consistent.")
else:
    print("Estimates are NOT statistically consistent.")

'''
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(nees_values, label='NEES per timestep')
plt.axhline(y=anees, color='r', linestyle='--', label=f'ANEES = {anees:.2f}')
plt.axhline(y=lower_bound, color='g', linestyle=':', label='95% Confidence Lower Bound')
plt.axhline(y=upper_bound, color='g', linestyle=':', label='95% Confidence Upper Bound')
plt.title('Normalized Estimation Error Squared (NEES) Over Time')
plt.xlabel('Timestep')
plt.ylabel('NEES')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("nees_plot.png")
plt.show()
'''
