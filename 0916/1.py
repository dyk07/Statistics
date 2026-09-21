import os
import numpy as np
import matplotlib.pyplot as plt

# Define the height categories based on the table's rows and columns
# We estimate the extreme bins ("74 and above" as 74.5, "Below 63" as 62.5) to keep the 1-inch intervals
men_heights = [74.5, 73.5, 72.5, 71.5, 70.5, 69.5, 68.5, 67.5, 66.5, 65.5, 64.5, 63.5, 62.5]
brother_heights = [62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5]

# The 2D frequency matrix from the center of the table (0 represents '..')
frequencies = [
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 5, 3, 12],
    [0, 0, 0, 0, 0, 1, 3, 4, 8, 3, 3, 2, 3],
    [0, 0, 0, 0, 1, 1, 6, 5, 9, 9, 8, 3, 5],
    [0, 1, 0, 1, 2, 8, 11, 18, 14, 20, 9, 4, 0],
    [0, 0, 1, 1, 7, 19, 30, 45, 36, 14, 9, 8, 1],
    [0, 1, 2, 1, 11, 20, 36, 55, 44, 17, 5, 4, 2],
    [0, 1, 5, 9, 18, 38, 46, 36, 30, 11, 6, 3, 0],
    [2, 4, 8, 26, 35, 38, 38, 20, 18, 8, 1, 1, 0],
    [4, 3, 10, 33, 28, 35, 20, 12, 7, 2, 1, 0, 0],
    [3, 3, 15, 18, 33, 36, 8, 2, 1, 1, 0, 0, 0],
    [3, 8, 12, 15, 10, 8, 5, 2, 1, 0, 0, 0, 0],
    [5, 2, 8, 3, 3, 4, 1, 1, 0, 1, 0, 0, 1],
    [5, 5, 3, 3, 4, 2, 0, 0, 0, 0, 0, 0, 1]
]

# Expand the matrix into flat lists of every individual data point for the regression
x_data_expanded = []
y_data_expanded = []

# Prepare data specifically for the bubble chart (unique pairs and their counts)
x_bubble = []
y_bubble = []
bubble_sizes = []

for i, man_h in enumerate(men_heights):
    for j, bro_h in enumerate(brother_heights):
        count = frequencies[i][j]
        if count > 0:
            # Add to bubble chart data
            x_bubble.append(man_h)
            y_bubble.append(bro_h)
            bubble_sizes.append(count * 8)  # Multiply by a scalar to make bubbles visible

            # Expand the data for accurate regression calculation
            x_data_expanded.extend([man_h] * count)
            y_data_expanded.extend([bro_h] * count)

# Convert to numpy arrays for calculation
x_data_expanded = np.array(x_data_expanded)
y_data_expanded = np.array(y_data_expanded)

# Calculate linear regression using all 1300+ data points
slope, intercept = np.polyfit(x_data_expanded, y_data_expanded, 1)

# Generate points for plotting the lines
line_x = np.array([62, 75])
regression_line = slope * line_x + intercept
equality_line = line_x  # y = x

# --- Plotting ---
plt.figure(figsize=(11, 8))

# Plot the bubble chart
plt.scatter(x_bubble, y_bubble, s=bubble_sizes, alpha=0.5, color='blue',
            label='Data Frequencies (Bubble Size = Count)')

# Plot the regression line
plt.plot(line_x, regression_line, color='red', linewidth=2.5,
         label=f'Regression Line (y = {slope:.3f}x + {intercept:.3f})')

# Plot the line of perfect equality
plt.plot(line_x, equality_line, color='green', linestyle='--', linewidth=2,
         label='Line of Equality (y = x)')

# Formatting
plt.title("Galton's Data: Height of Men vs. Height of Their Brothers", fontsize=14)
plt.xlabel("Height of Men (inches)", fontsize=12)
plt.ylabel("Height of Brothers (inches)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# --- Save the figure to the same folder as this script ---
try:
    # Folder where this script resides (works when running a .py file)
    output_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments (e.g., Jupyter / IDLE without __file__)
    output_dir = os.getcwd()

output_path = os.path.join(output_dir, "galton_heights.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()

# Print Analysis
print(f"Total cases analyzed: {len(x_data_expanded)}")
print(f"Calculated Regression Function: y = {slope:.4f}x + {intercept:.4f}")