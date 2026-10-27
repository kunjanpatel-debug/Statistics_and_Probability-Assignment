

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Task 1: 25th, 50th, and 75th Percentiles for 20 Zomato Delivery Times

delivery_times = np.array([
    22, 25, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 38, 40, 41, 42, 45, 48, 50, 52, 58
])

p25 = np.percentile(delivery_times, 25)
p50 = np.percentile(delivery_times, 50)
p75 = np.percentile(delivery_times, 75)

print("=" * 70)
print("TASK 1: Zomato Delivery Times Percentiles")
print("=" * 70)
print(f"Delivery Times (minutes): {delivery_times.tolist()}")
print(f"25th Percentile (Q1): {p25:.2f} mins")
print(f"50th Percentile (Median / Q2): {p50:.2f} mins")
print(f"75th Percentile (Q3): {p75:.2f} mins")


# Task 2: Dividing IPL Player Scores into Quartiles using numpy.quantile

ipl_scores = np.array([12, 18, 24, 31, 35, 42, 45, 52, 58, 64, 71, 78, 85, 92, 104, 118])

q1 = np.quantile(ipl_scores, 0.25)
q2 = np.quantile(ipl_scores, 0.50)
q3 = np.quantile(ipl_scores, 0.75)

print("\n" + "=" * 70)
print("TASK 2: IPL Player Scores Quartiles (numpy.quantile)")
print("=" * 70)
print(f"IPL Scores: {ipl_scores.tolist()}")
print(f"Q1 (25% Quartile): {q1:.2f} runs")
print(f"Q2 (50% Median):   {q2:.2f} runs")
print(f"Q3 (75% Quartile): {q3:.2f} runs")


# Task 3: 1 Month Steps IQR and Activity Consistency Analysis

steps = np.array([
    6200, 6500, 6700, 7000, 7100, 7200, 7300, 7400, 7500, 7600,
    7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700,
    8900, 9000, 9100, 9200, 9400, 9600, 9800, 10200, 10500, 11200
])

step_q1 = np.percentile(steps, 25)
step_q3 = np.percentile(steps, 75)
step_iqr = step_q3 - step_q1

print("\n" + "=" * 70)
print("TASK 3: Interquartile Range (IQR) for Daily Steps (1 Month)")
print("=" * 70)
print(f"Q1 (25th Percentile): {step_q1:.2f} steps")
print(f"Q3 (75th Percentile): {step_q3:.2f} steps")
print(f"IQR (Q3 - Q1):        {step_iqr:.2f} steps")
print("\nActivity Consistency Interpretation:")
print(f"The middle 50% of the user's daily steps fall within an IQR band of {step_iqr:.0f} steps")
print("(from 7,425 to 9,075 steps). A compact IQR relative to the overall range confirms high")
print("lifestyle consistency, indicating disciplined daily routine without erratic swings.")


# Task 4: IQR Method Outlier Detection Function for Swiggy Order Amounts

swiggy_orders = [180, 220, 240, 250, 260, 280, 300, 310, 320, 350, 380, 400, 420, 450, 850, 120, 1450]

def detect_outliers_iqr(data):
    arr = np.array(data)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
    return q1, q3, iqr, lower_bound, upper_bound, outliers.tolist()

q1_s, q3_s, iqr_s, lb_s, ub_s, outliers_s = detect_outliers_iqr(swiggy_orders)

print("\n" + "=" * 70)
print("TASK 4: Swiggy Order Amounts Outlier Detection via IQR Method")
print("=" * 70)
print(f"Order Amounts (INR): {swiggy_orders}")
print(f"Q1: ₹{q1_s:.2f}, Q3: ₹{q3_s:.2f}, IQR: ₹{iqr_s:.2f}")
print(f"Lower Bound (Q1 - 1.5*IQR): ₹{lb_s:.2f}")
print(f"Upper Bound (Q3 + 1.5*IQR): ₹{ub_s:.2f}")
print(f"Identified Outliers: {outliers_s}")


# Task 5: Matplotlib Boxplot of Weekly Screen Time (8 Weeks)

screen_time = np.array([28.5, 31.0, 29.5, 32.0, 30.5, 33.0, 29.0, 48.5])

plt.figure(figsize=(8, 5))
box = plt.boxplot(
    screen_time,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor='#D1E7DD', color='#0F5132', linewidth=1.5),
    medianprops=dict(color='#DC3545', linewidth=2),
    whiskerprops=dict(color='#0F5132', linewidth=1.5),
    capprops=dict(color='#0F5132', linewidth=1.5),
    flierprops=dict(marker='o', markerfacecolor='#DC3545', markeredgecolor='black', markersize=9, linestyle='none')
)

plt.title('Weekly Screen Time Distribution (Past 8 Weeks)', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('Screen Time (Hours / Week)', fontsize=11)
plt.xticks([1], ['User Screen Time'], fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)

outlier_val = screen_time.max()
plt.annotate(
    f'Outlier: {outlier_val} hrs\n(Binge-watching / Exam week)',
    xy=(1, outlier_val),
    xytext=(1.15, outlier_val - 2),
    arrowprops=dict(facecolor='#DC3545', shrink=0.08, width=1.5, headwidth=7),
    fontsize=9.5,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3CD', edgecolor='#FFEEBA', alpha=0.95)
)

plt.tight_layout()
plt.savefig('session5_task5_boxplot.png', dpi=150)
plt.close()
print("\n" + "=" * 70)
print("TASK 5: Boxplot saved as session5_task5_boxplot.png")
print("=" * 70)
