

import numpy as np

# Task 1: Range of Daily Step Counts

steps = np.array([6500, 7200, 8000, 6000, 9000, 7500, 8200])
step_min = steps.min()
step_max = steps.max()
step_range = step_max - step_min

print("=" * 70)
print("TASK 1: Range of Daily Step Counts")
print("=" * 70)
print(f"Daily Steps: {steps.tolist()}")
print(f"Minimum Steps: {step_min}")
print(f"Maximum Steps: {step_max}")
print(f"Range: {step_range} steps")
print("\nInterpretation:")
print("A range of 3,000 steps indicates moderate daily variability. The user maintains")
print("a consistent baseline activity level (at least 6,000 steps on their least active day)")
print("and occasionally reaches higher activity peaks (9,000 steps), reflecting regular daily movement.")


# Task 2: Variance and Standard Deviation of Zomato Order Amounts

zomato_orders = np.array([250, 400, 350, 300, 500])

# Sample metrics (ddof=1)
var_sample = np.var(zomato_orders, ddof=1)
std_sample = np.std(zomato_orders, ddof=1)

# Population metrics (ddof=0)
var_pop = np.var(zomato_orders, ddof=0)
std_pop = np.std(zomato_orders, ddof=0)

print("\n" + "=" * 70)
print("TASK 2: Variance & Standard Deviation of Zomato Orders")
print("=" * 70)
print(f"Order Amounts (INR): {zomato_orders.tolist()}")
print(f"Mean Order Amount: ₹{np.mean(zomato_orders):.2f}")
print(f"Sample Variance (s²): {var_sample:.2f} ₹²")
print(f"Sample Std Dev (s):   ₹{std_sample:.2f}")
print(f"Population Variance (σ²): {var_pop:.2f} ₹²")
print(f"Population Std Dev (σ):   ₹{std_pop:.2f}")

# Task 3: Coefficient of Variation (CV) for Mobile Data Usage

data_usage = np.array([3.2, 2.8, 3.5, 3.0, 2.9, 3.1])

mean_usage = np.mean(data_usage)
std_usage_sample = np.std(data_usage, ddof=1)
cv_sample = (std_usage_sample / mean_usage) * 100

print("\n" + "=" * 70)
print("TASK 3: Coefficient of Variation (CV) for Monthly Data Usage")
print("=" * 70)
print(f"Monthly Data Usage (GB): {data_usage.tolist()}")
print(f"Mean Usage: {mean_usage:.4f} GB")
print(f"Standard Deviation (Sample): {std_usage_sample:.4f} GB")
print(f"Coefficient of Variation (CV): {cv_sample:.2f}%")
print("\nInterpretation:")
print(f"With a CV of {cv_sample:.2f}% (< 10%), the user's mobile data consumption is")
print("highly consistent and stable across the 6-month observation period.")

# Task 4: Standard Deviation vs. Mean for Instagram Post Likes

likes = np.array([120, 150, 130, 110, 170, 140, 135, 160, 125, 155])

mean_likes = np.mean(likes)
std_likes = np.std(likes, ddof=1)
cv_likes = (std_likes / mean_likes) * 100

print("\n" + "=" * 70)
print("TASK 4: Instagram Likes - Consistency of Engagement")
print("=" * 70)
print(f"Likes Dataset: {likes.tolist()}")
print(f"Mean Likes (x̄): {mean_likes:.2f}")
print(f"Standard Deviation (s): {std_likes:.2f} likes")
print(f"Coefficient of Variation (CV): {cv_likes:.2f}%")
print("\nInterpretation:")
print(f"The standard deviation of {std_likes:.2f} likes is relatively small compared to the mean")
print(f"of {mean_likes:.2f} likes (CV = {cv_likes:.2f}%). This confirms that post engagement is")
print("predictable and consistent, with most posts reliably receiving around 120-160 likes.")

# Task 5: AI-Generated Daily UPI Spending & Stability Analysis

# Synthetic 8-day dataset representing daily routine UPI expenditures (tea, lunch, groceries)
daily_upi = np.array([140, 220, 180, 150, 210, 190, 160, 250])

mean_upi = np.mean(daily_upi)
var_upi = np.var(daily_upi, ddof=1)
std_upi = np.std(daily_upi, ddof=1)
cv_upi = (std_upi / mean_upi) * 100

print("\n" + "=" * 70)
print("TASK 5: Daily UPI Spending Stability Analysis")
print("=" * 70)
print(f"Daily UPI Transactions (INR): {daily_upi.tolist()}")
print(f"Mean Spending: ₹{mean_upi:.2f}")
print(f"Variance (s²): {var_upi:.2f} ₹²")
print(f"Standard Deviation (s): ₹{std_upi:.2f}")
print(f"Coefficient of Variation (CV): {cv_upi:.2f}%")
print("\nInterpretation:")
print(f"The low standard deviation of ₹{std_upi:.2f} around an average daily expense of ₹{mean_upi:.2f}")
print("(CV ~20%) indicates healthy spending stability without erratic luxury spikes or unpredictable expenses.")
