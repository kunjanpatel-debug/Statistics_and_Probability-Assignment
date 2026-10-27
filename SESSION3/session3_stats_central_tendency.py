

import numpy as np
from collections import Counter

# Task 1: Mean of 7 Daily Step Counts

daily_steps = [6500, 7200, 8100, 6900, 10200, 7500, 8400]

def calculate_mean_steps(steps_list):
    return sum(steps_list) / len(steps_list)

mean_steps = calculate_mean_steps(daily_steps)
print("=" * 65)
print("TASK 1: Mean Daily Step Count")
print("=" * 65)
print(f"Daily Steps: {daily_steps}")
print(f"Mean Steps:  {mean_steps:.2f}")


# Task 2: Median Delivery Time Function

delivery_times = [32, 28, 29, 45, 30, 31, 60, 30, 29]

def find_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    else:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2

median_delivery = find_median(delivery_times)
print("\n" + "=" * 65)
print("TASK 2: Median Food Delivery Time")
print("=" * 65)
print(f"Original Times: {delivery_times}")
print(f"Sorted Times:   {sorted(delivery_times)}")
print(f"Median Delivery Time: {median_delivery} minutes")


# Task 3: Mode of YouTube Genres

genres = [
    'Music', 'Vlog', 'Music', 'Tech', 'Music', 'Vlog',
    'Tech', 'Music', 'Comedy', 'Vlog', 'Music', 'Comedy'
]

def find_mode(data):
    counts = Counter(data)
    max_freq = max(counts.values())
    modes = [key for key, val in counts.items() if val == max_freq]
    return modes[0] if len(modes) == 1 else modes, max_freq, counts

mode_genre, freq, all_counts = find_mode(genres)
print("\n" + "=" * 65)
print("TASK 3: Mode of Watched YouTube Genres")
print("=" * 65)
print(f"Genre Counts: {dict(all_counts)}")
print(f"Mode (Most Watched Genre): '{mode_genre}' (watched {freq} times)")


# Task 4: UPI Transaction Amounts (Mean vs. Median & Outliers)

upi_amounts = [100, 120, 105, 110, 5000, 115, 108]

mean_upi = sum(upi_amounts) / len(upi_amounts)
median_upi = find_median(upi_amounts)

print("\n" + "=" * 65)
print("TASK 4: Daily UPI Transactions (Outlier Analysis)")
print("=" * 65)
print(f"UPI Amounts: {upi_amounts}")
print(f"Mean:   ₹{mean_upi:.2f}")
print(f"Median: ₹{median_upi:.2f}")
print("\nComparison Explanation:")
print(
    "The median (₹110.00) is a much better measure of a typical transaction.\n"
    "The value ₹5,000 is an extreme outlier (such as rent or a big purchase), "
    "which skews the mean up to ₹808.29—a figure that is over 7 times higher "
    "than almost every single daily transaction."
)
