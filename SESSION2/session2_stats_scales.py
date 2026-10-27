

import pandas as pd
import numpy as np


# TASK 1: Classification of Mock Zomato Columns

task1_data = {
    "Column Name": [
        "Restaurant Name", 
        "Cuisine Type", 
        "Average Cost for Two", 
        "User Rating", 
        "City", 
        "Open/Closed Status"
    ],
    "Primary Classification": [
        "Categorical", 
        "Categorical", 
        "Numerical", 
        "Categorical / Numerical", 
        "Categorical", 
        "Categorical"
    ],
    "Sub-Classification": [
        "Nominal", 
        "Nominal", 
        "Continuous", 
        "Ordinal (or Continuous score)", 
        "Nominal", 
        "Nominal (Binary)"
    ],
    "Measurement Scale": [
        "Nominal", 
        "Nominal", 
        "Ratio", 
        "Ordinal (Ranked satisfaction)", 
        "Nominal", 
        "Nominal"
    ],
    "Justification": [
        "Labels identifying distinct entities with no natural mathematical ordering.",
        "Qualitative cuisines (North Indian, Italian, etc.) with no hierarchy.",
        "Monetary expenditure on a continuous continuum with an absolute zero.",
        "Ordered star rating (1.0 to 5.0) reflecting ranked satisfaction levels.",
        "Geographic qualitative labels without arithmetic priority.",
        "Mutually exclusive binary states (Open vs. Closed)."
    ]
}

df_task1 = pd.DataFrame(task1_data)

# TASK 2: 8 Real-Life App Examples Across Measurement Scales

task2_data = {
    "App Name": [
        "Flipkart", 
        "Flipkart", 
        "Instagram", 
        "Instagram", 
        "IRCTC", 
        "IRCTC", 
        "IRCTC", 
        "Zomato / Swiggy"
    ],
    "Column / Metric": [
        "Payment Method (UPI, Card, COD)", 
        "Delivery Speed Tier (Standard, Express, Same-Day)", 
        "Number of Post Likes", 
        "Close Friends Flag (Yes / No)", 
        "Scheduled Departure Time (14:30 IST)", 
        "Travel Class (1A, 2A, 3A, Sleeper)", 
        "Ticket Fare Amount (in INR)", 
        "Delivery Distance (in Kilometers)"
    ],
    "Measurement Scale": [
        "Nominal", 
        "Ordinal", 
        "Ratio", 
        "Nominal", 
        "Interval", 
        "Ordinal", 
        "Ratio", 
        "Ratio"
    ],
    "Explanation": [
        "Distinct qualitative categories with no mathematical ranking or order.",
        "Ordered hierarchy of delivery speeds with non-uniform time differences.",
        "Quantitative count possessing a true, non-arbitrary zero point.",
        "Mutually exclusive binary indicator used to filter visibility.",
        "Clock time with equal intervals but an arbitrary origin (00:00 midnight).",
        "Hierarchical comfort categories where fare differences are not fixed.",
        "Continuous monetary value with a true zero; ratio calculations are valid.",
        "Physical continuous distance where 0 km represents zero displacement."
    ]
}

df_task2 = pd.DataFrame(task2_data)

# TASK 3: Spotify Listening History Classification

task3_explanation = """
Spotify Listening History Analysis:
1. 'Date' -> Time-Series / Temporal Index
   Reasoning: Anchors each observation chronologically across time intervals.
2. 'Song Name' -> Categorical (Nominal)
   Reasoning: Text label identifying distinct songs without mathematical ranking.
3. 'Play Count per Day' -> Numerical (Discrete / Ratio Scale)
   Reasoning: Whole integer count metric with a true zero (0 plays = no listening).
"""

# TASK 4: Automatic Measurement Scale Detector

def detect_measurement_scale(series: pd.Series) -> str:
    """
    Heuristic classifier that detects whether a data column is
    Nominal, Ordinal, Interval, or Ratio scale.
    """
    clean_s = series.dropna()
    if len(clean_s) == 0:
        return "Unknown (Empty Column)"

    # Rule 1: Temporal / Datetime check (Interval)
    if pd.api.types.is_datetime64_any_dtype(clean_s):
        return "Interval (Datetime column with calendar intervals)"
    try:
        if isinstance(clean_s.iloc[0], str) and len(str(clean_s.iloc[0])) >= 8:
            pd.to_datetime(clean_s.iloc[:5])
            return "Interval (Parsed Timestamp with equal time intervals)"
    except Exception:
        pass

    # Rule 2: Numeric Data (Check for Ordinal Ratings or Ratio metric)
    if pd.api.types.is_numeric_dtype(clean_s):
        unique_vals = set(clean_s.unique())
        # Ratings (1 to 5 or 1 to 10)
        if len(unique_vals) <= 10 and clean_s.min() >= 1 and clean_s.max() <= 10:
            if unique_vals.issubset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}):
                return "Ordinal (Discrete Likert/Rating score)"
        
        # Bounded by natural zero (Amount, Count, Weight)
        if (clean_s >= 0).all():
            return "Ratio (Quantitative metric with absolute true zero)"
        
        return "Interval (Signed continuous metric without true zero)"

    # Rule 3: Text Categories matching Ordinal sequences
    known_order_patterns = [
        {"low", "medium", "high"},
        {"bronze", "silver", "gold", "platinum"},
        {"poor", "average", "good", "excellent"},
        {"placed", "shipped", "out for delivery", "delivered"}
    ]
    str_vals = set(clean_s.astype(str).str.lower().unique())
    for pattern in known_order_patterns:
        if str_vals.issubset(pattern) or len(str_vals.intersection(pattern)) >= 2:
            return "Ordinal (Text categories with natural progression)"

    # Rule 4: Default qualitative text fallback
    return "Nominal (Unordered qualitative identifiers or categories)"


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 1: Zomato Column Classification")
    print("=" * 70)
    print(df_task1[["Column Name", "Primary Classification", "Sub-Classification", "Measurement Scale"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TASK 2: 8 Real-Life App Examples Across Measurement Scales")
    print("=" * 70)
    print(df_task2[["App Name", "Column / Metric", "Measurement Scale"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TASK 3: Spotify Listening History")
    print("=" * 70)
    print(task3_explanation.strip())

    print("\n" + "=" * 70)
    print("TASK 4: Test Automatic Measurement Scale Detector")
    print("=" * 70)
    mock_flipkart_data = {
        "order_id": ["OD101", "OD102", "OD103", "OD104"],
        "order_date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
        "order_amount": [1299.0, 450.0, 3200.5, 890.0],
        "delivery_status": ["Placed", "Shipped", "Out for Delivery", "Delivered"],
        "customer_rating": [5, 4, 3, 5]
    }
    df_test = pd.DataFrame(mock_flipkart_data)
    for col in df_test.columns:
        result = detect_measurement_scale(df_test[col])
        print(f"Column '{col}': {result}")
