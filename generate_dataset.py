import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_hotel_dataset(filename="13_hotel_bookings.csv", n_records=1200, seed=42):
    np.random.seed(seed)
    
    # 1. Booking IDs
    booking_ids = [f"HB{10000 + i}" for i in range(1, n_records + 1)]
    
    # 2. Checkin dates spanning 2024 (Jan 1, 2024 to Dec 31, 2024)
    start_date = datetime(2024, 1, 1)
    date_offsets = np.random.randint(0, 365, size=n_records)
    checkin_dates = [start_date + timedelta(days=int(offset)) for offset in date_offsets]
    
    # Seasonality weights for peak months (summer: June, July, August; holidays: December)
    # 3. Room Types: Standard (55%), Deluxe (30%), Suite (15%)
    room_types = np.random.choice(["Standard", "Deluxe", "Suite"], size=n_records, p=[0.55, 0.30, 0.15])
    
    # 4. Booking Channels: OTA (50%), Direct Website (30%), Travel Agent (20%)
    channels = np.random.choice(["OTA", "Direct Website", "Travel Agent"], size=n_records, p=[0.50, 0.30, 0.20])
    
    # 5. Customer Type: Leisure (65%), Business (35%)
    customer_types = np.random.choice(["Leisure", "Business"], size=n_records, p=[0.65, 0.35])
    
    # 6. Stay Nights:
    # Business stays: mostly 1-3 nights. Leisure stays: mostly 2-7 nights.
    nights = []
    for c_type in customer_types:
        if c_type == "Business":
            n = np.random.choice([1, 2, 3, 4], p=[0.40, 0.35, 0.15, 0.10])
        else:
            n = np.random.choice([2, 3, 4, 5, 6, 7], p=[0.20, 0.30, 0.25, 0.12, 0.08, 0.05])
        nights.append(n)
    nights = np.array(nights)
    
    # 7. Lead Time Days:
    # Exponential / gamma distribution skewed towards shorter lead times
    lead_times = np.random.gamma(shape=2.0, scale=18.0, size=n_records).astype(int)
    # Clip to realistic range 0 to 180
    lead_times = np.clip(lead_times, 0, 180)
    
    # 8. Room Rate ($ / night):
    # Base rates: Standard ~ $90-120, Deluxe ~ $140-190, Suite ~ $220-320
    # Add seasonal surge in June-August and December
    room_rates = []
    for r_type, dt in zip(room_types, checkin_dates):
        month = dt.month
        surge = 1.25 if month in [6, 7, 8, 12] else 1.0
        if r_type == "Standard":
            rate = np.random.normal(105, 12) * surge
        elif r_type == "Deluxe":
            rate = np.random.normal(165, 18) * surge
        else: # Suite
            rate = np.random.normal(260, 30) * surge
        room_rates.append(round(rate, 2))
    room_rates = np.array(room_rates)
    
    # Add a few realistic luxury/presidential suite outliers for IQR analysis (Task 13)
    outlier_indices = np.random.choice(n_records, size=8, replace=False)
    for idx in outlier_indices:
        room_rates[idx] = round(room_rates[idx] * 2.5 + 150, 2)
        
    # 9. Cancellation Status (Is_Canceled):
    # Cancellation rate depends on channel and lead time:
    # OTAs and high lead times have higher cancellations
    is_canceled = []
    for ch, lt in zip(channels, lead_times):
        # Base probability
        prob = 0.15
        if ch == "OTA":
            prob += 0.15
        elif ch == "Travel Agent":
            prob += 0.08
        else: # Direct Website
            prob += 0.02
        
        # Lead time impact: longer lead time -> higher cancellation
        if lt > 45:
            prob += 0.18
        elif lt > 15:
            prob += 0.08
            
        prob = min(max(prob, 0.05), 0.70)
        is_canceled.append(1 if np.random.rand() < prob else 0)
    is_canceled = np.array(is_canceled)
    
    # Format dates as YYYY-MM-DD
    date_strs = [dt.strftime("%Y-%m-%d") for dt in checkin_dates]
    
    # Build DataFrame
    df = pd.DataFrame({
        "Booking_ID": booking_ids,
        "Checkin_Date": date_strs,
        "Lead_Time_Days": lead_times.astype(float),
        "Room_Type": room_types,
        "Room_Rate": room_rates,
        "Nights": nights,
        "Booking_Channel": channels,
        "Customer_Type": customer_types,
        "Is_Canceled": is_canceled
    })
    
    # Inject ~25 missing values in Lead_Time_Days and ~20 in Room_Rate (Task 2 & 3)
    missing_lt_idx = np.random.choice(n_records, size=25, replace=False)
    df.loc[missing_lt_idx, "Lead_Time_Days"] = np.nan
    
    missing_rr_idx = np.random.choice(n_records, size=20, replace=False)
    df.loc[missing_rr_idx, "Room_Rate"] = np.nan
    
    # Inject 6 duplicate rows to test duplicate handling (Task 2 & 3)
    duplicate_rows = df.sample(n=6, random_state=101)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Shuffle slightly so duplicates aren't all at the bottom
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset generated successfully: {filename}")
    print(f"Total rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"Missing Lead_Time_Days: {df['Lead_Time_Days'].isna().sum()}")
    print(f"Missing Room_Rate: {df['Room_Rate'].isna().sum()}")
    print(f"Duplicate Booking_IDs: {df.duplicated(subset=['Booking_ID']).sum()}")

if __name__ == "__main__":
    generate_hotel_dataset("C:/Users/Dell/.gemini/antigravity/scratch/hotel-booking-analysis/13_hotel_bookings.csv")
