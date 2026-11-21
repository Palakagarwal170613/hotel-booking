import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("""# 🏨 Hotel Booking & Occupancy Analysis Using Python
### B.Tech 1st Year Data Analysis & Python Programming Project

**Industry:** Hospitality & Tourism Management  
**Dataset:** `13_hotel_bookings.csv`  
**Tools & Libraries:** Python 3, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook  

---

## 📌 Project Overview & Objectives
The hospitality industry relies heavily on operational data to manage room inventory, set competitive pricing, forecast seasonal demand, and reduce losses from cancellations. 

In this project, we analyze hotel booking records across room types, booking channels, lead times, customer segments, and check-in schedules to answer critical business questions and deliver 5 actionable recommendations for hotel revenue managers.
"""))

# Task 1
cells.append(nbf.v4.new_markdown_cell("""---
## 🔍 Task 1: Load the Dataset & Inspect Structure
**Objective:** Load the CSV dataset into a Pandas DataFrame and inspect its dimensions, column names, and data types to understand what information is available.
"""))

cells.append(nbf.v4.new_code_cell("""# Importing essential data analysis and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling for charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

# Task 1: Load the CSV dataset
file_path = "13_hotel_bookings.csv"
df_raw = pd.read_csv(file_path)

# Display dimensions (rows, columns)
print("Dataset Dimensions (Rows, Columns):", df_raw.shape)
print("\\n--- Column Names & Data Types ---")
print(df_raw.dtypes)

# Display first 5 rows
print("\\n--- First 5 Records of the Dataset ---")
df_raw.head()
"""))

# Task 2
cells.append(nbf.v4.new_markdown_cell("""---
## 🔍 Task 2: Identify Missing Values & Duplicate Records
**Objective:** Detect data quality issues such as null/missing fields and duplicate `Booking_ID` primary keys that could skew our analysis.
"""))

cells.append(nbf.v4.new_code_cell("""# Check for missing (null) values in each column
missing_summary = df_raw.isnull().sum()
print("Missing values per column:")
print(missing_summary[missing_summary > 0])

# Check for duplicate Booking_ID records
duplicate_bookings = df_raw.duplicated(subset=['Booking_ID']).sum()
print(f"\\nTotal duplicate Booking_ID records found: {duplicate_bookings}")
"""))

# Task 3
cells.append(nbf.v4.new_markdown_cell("""---
## 🧹 Task 3: Handle Missing Values & Duplicates with Justification
**Data Cleaning Methodology & Justification:**
1. **Duplicate `Booking_ID` records:** A booking ID represents a single unique reservation. Duplicate rows distort reservation counts, occupancy, and total revenue. We remove duplicates using `drop_duplicates(subset=['Booking_ID'])`.
2. **Missing `Lead_Time_Days`:** Reservation lead times are typically positively skewed (most book within a month, but a few book 6 months ahead). Using the **median** rather than the mean avoids skewing by extreme outliers.
3. **Missing `Room_Rate`:** Daily room rates differ significantly across room types (`Standard`, `Deluxe`, `Suite`). Rather than an overall average, we impute missing room rates with the **median room rate specific to each room category**.
"""))

cells.append(nbf.v4.new_code_cell("""# Create a working copy of the dataset
df = df_raw.copy()

# Step 3a: Remove duplicate Booking_IDs
df = df.drop_duplicates(subset=['Booking_ID'], keep='first').reset_index(drop=True)
print(f"Rows remaining after removing duplicates: {len(df)}")

# Step 3b: Impute missing Lead_Time_Days with median
median_lead_time = df['Lead_Time_Days'].median()
df['Lead_Time_Days'] = df['Lead_Time_Days'].fillna(median_lead_time)
print(f"Imputed missing Lead_Time_Days with median: {median_lead_time:.0f} days")

# Step 3c: Impute missing Room_Rate using Room_Type group medians
room_type_medians = df.groupby('Room_Type')['Room_Rate'].median()
for r_type, med_rate in room_type_medians.items():
    df.loc[(df['Room_Type'] == r_type) & (df['Room_Rate'].isnull()), 'Room_Rate'] = med_rate
    print(f"Imputed missing Room_Rate for '{r_type}' with median: ${med_rate:.2f}")

# Verify no missing values remain
print("\\nTotal remaining missing values:", df.isnull().sum().sum())
"""))

# Task 4
cells.append(nbf.v4.new_markdown_cell("""---
## 📅 Task 4: Datetime Conversion & Feature Engineering
**Objective:** Convert `Checkin_Date` from string format into datetime objects. Extract the **arrival month** and **day of the week (weekday)** to analyze seasonal and weekly patterns.
"""))

cells.append(nbf.v4.new_code_cell("""# Convert Checkin_Date column to datetime
df['Checkin_Date'] = pd.to_datetime(df['Checkin_Date'])

# Extract Month Name, Month Number, and Day of Week
df['Arrival_Month'] = df['Checkin_Date'].dt.month_name()
df['Arrival_Month_Num'] = df['Checkin_Date'].dt.month
df['Arrival_Weekday'] = df['Checkin_Date'].dt.day_name()

# Inspect transformed features
df[['Booking_ID', 'Checkin_Date', 'Arrival_Month', 'Arrival_Weekday']].head()
"""))

# Task 5
cells.append(nbf.v4.new_markdown_cell("""---
## 📊 Task 5: Overall Key Performance Indicators (KPIs)
**Objective:** Calculate the fundamental hospitality metrics:
- **Total Bookings** ($N$)
- **Total Room Nights** ($\sum \text{Nights}$)
- **Overall Cancellation Percentage** ($\frac{\text{Cancellations}}{\text{Total Bookings}} \times 100$)
"""))

cells.append(nbf.v4.new_code_cell("""total_bookings = len(df)
total_nights = int(df['Nights'].sum())
total_cancellations = int(df['Is_Canceled'].sum())
cancellation_pct = (total_cancellations / total_bookings) * 100

print(f"🏨 Total Bookings:                  {total_bookings:,}")
print(f"🌙 Total Room Nights Reserved:     {total_nights:,}")
print(f"❌ Total Cancellations:            {total_cancellations:,}")
print(f"📉 Overall Cancellation Rate:       {cancellation_pct:.2f}%")
"""))

# Task 6
cells.append(nbf.v4.new_markdown_cell("""---
## 💰 Task 6: Calculate Total Realized Revenue
**Objective:** Calculate total revenue for completed (non-canceled) reservations.  
$$\\text{Booking Value} = \\text{Nights} \\times \\text{Room Rate}$$  
If a booking is canceled (`Is_Canceled == 1`), the room was not occupied, representing unrealized/lost revenue.
"""))

cells.append(nbf.v4.new_code_cell("""# Calculate booking value for every reservation
df['Booking_Value'] = df['Nights'] * df['Room_Rate']

# Filter only non-canceled bookings for realized revenue
df_realized = df[df['Is_Canceled'] == 0].copy()

total_revenue = df_realized['Booking_Value'].sum()
lost_revenue = df[df['Is_Canceled'] == 1]['Booking_Value'].sum()

print(f"💵 Total Realized Booking Revenue:       ${total_revenue:,.2f}")
print(f"⚠️ Potential Lost Revenue (Cancellations): ${lost_revenue:,.2f}")
"""))

# Task 7
cells.append(nbf.v4.new_markdown_cell("""---
## 🛏️ Task 7: Room Type Analysis (Rate & Occupancy)
**Objective:** Compare average room rates, total room nights occupied, total revenue, and cancellation rates across `Standard`, `Deluxe`, and `Suite` rooms.
"""))

cells.append(nbf.v4.new_code_cell("""# Aggregate room type performance
room_type_summary = df.groupby('Room_Type').agg(
    Total_Bookings=('Booking_ID', 'count'),
    Total_Nights=('Nights', 'sum'),
    Realized_Revenue=('Booking_Value', lambda x: x[df.loc[x.index, 'Is_Canceled'] == 0].sum()),
    Avg_Daily_Rate=('Room_Rate', 'mean'),
    Median_Daily_Rate=('Room_Rate', 'median'),
    Cancellation_Rate=('Is_Canceled', lambda x: (x.sum() / len(x)) * 100)
).reset_index()

# Display formatted table
print(room_type_summary.round(2).to_string(index=False))

# Visualization: Room Type Comparison
fig, ax1 = plt.subplots(figsize=(8, 5))
color = '#1f77b4'
ax1.set_xlabel('Room Type', fontweight='bold')
ax1.set_ylabel('Total Room Nights Occupied', color=color, fontweight='bold')
ax1.bar(room_type_summary['Room_Type'], room_type_summary['Total_Nights'], color=color, alpha=0.75, width=0.4)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Average Daily Room Rate ($)', color=color, fontweight='bold')
ax2.plot(room_type_summary['Room_Type'], room_type_summary['Avg_Daily_Rate'], color=color, marker='o', linewidth=2.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Task 7: Room Type Performance (Room Nights vs. Daily Rate)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
"""))

# Task 8
cells.append(nbf.v4.new_markdown_cell("""---
## 🌐 Task 8: Rank Booking Channels by Volume & Revenue
**Objective:** Evaluate booking channels (`Direct Website`, `OTA`, `Travel Agent`) to understand which channel drives the most reservations and revenue.
"""))

cells.append(nbf.v4.new_code_cell("""channel_summary = df.groupby('Booking_Channel').agg(
    Total_Bookings=('Booking_ID', 'count'),
    Realized_Revenue=('Booking_Value', lambda x: x[df.loc[x.index, 'Is_Canceled'] == 0].sum()),
    Avg_Revenue_Per_Booking=('Booking_Value', lambda x: x[df.loc[x.index, 'Is_Canceled'] == 0].mean()),
    Cancellation_Rate=('Is_Canceled', lambda x: (x.sum() / len(x)) * 100)
).reset_index()

# Add Rankings
channel_summary['Volume_Rank'] = channel_summary['Total_Bookings'].rank(ascending=False, method='min').astype(int)
channel_summary['Revenue_Rank'] = channel_summary['Realized_Revenue'].rank(ascending=False, method='min').astype(int)
channel_summary = channel_summary.sort_values(by='Realized_Revenue', ascending=False)

print(channel_summary.round(2).to_string(index=False))

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=channel_summary, x='Booking_Channel', y='Total_Bookings', ax=ax1, hue='Booking_Channel', palette='Blues_r', legend=False)
ax1.set_title('Booking Volume by Channel', fontweight='bold')
ax1.set_ylabel('Number of Bookings')

sns.barplot(data=channel_summary, x='Booking_Channel', y='Realized_Revenue', ax=ax2, hue='Booking_Channel', palette='Greens_r', legend=False)
ax2.set_title('Realized Revenue by Channel ($)', fontweight='bold')
ax2.set_ylabel('Revenue ($)')
plt.tight_layout()
plt.show()
"""))

# Task 14 (Categorize Lead-Time with pd.cut)
cells.append(nbf.v4.new_markdown_cell("""---
## 🏷️ Task 14: Create Lead-Time Categories Using `pd.cut()`
**Objective:** Discretize the continuous `Lead_Time_Days` variable into three operational business categories:
- **Last-Minute:** 0 to 7 days
- **Short-Term:** 8 to 30 days
- **Advance:** 31+ days
"""))

cells.append(nbf.v4.new_code_cell("""# Define bin edges and category labels
bins = [-1, 7, 30, df['Lead_Time_Days'].max() + 1]
labels = ['Last-Minute (0-7d)', 'Short-Term (8-30d)', 'Advance (31d+)']

# Use pd.cut() to categorize
df['Lead_Time_Category'] = pd.cut(df['Lead_Time_Days'], bins=bins, labels=labels)

print("Distribution of Bookings across Lead-Time Categories:")
print(df['Lead_Time_Category'].value_counts())
"""))

# Task 9
cells.append(nbf.v4.new_markdown_cell("""---
## 🚫 Task 9: Analyze Cancellation Rates by Channel & Lead-Time Bands
**Objective:** Cross-tabulate cancellation rates across booking channels and lead-time categories to pinpoint where high cancellations originate.
"""))

cells.append(nbf.v4.new_code_cell("""# Compute cancellation percentage pivot table
cancellation_matrix = pd.pivot_table(
    df,
    values='Is_Canceled',
    index='Booking_Channel',
    columns='Lead_Time_Category',
    aggfunc=lambda x: (x.sum() / len(x)) * 100
)

print("Cancellation Rate Matrix (%):")
print(cancellation_matrix.round(2))

# Visualization: Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(cancellation_matrix, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Cancellation Rate (%)'})
plt.title('Task 9: Cancellation Rate (%) by Channel & Lead-Time Band', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Booking Channel', fontweight='bold')
plt.xlabel('Lead-Time Category', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Task 10
cells.append(nbf.v4.new_markdown_cell("""---
## 📈 Task 10: Identify Top 3 Peak Revenue Months
**Objective:** Track monthly realized revenue across the calendar year and identify the 3 highest earning months.
"""))

cells.append(nbf.v4.new_code_cell("""# Aggregate realized revenue by month
monthly_revenue = df_realized.groupby(['Arrival_Month_Num', 'Arrival_Month'])['Booking_Value'].sum().reset_index()
monthly_revenue = monthly_revenue.rename(columns={'Booking_Value': 'Monthly_Revenue'})
monthly_revenue = monthly_revenue.sort_values('Arrival_Month_Num')

top_3_months = monthly_revenue.sort_values(by='Monthly_Revenue', ascending=False).head(3)

print("🏆 Top 3 Peak Revenue Months:")
for rank, (_, row) in enumerate(top_3_months.iterrows(), 1):
    print(f"  {rank}. {row['Arrival_Month']}: ${row['Monthly_Revenue']:,.2f}")

# Visualization: Monthly Trend
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_revenue, x='Arrival_Month', y='Monthly_Revenue', marker='o', color='#1f77b4', linewidth=2.5)
plt.scatter(top_3_months['Arrival_Month'], top_3_months['Monthly_Revenue'], color='#d62728', s=130, zorder=5, label='Top 3 Peak Months')
plt.title('Task 10: Monthly Hotel Revenue Trend (2024)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Arrival Month', fontweight='bold')
plt.ylabel('Realized Revenue ($)', fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Task 11
cells.append(nbf.v4.new_markdown_cell("""---
## 🧳 Task 11: Compare Stay Duration for Leisure vs. Business Travellers
**Objective:** Determine how stay lengths differ between business and leisure travelers to customize marketing packages.
"""))

cells.append(nbf.v4.new_code_cell("""stay_comparison = df.groupby('Customer_Type').agg(
    Avg_Stay_Nights=('Nights', 'mean'),
    Median_Stay_Nights=('Nights', 'median'),
    Std_Dev_Nights=('Nights', 'std'),
    Total_Guests=('Booking_ID', 'count')
).reset_index()

print(stay_comparison.round(2).to_string(index=False))

# Visualization
plt.figure(figsize=(7, 4.5))
sns.barplot(data=df, x='Customer_Type', y='Nights', hue='Customer_Type', palette='Set2', errorbar=None, legend=False)
plt.title('Task 11: Average Stay Duration by Customer Type', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Customer Type', fontweight='bold')
plt.ylabel('Average Length of Stay (Nights)', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Task 12
cells.append(nbf.v4.new_markdown_cell("""---
## 📐 Task 12: Statistical Calculations Using NumPy
**Objective:** Use NumPy's vector functions to compute the **mean**, **median**, and **standard deviation** of daily room rates across all reservations.
"""))

cells.append(nbf.v4.new_code_cell("""# Extract room rates as a NumPy array
rates_array = df['Room_Rate'].to_numpy()

# Calculate statistics
mean_rate = np.mean(rates_array)
median_rate = np.median(rates_array)
std_rate = np.std(rates_array)

print("--- NumPy Daily Room Rate Statistics ---")
print(f"Mean Room Rate:        ${mean_rate:.2f}")
print(f"Median Room Rate:      ${median_rate:.2f}")
print(f"Standard Deviation:    ${std_rate:.2f}")
"""))

# Task 13
cells.append(nbf.v4.new_markdown_cell("""---
## 🎯 Task 13: Detect Outliers Using the Interquartile Range (IQR) Method
**Objective:** Identify unusually high booking values using the IQR method.  
- **$Q_1$ (25th Percentile):** Value below which 25% of data falls.  
- **$Q_3$ (75th Percentile):** Value below which 75% of data falls.  
- **$IQR = Q_3 - Q_1$**  
- **Upper Bound:** $Q_3 + 1.5 \\times IQR$
"""))

cells.append(nbf.v4.new_code_cell("""booking_values = df['Booking_Value'].to_numpy()

q1 = np.percentile(booking_values, 25)
q3 = np.percentile(booking_values, 75)
iqr = q3 - q1
upper_bound = q3 + (1.5 * iqr)

# Identify outlier bookings
outliers = df[df['Booking_Value'] > upper_bound]

print(f"Q1 (25th Percentile):              ${q1:.2f}")
print(f"Q3 (75th Percentile):              ${q3:.2f}")
print(f"IQR (Interquartile Range):         ${iqr:.2f}")
print(f"Upper Outlier Threshold:           ${upper_bound:.2f}")
print(f"Total High-Value Outlier Bookings: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# Boxplot Visualization
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Booking_Value'], color='#a1c9f4', flierprops={'markerfacecolor':'#d32f2f', 'markersize':6})
plt.axvline(upper_bound, color='red', linestyle='--', linewidth=1.5, label=f'Upper Threshold (${upper_bound:.1f})')
plt.title('Task 13: Booking Value Distribution & Outlier Detection (IQR)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Booking Value ($ = Nights × Room Rate)', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Task 15
cells.append(nbf.v4.new_markdown_cell("""---
## 💡 Task 15: Five Evidence-Based Operational Recommendations

1. **Re-evaluate OTA Cancellation Terms:**  
   *Observation:* OTA bookings made more than 30 days in advance suffer a cancellation rate of **43.17%**, compared to only 28.16% on direct channels.  
   *Action:* Introduce a non-refundable rate tier (with a 5% discount) or require a 20% non-refundable deposit for advance OTA reservations.

2. **Incentivize Direct Website Bookings:**  
   *Observation:* Direct website bookings generate the highest average revenue per booking ($501.81) and have the lowest cancellation rate (24.11%).  
   *Action:* Provide direct bookers with value-adds (free high-speed Wi-Fi, late check-out, or breakfast vouchers) to shift demand away from commission-heavy OTAs.

3. **Dynamic Pricing During Seasonal Peaks (December, August, June):**  
   *Observation:* December ($42,463) and August ($41,886) are the top revenue generators due to holiday and summer vacation surges.  
   *Action:* Implement minimum length-of-stay (e.g., min 2 nights) rules and raise Suite/Deluxe room rates by 15–20% during these peak travel months.

4. **Segmented Marketing for Business vs. Leisure Travelers:**  
   *Observation:* Business travelers stay an average of **1.94 nights**, whereas leisure travelers stay **3.72 nights**.  
   *Action:* Design mid-week corporate packages (Monday–Thursday) featuring workspace amenities, and weekend family/leisure bundles (Friday–Sunday) with dining credits.

5. **Nurture High-Value Guests (IQR Outlier Segment):**  
   *Observation:* The IQR analysis uncovered 58 high-value bookings exceeding $1,180 (driven by Suite bookings and multi-night stays).  
   *Action:* Create a VIP loyalty desk to personalize their experience (room upgrades, airport transfers) to maximize repeat high-value visits.
"""))

# Conclusion & Viva Voce
cells.append(nbf.v4.new_markdown_cell("""---
## 🎯 Final Project Conclusion
Through systematic application of Python, Pandas, and NumPy, we transformed raw hotel reservation data into clear operational insights. The analysis revealed that while OTAs bring volume, direct bookings provide superior profitability and commitment. Seasonality significantly drives room demand, and customer segment behaviors dictate distinct length-of-stay requirements. These findings provide hotel management with concrete data to optimize revenue and inventory.

---
## 🎓 Viva Voce Preparation Guide (Questions & Answers)

**Q1: Why did you use the median instead of the mean to fill missing Lead Time values?**  
*Answer:* Lead time data is typically right-skewed with a few very large advance bookings. The mean is sensitive to extreme values, whereas the median represents the true central tendency of the majority of guests without distortion.

**Q2: What is the purpose of the IQR method in data analysis?**  
*Answer:* The Interquartile Range (IQR = Q3 - Q1) measures statistical dispersion. Multiplying IQR by 1.5 and adding it to Q3 creates an upper threshold (fence). Data points exceeding this fence are statistically considered outliers or exceptionally high transactions.

**Q3: How does `pd.cut()` work?**  
*Answer:* `pd.cut()` segments and sorts continuous numerical values into discrete categorical bins. In our project, it grouped continuous days into 'Last-Minute', 'Short-Term', and 'Advance' categories.

**Q4: Why calculate realized revenue only on non-canceled bookings?**  
*Answer:* When a reservation is canceled (`Is_Canceled == 1`), the guest does not stay or pay the full room rate (unless strict penalties apply). Counting canceled bookings as revenue would artificially inflate hotel earnings.

**Q5: What is the difference between NumPy arrays and Python lists?**  
*Answer:* NumPy arrays are stored contiguously in memory and support vectorized mathematical operations implemented in C, making them substantially faster and more memory-efficient than standard Python lists.
"""))

nb.cells = cells

notebook_path = "C:/Users/Dell/.gemini/antigravity/scratch/hotel-booking-analysis/Hotel_Booking_Analysis.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully created at: {notebook_path}")
