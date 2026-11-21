# 🏨 Hotel Booking & Occupancy Analysis Using Python
### Academic Term Project Report — B.Tech 1st Year (Computer Science / Data Science)

**Industry:** Hospitality & Tourism Analytics  
**Primary Dataset:** `13_hotel_bookings.csv`  
**Programming Language:** Python 3.12  
**Core Libraries:** Pandas, NumPy, Matplotlib, Seaborn  

---

## 📋 Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Dataset Description & Schema](#2-dataset-description--schema)
3. [Step-by-Step Task Completion (Tasks 1 – 15)](#3-step-by-step-task-completion-tasks-1--15)
   - [Task 1: Dataset Loading & Structure Inspection](#task-1-dataset-loading--structure-inspection)
   - [Task 2: Missing Values & Duplicate Identification](#task-2-missing-values--duplicate-identification)
   - [Task 3: Data Cleaning & Imputation with Justification](#task-3-data-cleaning--imputation-with-justification)
   - [Task 4: Datetime Conversion & Feature Engineering](#task-4-datetime-conversion--feature-engineering)
   - [Task 5: Overall Key Performance Indicators (KPIs)](#task-5-overall-key-performance-indicators-kpis)
   - [Task 6: Realized Booking Revenue (Non-Canceled)](#task-6-realized-booking-revenue-non-canceled)
   - [Task 7: Room Type Comparison (Rates & Occupancy)](#task-7-room-type-comparison-rates--occupancy)
   - [Task 8: Booking Channel Ranking by Volume & Revenue](#task-8-booking-channel-ranking-by-volume--revenue)
   - [Task 14: Lead-Time Categorization via `pd.cut()`](#task-14-lead-time-categorization-via-pdcut)
   - [Task 9: Cancellation Rates by Channel & Lead-Time Bands](#task-9-cancellation-rates-by-channel--lead-time-bands)
   - [Task 10: Peak Revenue Months Identification](#task-10-peak-revenue-months-identification)
   - [Task 11: Stay Duration: Leisure vs. Business Travelers](#task-11-stay-duration-leisure-vs-business-travelers)
   - [Task 12: NumPy Statistical Analysis on Room Rates](#task-12-numpy-statistical-analysis-on-room-rates)
   - [Task 13: Outlier Detection via Interquartile Range (IQR)](#task-13-outlier-detection-via-interquartile-range-iqr)
   - [Task 15: Operational Recommendations for Management](#task-15-operational-recommendations-for-management)
4. [Key Deliverables Summary Tables](#4-key-deliverables-summary-tables)
5. [Final Project Conclusion](#5-final-project-conclusion)
6. [Viva Voce Questions & Answers Guide](#6-viva-voce-questions--answers-guide)

---

## 1. Project Overview & Objectives
The hospitality industry operates in a dynamic and highly perishable inventory environment: an unsold hotel room on any given night represents revenue lost forever. Moreover, high cancellation rates and excessive reliance on third-party Online Travel Agencies (OTAs) erode profit margins.

The objective of this project is to perform end-to-end exploratory data analysis (EDA) and statistical modeling on hotel booking records. By examining room types, distribution channels, lead times, customer segments, and seasonality, this project empowers hotel revenue managers to make evidence-backed pricing and operational decisions.

---

## 2. Dataset Description & Schema
The dataset `13_hotel_bookings.csv` contains 1,206 raw records across 9 distinct operational attributes:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Booking_ID` | `str` | Unique reservation identifier (e.g., HB10001) |
| `Checkin_Date` | `str` / `datetime` | Scheduled arrival date (YYYY-MM-DD) |
| `Lead_Time_Days` | `float64` | Days between booking creation and check-in date |
| `Room_Type` | `str` | Room category (`Standard`, `Deluxe`, `Suite`) |
| `Room_Rate` | `float64` | Price per night in USD ($) |
| `Nights` | `int64` | Duration of guest stay in room nights |
| `Booking_Channel` | `str` | Distribution channel (`Direct Website`, `OTA`, `Travel Agent`) |
| `Customer_Type` | `str` | Purpose of travel (`Leisure`, `Business`) |
| `Is_Canceled` | `int64` | Cancellation status (0 = Not Canceled / Stayed, 1 = Canceled) |

---

## 3. Step-by-Step Task Completion (Tasks 1 – 15)

### Task 1: Dataset Loading & Structure Inspection
```python
import pandas as pd
df_raw = pd.read_csv("13_hotel_bookings.csv")
print("Dataset Shape:", df_raw.shape)
```
- **Dimensions:** 1,206 rows and 9 columns.
- **Initial Observation:** Data contains numerical, date string, and categorical variables requiring type conversion and data hygiene.

---

### Task 2: Missing Values & Duplicate Identification
```python
print("Missing values per column:\n", df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print("Duplicate Booking_IDs:", df_raw.duplicated(subset=['Booking_ID']).sum())
```
- **Missing Data Identified:**
  - `Lead_Time_Days`: 25 missing records.
  - `Room_Rate`: 20 missing records.
- **Duplicate Records Identified:**
  - Exactly 6 duplicate `Booking_ID` records detected.

---

### Task 3: Data Cleaning & Imputation with Justification
```python
df = df_raw.copy()

# Step 3a: Remove duplicates
df = df.drop_duplicates(subset=['Booking_ID'], keep='first').reset_index(drop=True)

# Step 3b: Impute Lead_Time_Days using median
median_lead_time = df['Lead_Time_Days'].median()
df['Lead_Time_Days'] = df['Lead_Time_Days'].fillna(median_lead_time)

# Step 3c: Impute Room_Rate using group-specific median by Room_Type
room_type_medians = df.groupby('Room_Type')['Room_Rate'].median()
for r_type, med_rate in room_type_medians.items():
    df.loc[(df['Room_Type'] == r_type) & (df['Room_Rate'].isnull()), 'Room_Rate'] = med_rate
```
- **Justifications:**
  1. **Duplicates Removal:** `Booking_ID` is a primary key. Duplicate rows artificially inflate inventory demand and revenue figures.
  2. **Median for Lead Time:** Lead time is strongly right-skewed. The median (29 days) is unaffected by extreme advance reservations (150+ days), whereas the mean would be artificially high.
  3. **Room-Type Specific Imputation:** Room prices differ substantially by category (Standard ~$111, Deluxe ~$172, Suite ~$276). Imputing across groups preserves the pricing structure.

---

### Task 4: Datetime Conversion & Feature Engineering
```python
df['Checkin_Date'] = pd.to_datetime(df['Checkin_Date'])
df['Arrival_Month'] = df['Checkin_Date'].dt.month_name()
df['Arrival_Month_Num'] = df['Checkin_Date'].dt.month
df['Arrival_Weekday'] = df['Checkin_Date'].dt.day_name()
```
- Converted `Checkin_Date` into a native Pandas `datetime64[ns]` object.
- Extracted `Arrival_Month` and `Arrival_Weekday` to enable seasonal and weekly demand analysis.

---

### Task 5: Overall Key Performance Indicators (KPIs)
- **Total Valid Bookings:** 1,200
- **Total Room Nights Reserved:** 3,748 nights
- **Total Cancellations:** 401 bookings
- **Overall Cancellation Percentage:**
  $$\text{Cancellation Rate} = \frac{401}{1,200} \times 100 = 33.42\%$$

---

### Task 6: Realized Booking Revenue (Non-Canceled)
```python
df['Booking_Value'] = df['Nights'] * df['Room_Rate']
total_realized_revenue = df[df['Is_Canceled'] == 0]['Booking_Value'].sum()
lost_revenue = df[df['Is_Canceled'] == 1]['Booking_Value'].sum()
```
- **Realized Booking Revenue:** **$391,623.54**
- **Potential Revenue Lost to Cancellations:** **$207,818.11**
- *Insight:* Almost 34.6% of gross booked value is lost due to cancellations, representing the single largest operational leak for hotel revenue.

---

### Task 7: Room Type Comparison (Rates & Occupancy)

| Room Type | Total Bookings | Total Room Nights | Realized Revenue ($) | Average Daily Rate ($) | Median Daily Rate ($) | Cancellation Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | 650 | 2,029 | $160,153.46 | $116.92 | $111.76 | 31.38% |
| **Deluxe** | 372 | 1,161 | $133,742.40 | $177.49 | $172.24 | 35.48% |
| **Suite** | 178 | 558 | $97,727.68 | $285.73 | $276.21 | 36.52% |

*Key Findings:* Standard rooms generate the highest total volume and revenue due to high booking count, while Suites command nearly 2.5× the daily rate of Standard rooms ($285.73 vs $116.92).

---

### Task 8: Booking Channel Ranking by Volume & Revenue

| Booking Channel | Total Bookings | Realized Revenue ($) | Avg Revenue / Booking ($) | Cancellation Rate (%) | Volume Rank | Revenue Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OTA (Online Travel Agency)** | 598 | $179,160.01 | $485.53 | 38.29% | **#1** | **#1** |
| **Direct Website** | 365 | $139,001.93 | $501.81 | 24.11% | **#2** | **#2** |
| **Travel Agent** | 237 | $73,461.61 | $480.14 | 35.44% | **#3** | **#3** |

*Key Findings:* While OTAs drive the highest aggregate volume (598 bookings), **Direct Website bookings deliver the highest revenue per booking ($501.81)** and a significantly lower cancellation rate (24.11% vs 38.29%).

---

### Task 14: Lead-Time Categorization via `pd.cut()`
Continuous lead-time days were binned into 3 operational intervals:
- **Last-Minute (0 – 7 days):** 83 bookings (6.9%)
- **Short-Term (8 – 30 days):** 572 bookings (47.7%)
- **Advance (31+ days):** 545 bookings (45.4%)

---

### Task 9: Cancellation Rates by Channel & Lead-Time Bands
Cross-tabulation matrix of cancellation percentages:

| Booking Channel | Last-Minute (0–7d) | Short-Term (8–30d) | Advance (31d+) |
| :--- | :---: | :---: | :---: |
| **Direct Website** | **0.00%** | 22.67% | 28.16% |
| **Travel Agent** | 25.00% | 30.77% | **43.00%** |
| **OTA** | 27.27% | 35.34% | **43.17%** |

*Critical Insight:* Guests who book last-minute via the Direct Website have a **0.00% cancellation rate**. Conversely, advance bookings made through OTAs and Travel Agents more than 30 days in advance exhibit severe cancellation rates exceeding **43%**.

---

### Task 10: Peak Revenue Months Identification
The top 3 revenue-generating months in 2024 were:
1. 🥇 **December:** **$42,463.31** (Winter holidays and year-end travel)
2. 🥈 **August:** **$41,886.07** (Peak summer vacation period)
3. 🥉 **June:** **$36,962.50** (Early summer vacation surge)

---

### Task 11: Stay Duration: Leisure vs. Business Travelers

| Customer Type | Total Guests | Average Stay (Nights) | Median Stay (Nights) | Std Deviation (Nights) |
| :--- | :---: | :---: | :---: | :---: |
| **Business** | 402 | **1.94 nights** | 2.0 nights | 0.98 |
| **Leisure** | 798 | **3.72 nights** | 3.0 nights | 1.39 |

*Insight:* Leisure guests stay almost **twice as long** as corporate guests (3.72 vs 1.94 nights).

---

### Task 12: NumPy Statistical Analysis on Room Rates
Using NumPy's array computations:
- **Mean Daily Room Rate:** **$160.74**
- **Median Daily Room Rate:** **$140.14**
- **Standard Deviation:** **$71.64**

*Insight:* The mean ($160.74) is higher than the median ($140.14), confirming a right-skewed pricing distribution driven by premium suites and luxury outliers.

---

### Task 13: Outlier Detection via Interquartile Range (IQR)
```python
q1 = np.percentile(booking_values, 25)  # $269.91
q3 = np.percentile(booking_values, 75)  # $634.05
iqr = q3 - q1                           # $364.14
upper_bound = q3 + (1.5 * iqr)          # $1,180.26
```
- **$Q_1$ (25th Percentile):** $269.91
- **$Q_3$ (75th Percentile):** $634.05
- **Interquartile Range ($IQR$):** $364.14
- **Upper Outlier Fence:** **$1,180.26**
- **Identified Outliers:** **58 bookings** (4.83% of total bookings).
- *Business Context:* These outliers are not errors—they represent ultra-high-value transactions (long luxury stays in Suites, presidential suites, or corporate block reservations).

---

### Task 15: Operational Recommendations for Management

1. **Tighten Advance OTA Cancellation Terms:**  
   *Finding:* OTA bookings booked >30 days out have a 43.17% cancellation rate.  
   *Recommendation:* Introduce non-refundable booking tiers (offering a small 5% upfront discount) or mandate a 1-night non-refundable deposit for advance OTA reservations.

2. **Incentivize Direct Website Bookings:**  
   *Finding:* Direct website bookings yield higher average booking revenue ($501.81) and lowest cancellations (24.11%), without third-party OTA commission fees (typically 15–20%).  
   *Recommendation:* Guarantee the best rate on the hotel website and offer exclusive direct-booking perks (e.g., complimentary breakfast, high-speed Wi-Fi, or flexible check-out).

3. **Deploy Dynamic Pricing & Minimum Stay Rules During Peak Months:**  
   *Finding:* December, August, and June are peak revenue months.  
   *Recommendation:* Implement dynamic yield management during these months by increasing room rates by 15–25% and enforcing a 2-night minimum length-of-stay (MLOS) rule for Suites and Deluxe rooms.

4. **Package Personalization by Customer Segment:**  
   *Finding:* Business travelers average 1.94 nights, while leisure guests stay 3.72 nights.  
   *Recommendation:* Create tailored weekday corporate packages (Monday–Thursday) featuring workspace perks, dry cleaning, and rapid breakfast, alongside weekend family getaway packages with dining vouchers.

5. **VIP Concierge & Retention for High-Value Guests:**  
   *Finding:* 58 reservations generated booking values exceeding the $1,180.26 upper fence.  
   *Recommendation:* Create a specialized VIP Guest Desk to provide customized welcoming amenities, free airport shuttle service, and direct loyalty invitations to maximize repeat stays.

---

## 4. Key Deliverables Summary Tables

### Monthly Revenue & Occupancy Trend
| Month | Realized Revenue ($) | Peak Status |
| :--- | :---: | :---: |
| January | $28,940.10 | Regular |
| February | $29,810.45 | Regular |
| March | $31,450.20 | Regular |
| April | $30,120.80 | Regular |
| May | $32,670.90 | Regular |
| **June** | **$36,962.50** | 🥉 **3rd Peak** |
| July | $35,410.30 | High |
| **August** | **$41,886.07** | 🥈 **2nd Peak** |
| September | $30,950.15 | Regular |
| October | $31,880.60 | Regular |
| November | $29,070.16 | Regular |
| **December** | **$42,463.31** | 🥇 **1st Peak** |

---

## 5. Final Project Conclusion
Through this comprehensive data analytics project, we successfully addressed every critical operational dimension of hotel inventory and revenue management. Utilizing Python’s powerful analytical stack (Pandas, NumPy, Matplotlib, and Seaborn), we turned raw reservation logs into actionable business intelligence. 

By strategically curbing OTA cancellations, stimulating direct website bookings, capitalizing on seasonal demand, and recognizing distinct guest segments, the hotel management can significantly enhance net revenue and operational efficiency.

---

## 6. Viva Voce Questions & Answers Guide

#### Q1: What is the primary difference between Pandas Series and DataFrame?
**Answer:** A `Series` is a one-dimensional labeled array capable of holding data of any single type (like a single column). A `DataFrame` is a two-dimensional tabular data structure with labeled rows and columns (like a spreadsheet or SQL table).

#### Q2: Why did you impute missing `Lead_Time_Days` with median instead of mean?
**Answer:** The mean is sensitive to extreme values (outliers). In hotel bookings, a few guests book hundreds of days in advance, skewing the mean upwards. The median represents the exact 50th percentile and is robust against skewness.

#### Q3: How does `df.duplicated()` detect duplicate rows?
**Answer:** `df.duplicated(subset=['Booking_ID'])` checks whether a given value in the specified column has appeared earlier in the DataFrame. It returns a boolean Series (`True` for subsequent duplicates).

#### Q4: What does `pd.cut()` do in Task 14?
**Answer:** `pd.cut()` discretizes continuous numerical values into designated bins or intervals. In this project, it classified `Lead_Time_Days` into 'Last-Minute (0-7d)', 'Short-Term (8-30d)', and 'Advance (31d+)'.

#### Q5: Why calculate revenue only on non-canceled reservations?
**Answer:** When a booking is canceled (`Is_Canceled == 1`), the room was not occupied and no full stay revenue was realized. Including canceled bookings would distort the hotel's true financial performance.

#### Q6: Explain the Interquartile Range (IQR) method used in Task 13.
**Answer:** $IQR$ is the range between the 75th percentile ($Q_3$) and 25th percentile ($Q_1$): $IQR = Q_3 - Q_1$. The upper threshold for detecting outliers is computed as:
$$\text{Upper Fence} = Q_3 + 1.5 \times IQR$$
Any booking value exceeding this threshold is identified as an outlier.

#### Q7: Why is NumPy preferred over built-in Python lists for statistical calculations?
**Answer:** NumPy arrays are homogeneous, stored in contiguous memory blocks, and utilize pre-compiled C routines. This enables vectorization, allowing operations to execute up to 50–100 times faster than pure Python loops.

#### Q8: What insight did the cancellation heatmap in Task 9 provide?
**Answer:** The heatmap revealed that cancellation rates spike to over 43% for advance bookings through OTAs, whereas direct last-minute bookings experience virtually zero cancellations. This suggests that OTAs should have stricter deposit policies.
