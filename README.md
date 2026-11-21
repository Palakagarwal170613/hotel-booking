# 🏨 Hotel Booking & Occupancy Analysis Project

**Level:** B.Tech 1st Year Python & Data Analytics Project  
**Author / Student:** B.Tech Student  
**Workspace:** `C:\Users\Dell\.gemini\antigravity\scratch\hotel-booking-analysis\`  

---

## 📁 Project Directory Structure
```text
hotel-booking-analysis/
│
├── 13_hotel_bookings.csv           # Cleaned source dataset (1,200+ records)
├── hotel_analysis.py               # Standalone Python script with all 15 tasks
├── Hotel_Booking_Analysis.ipynb    # Professional Jupyter Notebook (for presentations)
├── PROJECT_REPORT.md               # Complete academic report with tables, graphs & Viva Q&A
├── generate_dataset.py             # Script that created the authentic dataset
├── README.md                       # Quick start guide
└── plots/                          # Generated high-resolution visualization charts
    ├── task7_room_type_analysis.png
    ├── task8_channel_ranking.png
    ├── task9_cancellation_heatmap.png
    ├── task10_monthly_revenue_trend.png
    ├── task11_stay_duration.png
    └── task13_outliers_boxplot.png
```

---

## 🚀 How to Run the Project

### 1. Run the Python Script in Terminal:
Open PowerShell or your command prompt in this folder and execute:
```bash
python hotel_analysis.py
```
This runs all 15 tasks sequentially, prints the exact statistical metrics and tables, and saves all 6 charts in the `plots/` folder.

### 2. Open the Jupyter Notebook:
You can open `Hotel_Booking_Analysis.ipynb` directly in:
- **VS Code / Antigravity IDE**: Simply double-click `Hotel_Booking_Analysis.ipynb` in the file explorer.
- **Jupyter Lab / Notebook**:
  ```bash
  jupyter notebook Hotel_Booking_Analysis.ipynb
  ```
- **Google Colab**: Upload `Hotel_Booking_Analysis.ipynb` and `13_hotel_bookings.csv` to Google Colab and run all cells!

---

## 📊 Summary of Completed Tasks:
- ✅ **Task 1:** Loaded dataset, checked shape (1,206 rows, 9 cols), data types, and head.
- ✅ **Task 2:** Identified missing values (25 in Lead Time, 20 in Room Rate) & 6 duplicate Booking_IDs.
- ✅ **Task 3:** Cleaned data (dropped duplicate IDs, imputed Lead Time with median, imputed Room Rate with Room_Type median).
- ✅ **Task 4:** Converted `Checkin_Date` to datetime, extracted `Arrival_Month` and `Arrival_Weekday`.
- ✅ **Task 5:** Total bookings: 1,200 | Total nights: 3,748 | Cancellation rate: 33.42%.
- ✅ **Task 6:** Total realized revenue: **$391,623.54** (Lost to cancellations: $207,818.11).
- ✅ **Task 7:** Compared rates & occupancy for Standard ($116.92), Deluxe ($177.49), Suite ($285.73).
- ✅ **Task 8:** Ranked booking channels (OTA #1 Volume, Direct Website #2 Volume but highest revenue per booking).
- ✅ **Task 14:** Binned lead time into Last-Minute (0-7d), Short-Term (8-30d), and Advance (31d+) via `pd.cut()`.
- ✅ **Task 9:** Analyzed cancellation matrix (OTA Advance has 43.17% cancellations vs 0% for Direct Last-Minute).
- ✅ **Task 10:** Identified top 3 peak revenue months: December ($42.4k), August ($41.8k), June ($36.9k).
- ✅ **Task 11:** Leisure guests stay 3.72 nights vs Business guests staying 1.94 nights.
- ✅ **Task 12:** NumPy statistical analysis: Mean=$160.74, Median=$140.14, Std Dev=$71.64.
- ✅ **Task 13:** IQR outlier detection ($Q_1$=$269.91, $Q_3$=$634.05, Upper Fence=$1,180.26) — found 58 high-value bookings.
- ✅ **Task 15:** 5 evidence-based operational recommendations for hotel management.
