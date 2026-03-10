# 💊 PharmaSight
### An Intelligent Pharmacy Inventory Risk Analytics System for Pharmaceutical Supply Chain Optimization 

> **Business Problem:** Pharmacies lose thousands every month — either by overstocking medicines that expire on the shelf, or by running out of critical drugs and losing patient trust. Manual tracking can't keep up. This system solves that with data-driven forecasting and real-time decision support.

---

## 🎯 What This Project Does

PharmaStock Optimizer is a full-stack data analytics application built with **Python, SQLite, XGBoost, and Streamlit**. It gives pharmacy managers a single platform to:

- **Predict stockouts** before they happen using machine learning
- **Monitor 33,000+ sales records** across 30+ medicines from 2022–2024
- **Track expiry dates** and flag medicines approaching end-of-life
- **Manage suppliers** and understand which suppliers cover which medicines
- **Place orders** directly through the app and auto-update inventory

This is not just a dashboard — it's an operational tool that replaces spreadsheets and gut-feeling decisions with intelligent automation.

---

## 📊 Dataset Overview

| Feature | Detail |
|---|---|
| **Records** | 33,000+ daily sales transactions |
| **Time Period** | January 2022 – December 2024 (3 years) |
| **Medicines Tracked** | 30+ medicines across 10 therapeutic categories |
| **Suppliers** | 10 suppliers (SUP001–SUP010) |
| **Key Fields** | Date, Medicine Name, ATC Code, Stock Available, Stock Sold, Stockout Flag, External Factor, Price per Unit, Reorder Level, Days to Stockout |

**Therapeutic categories covered:** Analgesics, Antibiotics, Antidiabetics, Statins, Antihistamines, Bronchodilators, Benzodiazepines, Corticosteroids, Contraceptives, Immunosuppressants

---

## 🧠 Key Features & Technical Stack

### 🔮 Demand Forecasting with XGBoost
The system trains an **XGBoost regression model** per medicine using historical daily sales. It predicts:
- How many units will sell on any given future day
- Exactly how many days remain before current stock runs out (`Days_to_Stockout`)

This powers real-time reorder alerts — no manual calculation needed.

### 📦 Inventory Management
- View live inventory for all medicines
- Update stock, price, reorder level, and expiry date
- Automatically recalculates `Days_to_Stockout` on every update using the trained ML model
- Sidebar KPIs: Total Medicines, Low Stock Count, Expired Medicines

### 📈 Sales Analytics
- Filter sales by **date, month, or year**
- Interactive **Bar Chart** and **Line Chart** (toggle between views)
- Filter by individual medicine to drill into sales trends
- Custom date range selector for time-series analysis

### 🏭 Supplier Intelligence
- Maps each supplier to the medicines they provide
- Stock level bar chart per supplier
- Quickly identify which supplier to contact when a medicine is running low

### 🛒 Orders & Payments
- Add medicines to a shopping cart
- View total order cost before confirming
- Place orders that instantly update inventory in the database

### 🔐 User Authentication
- Secure login / registration with **SHA-256 password hashing**
- Email-based username recovery
- Session state management via Streamlit

---

## 🗄️ Database Schema (SQLite)

```
pharmastock_optimizer.db
│
├── Inventory       → Medicine_Name, Stock_Available, Price_per_Unit,
│                     Reorder_Level, Days_to_Stockout, Expiry_Date
│
├── Sales           → Date, Month, Year, Medicine_Name, ATC_Code,
│                     Stock_Sold, Stockout_Flag, External_Factor,
│                     Price_per_Unit, Supplier_ID, Reorder_Level, Days_to_Stockout
│
├── Medicine        → Medicine_Name, ATC_Code, Supplier_ID, Description,
│                     Uses, Cautions, Dosage, Alternatives
│
└── Users           → username, email, password (SHA-256 hashed)
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Suresh-Note/PharmaStock_Optimizer.git
cd PharmaStock_Optimizer
```

### 2. Install Dependencies
```bash
pip install streamlit pandas plotly xgboost numpy matplotlib seaborn
```

### 3. Run the App
```bash
streamlit run PharmaStock.py
```

### 4. Register & Login
Open `http://localhost:8501` in your browser, register a new account, and log in to access the full dashboard.

---

## 📁 Project Structure

```
PharmaStock_Optimizer/
│
├── PharmaStock.py                  # Main Streamlit application (1,043 lines)
├── pharmastock_optimizer.db        # SQLite database (inventory + sales + users)
├── expanded_sales_data.csv         # Raw sales data (33,000+ records, 2022–2024)
├── generate_outputs.py             # Python EDA script → generates analysis charts
│
├── app_screenshots/                # Live app screenshots (12 screens)
└── analysis_outputs/               # Python EDA chart outputs (6 charts)
```

---

## 💡 Business Insights the System Surfaces

| Insight | Business Impact |
|---|---|
| Medicines with `Days_to_Stockout < 7` | Trigger emergency reorder before stockout |
| Medicines with `Stock_Available < Reorder_Level` | Systematic low-stock alert |
| Expired medicines count | Reduce financial loss from wastage |
| Supplier-to-medicine mapping | Speed up procurement decisions |
| Sales trend by medicine over time | Identify seasonal demand spikes |
| External factor column (High/Medium/Low) | Correlate demand with external events |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application logic |
| **Streamlit** | Interactive web dashboard |
| **SQLite** | Local relational database |
| **Pandas** | Data manipulation & analysis |
| **XGBoost** | ML-based demand forecasting |
| **Plotly** | Interactive charts & visualizations |
| **Matplotlib / Seaborn** | EDA & analysis outputs |
| **Hashlib** | Secure password hashing (SHA-256) |
| **smtplib** | Email notifications |

---

## 📌 What Makes This Project Stand Out

✅ **End-to-end system** — from raw data → database → ML model → interactive UI  
✅ **Real ML model** — XGBoost trained on actual historical sales, not hardcoded thresholds  
✅ **Production-ready patterns** — authentication, session management, database transactions  
✅ **3 years of data** — 33,000+ records across 30 medicines, giving the model real signal  
✅ **Domain knowledge** — ATC codes, dosage info, drug alternatives built into the medicine lookup  

---

## 🔭 Future Improvements

- [ ] Add Power BI / Tableau dashboard integration for executive reporting
- [ ] Implement seasonal decomposition for better demand forecasting
- [ ] Add automated email alerts when stock drops below reorder level
- [ ] Deploy on Streamlit Cloud for public demo access
- [ ] Add CSV export for inventory and sales reports

---

## 🖥️ Live App Screenshots

### 🔐 Login — Secure Authentication
<img width="612" height="544" alt="Screenshot 2026-03-09 194022" src="https://github.com/user-attachments/assets/a85b6873-8fa8-48b6-8dae-e2d012b3b3ba" />

> SHA-256 hashed authentication with register, login, and username recovery via email.

---

### 📊 Dashboard — Inventory Overview
<img width="1416" height="581" alt="Screenshot 2026-03-09 193639" src="https://github.com/user-attachments/assets/d54e0ab0-5d0a-4f8f-b0ec-2c0efbcae5dc" />


> Live inventory table: 33 medicines with Stock Available, Price, Reorder Level, Days to Stockout, and Expiry Date. Sidebar KPIs (Total Medicines: 33 | Low Stock: 4 | Expired: 3) update in real time.

### 📈 Dashboard — Stock Level Chart
<img width="1895" height="897" alt="Screenshot 2026-03-09 193604" src="https://github.com/user-attachments/assets/c7b768fe-3840-4dbf-8db1-db4db0e15817" />

> Red bars = below reorder level (critical). Blue gradient = healthy stock. Powered by live SQLite data — updates every time inventory changes.

---

### 📦 Inventory Management
![Inventory](app_screenshots/app_04_inventory.png)
<img width="1906" height="870" alt="Screenshot 2026-03-09 193741" src="https://github.com/user-attachments/assets/b1177c85-983d-47d6-9ce9-7398a668da6b" />

> Full inventory table with the ability to update stock, price, reorder level, and expiry date. Every update auto-recalculates `Days_to_Stockout` using the XGBoost model.

---

### 🛒 Orders — Add to Cart
![Orders Add](app_screenshots/app_05_orders_add.png)
<img width="1882" height="887" alt="Screenshot 2026-03-09 194613" src="https://github.com/user-attachments/assets/3d988608-e898-4175-b243-589ee70a9eaa" />

> Select medicine and quantity. Live stock validation prevents over-ordering. Real-time cart with total cost calculation.

### ✅ Orders — Confirmed & Inventory Updated
![Orders Placed](app_screenshots/app_06_orders_placed.png)
<img width="1882" height="887" alt="Screenshot 2026-03-09 194613" src="https://github.com/user-attachments/assets/97863616-2945-4368-a13b-070ff02b2591" />

> On order confirmation, inventory is instantly updated in SQLite. No manual step required.

---

### 📋 Sales — Data Filtering
![Sales Table](app_screenshots/app_07_sales_table.png)
<img width="1900" height="915" alt="Screenshot 2026-03-09 194658" src="https://github.com/user-attachments/assets/c4bc8431-d2d0-47b6-8b6c-0baec9143638" />

> 33,000+ sales records filterable by date, month, or year. All columns visible: Date, Medicine, ATC Code, Stock Sold, Stockout Flag, External Factor, Supplier.

### 📊 Sales — Interactive Bar Chart
![Sales Bar](app_screenshots/app_08_sales_bar.png)
<img width="1875" height="838" alt="Screenshot 2026-03-09 194830" src="https://github.com/user-attachments/assets/0f5e9ee0-d3c3-4540-8697-cd601e075d8e" />

> Color-coded bar chart per medicine with hover tooltips. Filterable by medicine and custom date range (shown: Jan–Mar 2024).

### 📉 Sales — Trend Line Chart
![Sales Line](app_screenshots/app_09_sales_line.png)
<img width="1883" height="886" alt="Screenshot 2026-03-09 194857" src="https://github.com/user-attachments/assets/cc2678a4-6057-4a27-8375-3ae81c0489ae" />


> Multi-medicine line chart for spotting trends. Toggle between bar and line view dynamically.

---

### 💊 Medicine Information — Clinical Reference
![Medicine Info](app_screenshots/app_10_medicine_info.png)
<img width="1750" height="859" alt="Screenshot 2026-03-09 194925" src="https://github.com/user-attachments/assets/4170a648-0377-49b6-8abb-e763b362baea" />

> Built-in reference for all 33 medicines: ATC Code, Description, Uses, Cautions, Dosage, Alternatives, and live stock availability in one view.

---

### 🏭 Supplier Intelligence — Stock by Supplier
![Supplier Table](app_screenshots/app_11_suppliers_table.png)
<img width="1889" height="719" alt="Screenshot 2026-03-09 195043" src="https://github.com/user-attachments/assets/6edf4768-ed8e-468a-9fc2-b6a31e74d774" />

> Select any supplier to view their medicines and current stock. SUP005 shown: Dexibuprofen (5 units) and Aspirin (2 units) are critically low.

![Supplier Chart](app_screenshots/app_12_suppliers_chart.png)
<img width="1470" height="589" alt="Screenshot 2026-03-09 195053" src="https://github.com/user-attachments/assets/c02996ac-795f-4f21-8efe-10bf2b795659" />

> Visual stock level chart per supplier — instantly shows which medicines need emergency reorder.

---

## 📊 Data Analysis Outputs

*Generated from 33,000+ records using Python (Pandas, Matplotlib, Seaborn).*

### 01 — Monthly Revenue Trend (2022–2024)
![Monthly Revenue](analysis_outputs/01_monthly_revenue_trend.png)
<img width="1420" height="559" alt="Screenshot 2026-03-09 231549" src="https://github.com/user-attachments/assets/00cefaa9-84e9-49d1-a376-42c6abf13c1a" />

> 36-month revenue trend with peak identification. Consistent demand with seasonal spikes — useful for budget planning and procurement cycles.

### 02 — Top 10 Medicines by Units Sold
![Top 10](analysis_outputs/02_top10_medicines_sold.png)
<img width="1426" height="633" alt="Screenshot 2026-03-10 002111" src="https://github.com/user-attachments/assets/499f742c-f7a2-4991-b04d-a3be6b99ac28" />

> Ranked best-sellers over 3 years. High-volume medicines need larger safety stock buffers.

### 03 — Stock Alert Dashboard (All Medicines)
![Stock Alert](analysis_outputs/03_stock_alert_dashboard.png)
<img width="880" height="647" alt="Screenshot 2026-03-10 002156" src="https://github.com/user-attachments/assets/62398c9e-361c-4e89-8ce8-ac1f5c936ef5" />

> Red = below reorder threshold. Green = healthy. Orange line = reorder level per medicine. Complete stock health at a glance.

### 04 — Stockout Frequency Heatmap (Medicine × Year)
![Stockout Heatmap](analysis_outputs/04_stockout_heatmap.png)
<img width="907" height="937" alt="Screenshot 2026-03-10 002554" src="https://github.com/user-attachments/assets/ea4a6bc8-c93b-45ef-a25f-087992367f6b" />

> Chronic stockout offenders by medicine and year — identifies which medicines need higher safety stock or supplier renegotiation.

### 05 — External Demand Factor Impact
![External Factor](analysis_outputs/05_external_factor_impact.png)
<img width="1419" height="630" alt="Screenshot 2026-03-10 002241" src="https://github.com/user-attachments/assets/a662a3b1-0403-4fed-a68c-404fc51e931e" />

> High external demand → significantly more daily units sold. External factors are real demand drivers and must be factored into reorder planning.

### 06 — Seasonal Demand by Therapeutic Category
![Seasonal](analysis_outputs/06_seasonal_demand_by_category.png)
<img width="1212" height="934" alt="Screenshot 2026-03-10 002452" src="https://github.com/user-attachments/assets/ce807c27-9b70-4527-b8b3-c1e20f036832" />

> Monthly demand per drug category. Antibiotics and Antihistamines spike at different seasons — enabling proactive stocking before demand peaks.

---

## 👤 Author

**Suresh** — Aspiring Data Analyst  
🔗 [GitHub Profile](https://github.com/Suresh-Note)

> *"Data is the medicine for bad decisions. This project is proof of that."*

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
