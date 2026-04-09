<div align="center">

# 💊 PharmaStock Optimizer

**AI-Powered Pharmaceutical Inventory Management & Sales Analytics Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-F37626?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-24%20Passed-4FC08D?style=for-the-badge&logo=pytest&logoColor=white)](#testing)

---

*An industrial-grade inventory management system combining real-time stock tracking, interactive sales analytics, and XGBoost-based stockout prediction to optimize pharmaceutical supply chains.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Database Schema](#-database-schema)
- [Machine Learning](#-machine-learning)
- [Security](#-security)
- [Testing](#-testing)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)

---

## 🔍 Overview

PharmaStock Optimizer addresses a critical challenge in pharmaceutical retail — **predicting when medicines will go out of stock** before it happens. Traditional inventory systems rely on static reorder points, but demand for medicines varies by season, external factors, and trends.

This platform integrates **XGBoost regression models** trained on 33,000+ historical sales records to dynamically predict stockout timelines, enabling pharmacies to:

- 🎯 **Proactively reorder** before critical shortages
- 📉 **Reduce wastage** from overstocking perishable medicines
- 📊 **Analyze sales trends** with interactive, filterable visualizations
- 🏢 **Manage supplier relationships** with per-supplier medicine mappings

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐│
│  │Dashboard │ │Inventory │ │  Sales   │ │Supplier│ │Meds  ││
│  │  Page    │ │  Page    │ │  Page    │ │  Page  │ │ Page ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ └──┬───┘│
├───────┼─────────────┼────────────┼───────────┼─────────┼────┤
│       │    Service Layer (services/)         │         │     │
│  ┌────▼──────┐ ┌────▼────┐ ┌─────▼────┐ ┌───▼────┐   │     │
│  │ Inventory │ │  Sales  │ │  Orders  │ │Supplier│   │     │
│  │  Service  │ │ Service │ │ Service  │ │Service │   │     │
│  └────┬──────┘ └────┬────┘ └─────┬────┘ └───┬────┘   │     │
├───────┼─────────────┼────────────┼───────────┼────────┼─────┤
│       │    Data Layer                        │        │     │
│  ┌────▼──────────────▼────────────▼──────────▼────┐ ┌─▼───┐│
│  │         SQLAlchemy ORM (database/)             │ │JSON ││
│  │  Models: User │ Medicine │ Inventory │ Sale    │ │Data ││
│  └──────────────────┬────────────────────────────── └─────┘│
│                     │                                      │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │              Cross-Cutting Concerns                   │  │
│  │  Auth (bcrypt+RBAC) │ ML (XGBoost) │ Logging │ Config│  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Design Patterns

| Pattern | Implementation |
|---------|---------------|
| **Layered Architecture** | UI → Service → ORM → Database |
| **Repository Pattern** | Service classes abstract all data access |
| **Dependency Injection** | SQLAlchemy sessions injected into services |
| **Configuration as Code** | Pydantic `BaseSettings` with `.env` override |
| **Context Manager** | `get_session()` handles commit/rollback/close |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI |
| **Visualization** | Plotly Express | Interactive charts |
| **ORM** | SQLAlchemy 2.0 | Database abstraction |
| **Database** | SQLite (WAL mode) | Relational storage |
| **Validation** | Pydantic v2 | Input/config validation |
| **Machine Learning** | XGBoost | Stockout forecasting |
| **Authentication** | bcrypt | Salted password hashing |
| **Configuration** | pydantic-settings | Environment management |
| **Testing** | pytest | Unit test suite |
| **Containerization** | Docker | Production deployment |

---

## ✨ Key Features

### 📊 Dashboard
Real-time inventory overview with color-coded stock charts and sidebar metrics

### 📦 Inventory Management
CRUD operations with AI-powered stockout prediction on every update

### 🛒 Orders & Payments
Shopping cart with cumulative stock validation and automatic inventory deduction

### 📈 Sales Analytics
Filter by date/month/year with interactive bar and line charts

### 🏭 Supplier Management
View medicines grouped by supplier with stock visualizations

### 💊 Medicine Reference
33 medicines with ATC codes, dosage, cautions, and real-time stock display

### 🔐 Authentication & RBAC
bcrypt hashing, role-based access (admin/user), session timeout, email recovery

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** 
- **pip** package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/pharma_stock.git
cd pharma_stock

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your SMTP credentials
```

### Configure `.env`

```env
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
```

### Run

```bash
# Run the modular app
streamlit run app.py

# Or run the legacy single-file version
streamlit run PharmaStock.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
pharma_stock/
├── app.py                          # Entry point (thin router)
├── config.py                       # Centralized Pydantic settings
├── PharmaStock.py                  # Legacy single-file version
│
├── database/                       # Data access layer
│   ├── connection.py               # SQLAlchemy engine + session factory
│   └── models.py                   # ORM models (User, Medicine, Inventory, Sale)
│
├── auth/                           # Authentication module
│   ├── security.py                 # bcrypt hashing + session management
│   ├── email.py                    # SMTP email service
│   └── service.py                  # Auth orchestration (register/login/delete)
│
├── services/                       # Business logic layer
│   ├── inventory.py                # Inventory CRUD + summary
│   ├── sales.py                    # Sales filtering + queries
│   ├── orders.py                   # Cart management + order processing
│   └── suppliers.py                # Supplier-medicine mappings
│
├── ml/                             # Machine learning pipeline
│   └── forecasting.py              # XGBoost training + stockout prediction
│
├── pages_ui/                       # Streamlit page modules
│   ├── dashboard.py                # Dashboard view
│   ├── inventory_page.py           # Inventory management
│   ├── sales_page.py               # Sales analytics
│   ├── medicines_page.py           # Medicine reference
│   └── suppliers_page.py           # Supplier view
│
├── utils/                          # Cross-cutting utilities
│   ├── logger.py                   # Structured logging
│   ├── exceptions.py               # Custom exception hierarchy
│   └── validators.py               # Pydantic input schemas
│
├── data/
│   ├── medicines_info.json         # Medicine reference data
│   └── expanded_sales_data.csv     # Historical sales dataset (33K rows)
│
├── tests/                          # pytest test suite (24 tests)
│   ├── conftest.py                 # Fixtures (in-memory DB, sample data)
│   ├── test_auth.py                # Auth service tests (9 tests)
│   ├── test_inventory.py           # Inventory CRUD tests (9 tests)
│   └── test_forecasting.py         # ML prediction tests (4 tests)
│
├── Dockerfile                      # Production container
├── docker-compose.yml              # Container orchestration
├── requirements.txt                # Python dependencies
├── .env                            # Environment secrets (not committed)
└── .gitignore                      # Git exclusions
```

---

## 🗄 Database Schema

### `Inventory` — Current stock levels (33 medicines)

| Column | Type | Description |
|--------|------|-------------|
| Medicine_Name | TEXT (PK) | Medicine identifier |
| ATC_Code | TEXT | ATC classification |
| Stock_Available | INTEGER | Current units in stock |
| Price_per_Unit | REAL | Current price |
| Reorder_Level | INTEGER | Low-stock threshold |
| Days_to_Stockout | INTEGER | ML-predicted days |
| Expiry_Date | DATE | Batch expiry |

### `Sales` — Historical records (~33,000 rows)

| Column | Type | Description |
|--------|------|-------------|
| Date | TEXT | Sale date (YYYY-Mon-DD) |
| Month | TEXT | Full month name |
| Year | INTEGER | Sale year |
| Medicine_Name | TEXT | Medicine identifier |
| Stock_Sold | INTEGER | Units sold |
| External_Factor | TEXT | Demand factor |

### `users` — Registered accounts (RBAC-enabled)

| Column | Type | Description |
|--------|------|-------------|
| username | TEXT (PK) | Unique identifier |
| email | TEXT | Recovery email |
| password | TEXT | bcrypt hash |
| role | TEXT | "admin" or "user" |
| created_at | DATETIME | Registration timestamp |

---

## 🤖 Machine Learning

### XGBoost Stockout Prediction

Per-medicine regression models predict daily sales volume:

```
Input:  Day index (days since first recorded sale)
Output: Predicted units sold per day
Model:  XGBRegressor (100 estimators, squared error)
```

**Process:**
1. Train one model per medicine from historical sales data
2. Starting from current stock, iterate day-by-day subtracting predicted sales
3. Day count when stock reaches zero = **Days to Stockout**

**Safeguards:**
- Models cached with `@st.cache_resource` (1-hour TTL)
- 365-day iteration cap prevents infinite loops
- Negative predictions clamped to zero

---

## 🔐 Security

| Feature | Implementation |
|---------|---------------|
| **Password Hashing** | bcrypt with configurable rounds (default: 12) |
| **RBAC** | Admin/user roles stored in database |
| **Session Timeout** | Configurable expiry (default: 30 minutes) |
| **Credential Storage** | `.env` file, excluded from Git |
| **SQL Injection** | SQLAlchemy parameterized queries |
| **Custom Exceptions** | Structured error hierarchy (never leaks internals) |

---

## 🧪 Testing

**24 tests** covering 3 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_auth.py` | 9 | Password hashing, register, login, delete, RBAC |
| `test_inventory.py` | 9 | CRUD, stock deduction, summary, properties |
| `test_forecasting.py` | 4 | Bounds, clamping, mock predictions |

```bash
python -m pytest tests/ -v
# ======================= 24 passed in 8.91s ========================
```

Tests use **in-memory SQLite** databases — no effect on production data.

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t pharmastock .
docker run -p 8501:8501 --env-file .env pharmastock
```

The container includes a health check endpoint at `/_stcore/health`.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run tests (`python -m pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add your feature'`)
5. Push and open a Pull Request

---

<div align="center">

**Built with ❤️ using Streamlit · SQLAlchemy · XGBoost · Pydantic · Docker**

</div>
