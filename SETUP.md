# How to Run PharmaStock Optimizer

Step-by-step guide to run this project on **any PC** (Windows, macOS, or Linux).

---

## Prerequisites

1. **Python 3.10 or higher**
   - Download from: https://www.python.org/downloads/
   - During installation, check **"Add Python to PATH"**
   - Verify: open terminal and type `python --version`

2. **Git** (optional, for cloning)
   - Download from: https://git-scm.com/downloads

---

## Step 1: Get the Project

**Option A — Clone with Git:**
```bash
git clone https://github.com/your-username/pharma_stock.git
cd pharma_stock
```

**Option B — Download ZIP:**
- Download the project ZIP file
- Extract it to any folder
- Open terminal and navigate to the extracted folder:
```bash
cd path/to/pharma_stock
```

---

## Step 2: Create Virtual Environment (Recommended)

This keeps dependencies isolated from your system Python.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal line.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- Streamlit (UI framework)
- FastAPI + Uvicorn (REST API)
- SQLAlchemy (Database ORM)
- XGBoost (Machine Learning)
- Plotly (Charts)
- bcrypt (Security)
- And more...

---

## Step 4: Configure Environment

Create a `.env` file in the project root folder:

**Windows (PowerShell):**
```powershell
New-Item .env -ItemType File
```

**macOS / Linux:**
```bash
touch .env
```

Then open `.env` in any text editor and add:
```env
SMTP_EMAIL=your_gmail@gmail.com
SMTP_PASSWORD=your_gmail_app_password
```

> **Note:** The email settings are optional. The app works without them — you just won't be able to use the "Retrieve Username" feature. To get a Gmail App Password, follow: https://support.google.com/accounts/answer/185833

---

## Step 5: Run the Application

You have **3 ways** to run this project:

### Option A: Streamlit UI Only (Simplest)
```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

### Option B: Streamlit UI + REST API
Open **two separate terminals**:

**Terminal 1 — API Server:**
```bash
uvicorn api.main:app --port 8000
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run app.py
```

- Streamlit UI: **http://localhost:8501**
- REST API: **http://localhost:8000**
- API Documentation: **http://localhost:8000/docs**

### Option C: Docker (Production)
Requires Docker Desktop installed: https://www.docker.com/products/docker-desktop/
```bash
docker-compose up --build
```
This starts PostgreSQL + API + UI automatically.

---

## Step 6: Create Your Account

1. Open the app in your browser (http://localhost:8501)
2. Select **"Register"** from the dropdown
3. Enter a username, email, and password
4. Switch to **"Login"** and sign in
5. You'll see the dashboard with inventory data and charts

---

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Port already in use" error
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F

# macOS / Linux
lsof -i :8501
kill -9 <PID>
```

### "streamlit: command not found"
Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Database errors after first run
The database file (`pharmastock_optimizer.db`) is included with the project. If you need to reset it, delete the file and restart — it will be recreated automatically.

### XGBoost installation fails
On some systems, XGBoost needs extra build tools:
```bash
# Windows — install Visual C++ Build Tools
# macOS
brew install libomp
# Linux
sudo apt-get install libgomp1
```

---

## Project URLs (When Running)

| Service | URL | Description |
|---------|-----|-------------|
| Streamlit UI | http://localhost:8501 | Main web interface |
| FastAPI REST API | http://localhost:8000 | API endpoints |
| Swagger Docs | http://localhost:8000/docs | Interactive API docs |
| ReDoc | http://localhost:8000/redoc | Alternative API docs |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: `24 passed`

---

## File Structure Quick Reference

```
pharma_stock/
├── app.py              ← Run this for Streamlit UI
├── api/main.py         ← Run this for REST API
├── config.py           ← All settings (edit .env to configure)
├── requirements.txt    ← Dependencies list
├── .env                ← Your secrets (create this yourself)
└── pharmastock_optimizer.db  ← Database (included)
```

---

## Need Help?

If something doesn't work:
1. Make sure Python 3.10+ is installed
2. Make sure you activated the virtual environment
3. Make sure all dependencies are installed
4. Check the terminal for error messages
