import streamlit as st
import sqlite3
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import calendar
import bcrypt
from dotenv import load_dotenv

import xgboost as xgb
import numpy as np

# --- Load environment variables ---
load_dotenv()

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(layout="wide", page_title="PharmaStock Optimizer")

# --- Initialize Session State ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# --- Database Connection Factory ---
# Standardized: all DB access goes through this single factory
@st.cache_resource
def get_db_connection():
    """Returns a shared database connection with row_factory enabled."""
    conn = sqlite3.connect('pharmastock_optimizer.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_raw_connection():
    """Returns a fresh connection without row_factory for pandas queries."""
    return sqlite3.connect('pharmastock_optimizer.db')

# --- Database Setup ---
conn = get_db_connection()
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    )
''')
conn.commit()


# --- Utility Functions ---
def hash_password(password):
    """Hashes a password using bcrypt with automatic salting."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    """Verifies a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, email, password):
    """Registers a new user with bcrypt-hashed password."""
    hashed_password = hash_password(password)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                  (username, email, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    """Authenticates a user using bcrypt verification."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result is None:
        return False
    return verify_password(password, result['password'])

def retrieve_username(email):
    """Retrieves a username based on email."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE email = ?', (email,))
    result = c.fetchone()
    return result['username'] if result else None

def delete_account(username, password):
    """Deletes a user account after verifying credentials."""
    if authenticate_user(username, password):
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('DELETE FROM users WHERE username = ?', (username,))
        conn.commit()
        return True
    return False

def send_email(to_email, subject, body):
    """Sends an email using SMTP credentials from environment variables."""
    from_email = os.getenv('SMTP_EMAIL')
    from_password = os.getenv('SMTP_PASSWORD')
    
    if not from_email or not from_password:
        st.error("Email credentials not configured. Please set SMTP_EMAIL and SMTP_PASSWORD in .env file.")
        return False

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the Gmail SMTP server and send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# --- Callback Functions ---
def login_callback():
    """Handles the login process."""
    username = st.session_state.login_username
    password = st.session_state.login_password
    if authenticate_user(username, password):
        st.session_state.authenticated = True
        st.session_state.username = username
    else:
        st.error("Invalid username or password.")

def register_callback():
    """Handles the registration process."""
    username = st.session_state.register_username
    email = st.session_state.register_email
    password = st.session_state.register_password
    if register_user(username, email, password):
        st.success("Registration successful! Please log in.")
    else:
        st.error("Username already exists. Please choose a different one.")

def retrieve_username_callback():
    """Handles username retrieval."""
    email = st.session_state.retrieve_email
    username = retrieve_username(email)
    if username:
        subject = "Your Username Retrieval"
        body = f"Dear user,\n\nYour username is: {username}\n\nBest regards,\nPharmaStock Optimizer Team"
        if send_email(email, subject, body):
            st.success("Your username has been sent to your email address.")
        else:
            st.error("Failed to send email. Please try again later.")
    else:
        st.error("Email not found.")

def delete_account_callback():
    """Handles account deletion."""
    username = st.session_state.delete_username
    password = st.session_state.delete_password
    if delete_account(username, password):
        st.success("Account deleted successfully.")
    else:
        st.error("Invalid username or password.")

def logout_callback():
    """Handles the logout process."""
    st.session_state.authenticated = False
    st.session_state.username = ''
    

def get_data():
    """Fetch all inventory data as a DataFrame."""
    conn_raw = get_raw_connection()
    df = pd.read_sql_query("SELECT * FROM Inventory", conn_raw)
    conn_raw.close()
    return df

# Initialize session state for the cart
if 'cart' not in st.session_state:
    st.session_state.cart = {}

# Function to fetch inventory data
def get_inventory_data():
    conn_raw = get_raw_connection()
    df = pd.read_sql_query("SELECT * FROM Inventory", conn_raw)
    conn_raw.close()
    return df

# Function to fetch sales data
def get_sales_data():
    conn_raw = get_raw_connection()
    df = pd.read_sql_query("SELECT * FROM Sales", conn_raw)
    conn_raw.close()
    return df

# Function to train XGBoost models for sales prediction (cached)
@st.cache_resource(ttl=3600)  # Cache for 1 hour to avoid retraining on every interaction
def train_xgboost_model():
    """Train XGBoost models for all medicines. Cached to avoid retraining on every update."""
    conn_raw = get_raw_connection()
    sales_data = pd.read_sql_query("SELECT * FROM Sales", conn_raw)
    conn_raw.close()
    
    sales_data["Date"] = pd.to_datetime(sales_data["Date"])
    grouped_sales = sales_data.groupby(["Medicine_Name", pd.Grouper(key="Date", freq="D")])["Stock_Sold"].sum().reset_index()

    models = {}
    for medicine in grouped_sales["Medicine_Name"].unique():
        df = grouped_sales[grouped_sales["Medicine_Name"] == medicine].copy()
        df["Day"] = (df["Date"] - df["Date"].min()).dt.days
        X = df[["Day"]]
        y = df["Stock_Sold"]
        if len(X) > 1:
            model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
            model.fit(X, y)
            models[medicine] = model

    return models

# Function to predict days to stockout (with infinite loop guard)
def predict_days_to_stockout(models, medicine_name, stock_available):
    """Predict days until stockout with a safety cap of 365 days."""
    MAX_DAYS = 365
    if medicine_name in models:
        model = models[medicine_name]
        days = 0
        while stock_available > 0 and days < MAX_DAYS:
            predicted_sales = model.predict([[days]])[0]
            stock_available -= max(predicted_sales, 0)
            days += 1
        return days if days < MAX_DAYS else None
    return None

# Function to update inventory
def update_inventory(medicine_name, stock_available, price_per_unit, reorder_level, expiry_date):
    conn_raw = get_raw_connection()
    cursor = conn_raw.cursor()

    models = train_xgboost_model()
    days_to_stockout = predict_days_to_stockout(models, medicine_name, stock_available)

    update_query = """
    UPDATE Inventory 
    SET Stock_Available = ?, 
        Price_per_Unit = ?, 
        Reorder_Level = ?, 
        Days_to_Stockout = ?, 
        Expiry_Date = ?
    WHERE Medicine_Name = ?
    """
    cursor.execute(update_query, (stock_available, price_per_unit, reorder_level, days_to_stockout, expiry_date, medicine_name))
    conn_raw.commit()
    conn_raw.close()
    
    stockout_msg = f"Days to Stockout: {days_to_stockout}" if days_to_stockout else "Days to Stockout: >365 (stable)"
    st.success(f"Updated {medicine_name} successfully! {stockout_msg}, Expiry Date: {expiry_date}")

# Function to update inventory after an order is placed
def update_inventory_after_order(cart):
    conn_raw = get_raw_connection()
    cursor = conn_raw.cursor()
    for medicine, details in cart.items():
        new_stock = details['stock_available'] - details['quantity']
        cursor.execute("""
            UPDATE Inventory
            SET Stock_Available = ?
            WHERE Medicine_Name = ?
        """, (new_stock, medicine))
    conn_raw.commit()
    conn_raw.close()

# Inventory management interface
def inventory_management():
    st.title("📦 Inventory Management")
    option = st.radio("Select an Option:", ["View Inventory", "Manage Inventory", "Orders and Payments"])

    if option == "View Inventory":
        df = get_inventory_data()
        st.dataframe(df)

    elif option == "Manage Inventory":
        df = get_inventory_data()
        medicine_selected = st.selectbox("Select Medicine", df["Medicine_Name"].unique())

        stock_available = st.number_input("Stock Available", min_value=0, value=int(df[df["Medicine_Name"] == medicine_selected]["Stock_Available"].values[0]))
        price_per_unit = st.number_input("Price per Unit", min_value=0.0, value=float(df[df["Medicine_Name"] == medicine_selected]["Price_per_Unit"].values[0]))
        reorder_level = st.number_input("Reorder Level", min_value=0, value=int(df[df["Medicine_Name"] == medicine_selected]["Reorder_Level"].values[0]))
        existing_expiry = df[df["Medicine_Name"] == medicine_selected]["Expiry_Date"].values[0]
        expiry_date = st.date_input("Select Expiry Date", value=pd.to_datetime(existing_expiry))

        if st.button("Update Inventory"):
            expiry_date_str = expiry_date.strftime("%Y-%m-%d")
            update_inventory(medicine_selected, stock_available, price_per_unit, reorder_level, expiry_date_str)

    elif option == "Orders and Payments":
        orders_and_payments()

# Orders and payments interface
def orders_and_payments():
    st.subheader("🛒 Orders and Payments")

    # Initialize session state for cart if not already present
    if 'cart' not in st.session_state:
        st.session_state.cart = {}

    df = get_inventory_data()
    medicine_selected = st.selectbox("Select Medicine", df["Medicine_Name"].unique())
    quantity = st.number_input("Quantity", min_value=1, value=1)
    price_per_unit = float(df[df["Medicine_Name"] == medicine_selected]["Price_per_Unit"].values[0])
    stock_available = int(df[df["Medicine_Name"] == medicine_selected]["Stock_Available"].values[0])

    if st.button("Add to Cart"):
        # Check cumulative cart quantity against available stock
        existing_qty = st.session_state.cart.get(medicine_selected, {}).get('quantity', 0)
        total_requested = existing_qty + quantity
        
        if total_requested > stock_available:
            st.error(f"Cannot add {quantity} units of {medicine_selected} to cart. "
                     f"Already {existing_qty} in cart, only {stock_available} units available.")
        else:
            if medicine_selected in st.session_state.cart:
                st.session_state.cart[medicine_selected]['quantity'] += quantity
            else:
                st.session_state.cart[medicine_selected] = {
                    'quantity': quantity,
                    'price_per_unit': price_per_unit,
                    'stock_available': stock_available
                }
            st.success(f"Added {quantity} units of {medicine_selected} to cart.")

    if st.session_state.cart:
        st.subheader("Cart Items")
        cart_items = []
        total_cost = 0
        for medicine, details in st.session_state.cart.items():
            item_total = details['quantity'] * details['price_per_unit']
            total_cost += item_total
            cart_items.append({
                'Medicine': medicine,
                'Quantity': details['quantity'],
                'Price per Unit': details['price_per_unit'],
                'Total': item_total
            })
        st.table(cart_items)
        st.write(f"**Total Cost: ₹{total_cost:.2f}**")

        if st.button("Place Order"):
            update_inventory_after_order(st.session_state.cart)
            st.session_state.cart = {}  # Clear the cart after placing the order
            st.success("Order placed successfully! Inventory has been updated.")
        


DB_PATH = "pharmastock_optimizer.db"

# Function: Fetch sales by date
def fetch_sales_by_date(date_filter):
    conn_raw = get_raw_connection()
    query = "SELECT * FROM Sales WHERE Date = ?"
    df = pd.read_sql_query(query, conn_raw, params=[date_filter])
    conn_raw.close()
    return df

# Function: Fetch sales by month & year (using Month column directly)
def fetch_sales_by_month(month_name, year_filter):
    """Fetch sales filtered by month name and year using the Month column."""
    conn_raw = get_raw_connection()
    query = "SELECT * FROM Sales WHERE Year = ? AND Month = ?"
    df = pd.read_sql_query(query, conn_raw, params=[year_filter, month_name])
    conn_raw.close()
    return df

# Function: Fetch sales by year
def fetch_sales_by_year(year_filter):
    conn_raw = get_raw_connection()
    query = "SELECT * FROM Sales WHERE Year = ?"
    df = pd.read_sql_query(query, conn_raw, params=[year_filter])
    conn_raw.close()
    return df

# Function to fetch supplier-medicine-stock data
def get_supplier_stock():
    conn_raw = get_raw_connection()
    query = """
    SELECT m.Medicine_Name, m.Supplier_ID, i.Stock_Available
    FROM Medicine m
    JOIN Inventory i ON m.Medicine_Name = i.Medicine_Name
    """
    df = pd.read_sql_query(query, conn_raw)
    
    # Rename columns to ensure consistency
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns={"stock_available": "Stock_Available"}, inplace=True)

    conn_raw.close()
    return df

def get_stock_data():
    conn_raw = get_raw_connection()
    query = "SELECT Medicine_Name, Stock_Available FROM Inventory"
    stock_data = pd.read_sql_query(query, conn_raw)
    conn_raw.close()
    return stock_data

# Function to render inventory summary in sidebar
def render_sidebar_stats():
    """Renders inventory summary statistics in the sidebar."""
    conn_raw = get_raw_connection()
    cursor = conn_raw.cursor()
    query = """
    SELECT 
        COUNT(DISTINCT Medicine_Name) AS Total_Medicines, 
        SUM(CASE WHEN Stock_Available < Reorder_Level THEN 1 ELSE 0 END) AS Low_Stock_Medicines, 
        SUM(CASE WHEN Expiry_Date < DATE('now') THEN 1 ELSE 0 END) AS Expired_Medicines
    FROM Inventory;
    """
    cursor.execute(query)
    result = cursor.fetchone()
    conn_raw.close()

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write(f"**Total Medicines:** {result[0]}")
    st.sidebar.write(f"**Low Stock Medicines:** {result[1]}")
    st.sidebar.write(f"**Expired Medicines:** {result[2]}")

# --- Main Application ---

if st.session_state.authenticated:
    # --- Dashboard ---
    
    st.sidebar.title("PharmaStock Optimizer")
   
    st.sidebar.button("Logout", on_click=logout_callback)
    navigation_options = ["Dashboard", "Inventory", "Sales", "Medicines", "Suppliers"]
    selected_page = st.sidebar.radio("Go to", navigation_options)
    
    if selected_page == "Dashboard":
        st.title(f"Welcome, {st.session_state.username}!")
        st.write("This is your dashboard.")
        stock_data = get_data()
        st.dataframe(stock_data, use_container_width=True)
        # Define colors: Red for low stock, continuous "Blues" for others
        stock_data["Bar_Color"] = stock_data.apply(lambda row: "#d62728" if row["Stock_Available"] <= row["Reorder_Level"] else None, axis=1)

        # Separate stock levels
        low_stock = stock_data[stock_data["Stock_Available"] <= stock_data["Reorder_Level"]]
        sufficient_stock = stock_data[stock_data["Stock_Available"] > stock_data["Reorder_Level"]]

        # Plot for low stock (Red)
        fig_low = px.bar(low_stock, 
                        x="Medicine_Name", 
                        y="Stock_Available",
                        title="Stock Levels",
                        labels={"Medicine_Name": "Medicine", "Stock_Available": "Quantity"},
                        color_discrete_sequence=["#d62728"])  # Red

        # Plot for sufficient stock (Blues)
        fig_sufficient = px.bar(sufficient_stock, 
                                x="Medicine_Name", 
                                y="Stock_Available",
                                labels={"Medicine_Name": "Medicine", "Stock_Available": "Quantity"},
                                color="Stock_Available",
                                color_continuous_scale="Blues")  # Gradient blue

        # Combine both plots
        fig_low.add_traces(fig_sufficient.data)

        # Display chart
        st.plotly_chart(fig_low, use_container_width=True)

        # Render sidebar stats (single location, no duplication)
        render_sidebar_stats()
        
        
    elif selected_page == "Sales":
        
        # Load Sales Data
        sales_data = get_sales_data()
        # Streamlit UI
        st.title("📊 Sales Data Filtering")

        with st.sidebar:
            st.header("🔍 Sales Filters")
            date_option = st.radio("📅 Filter by Date:", ["None", "Select a Date"])
            if date_option == "Select a Date":
                date_f = st.date_input("📅 Select a Date:", value=None)
                date_filter = date_f.strftime("%Y-%b-%d") if date_f else None
            else:
                date_filter = "None"
            month_filter = st.selectbox("Select Month:", ["None", "All"] + list(calendar.month_name[1:]))  # Full month names
            year_filter = st.selectbox("Select Year:", ["None"] + [str(y) for y in range(2020, 2031)])  # Adjust as needed

        # Filtering Logic
        if date_filter and date_filter != "None":
            st.write(f"📅 Filtering by Date: {date_filter}")
            df = fetch_sales_by_date(date_filter)
            st.write(df if not df.empty else "No data found for this date.")

        elif month_filter != "None" and month_filter != "All":
            if year_filter == "None":
                st.warning("⚠️ Please select a year for month filtering.")
            else:
                # Use full month name to filter against the Month column directly
                st.write(f"📅 Filtering by Month: {month_filter} and Year: {year_filter}")
                df = fetch_sales_by_month(month_filter, year_filter)
                st.write(df if not df.empty else "No data found for this month and year.")

        elif year_filter != "None":
            st.write(f"📅 Filtering by Year: {year_filter}")
            df = fetch_sales_by_year(year_filter)
            st.write(df if not df.empty else "No data found for this year.")

        else:
            st.write("📊 Displaying all sales data (Default View).")
            conn_raw = get_raw_connection()
            df = pd.read_sql_query("SELECT * FROM Sales", conn_raw)
            conn_raw.close()
            st.write(df if not df.empty else "No sales data available.")
            
        # Sales Visualization
        sales_data1 = get_sales_data()
        sales_data1["Date"] = pd.to_datetime(sales_data1["Date"], format="%Y-%b-%d")

        st.sidebar.header("📈 Sales Visualization")

        # Medicine Filter
        medicine_filter = st.sidebar.selectbox(
            "🔬 Select Medicine", 
            ["None"] + sorted(sales_data["Medicine_Name"].dropna().unique()), 
            key="medicine_select"
        )
        # Date Range Filters
        from_date = st.sidebar.date_input("📆 From Date", value=sales_data1["Date"].min())
        to_date = st.sidebar.date_input("📆 To Date", value=sales_data1["Date"].max())

        # Convert date_input to pandas datetime
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)

        # Apply Filters to Graph Data
        graph_data = sales_data1[(sales_data1["Date"] >= from_date) & (sales_data1["Date"] <= to_date)]

        if medicine_filter != "None":
            graph_data = graph_data[graph_data["Medicine_Name"] == medicine_filter]

        # Sales Trend Graph (Bar Chart)
        fig = px.bar(
            graph_data, x="Date", y="Stock_Sold", color="Medicine_Name",
            title="📊 Medicine Sales Over Time",
            labels={"Date": "Date", "Stock_Sold": "Units Sold", "Medicine_Name": "Medicine"},
            barmode="group"
        )

        # Improve Interaction (Zoom, Hover, & Responsiveness)
        fig.update_layout(
            width=1000, height=600,
            xaxis=dict(fixedrange=False),
            yaxis=dict(fixedrange=False),
            dragmode="zoom",
            hovermode="x"
        )
        
        # Sales Trend Graph (Line Chart)
        fig1 = px.line(graph_data, x="Date", y="Stock_Sold", color="Medicine_Name",
                    title="📊 Medicine Sales Over Time",
                    labels={"Date": "Date", "Stock_Sold": "Units Sold", "Medicine_Name": "Medicine"},
                    markers=True)

        # Increase Figure Size
        fig1.update_layout(width=1000, height=600)  

        # Display the Graphs
        bar = st.toggle("Bar Chart", disabled=False)
        if bar:
            st.plotly_chart(fig, use_container_width=True)
            
        line = st.toggle("Line Chart", disabled=False)
        if line:
            st.plotly_chart(fig1, use_container_width=True)


    elif selected_page == "Suppliers":
        # Fetch data
        supplier_stock_data = get_supplier_stock()

        # Step 1: Extract and clean supplier list
        supplier_mapping = {}  # Dictionary to store supplier → medicines mapping
        unique_suppliers = set()  # Set to store unique suppliers

        for _, row in supplier_stock_data.iterrows():
            medicine = row["Medicine_Name"]
            suppliers = row["Supplier_ID"].split(",")  # Split comma-separated suppliers
            stock = row.get("Stock_Available", 0)  # Use .get() to prevent KeyError

            for supplier in suppliers:
                supplier = supplier.strip()  # Remove extra spaces
                unique_suppliers.add(supplier)  # Store unique suppliers

                # Step 2: Map suppliers to medicines & stock levels
                if supplier not in supplier_mapping:
                    supplier_mapping[supplier] = []
                supplier_mapping[supplier].append((medicine, stock))

        # Convert set to sorted list
        unique_suppliers = sorted(unique_suppliers)

        # Sidebar filter
        st.sidebar.header("🔍 Select a Supplier")
        selected_supplier = st.sidebar.selectbox("Select Supplier", ["Select"] + unique_suppliers)

        # Step 3: Display Table & Stock Levels as Graph
        if selected_supplier and selected_supplier != "Select":
            medicine_stock_list = supplier_mapping.get(selected_supplier, [])
            
            if medicine_stock_list:
                # Convert to DataFrame for display
                stock_df = pd.DataFrame(medicine_stock_list, columns=["Medicine", "Stock Available"])

                # Display table
                st.subheader(f"📋 Medicines Supplied by {selected_supplier}")
                st.dataframe(stock_df)

                # Create a bar chart
                fig = px.bar(stock_df, x="Medicine", y="Stock Available", 
                            title=f"Stock Levels of Medicines Supplied by {selected_supplier}",
                            labels={"Stock Available": "Stock Level"},
                            color="Stock Available",
                            color_continuous_scale="Blues")
                
                st.plotly_chart(fig)  # Display the bar chart
            else:
                st.warning(f"No stock data available for supplier: {selected_supplier}")
                
    elif selected_page == "Inventory":
        inventory_management()
        
    elif selected_page == "Medicines":
        medicines_info = [
            {
                "Medicine_Name": "Diclofenac",
                "ATC_Code": "M01AB05",
                "Description": "A nonsteroidal anti-inflammatory drug (NSAID) used to treat pain and inflammatory diseases such as gout.",
                "Uses": "Reduces inflammation and pain in conditions like arthritis or acute injury.",
                "Cautions": "May increase the risk of serious cardiovascular events. Use with caution in patients with cardiovascular disease.",
                "Dosage": "50 mg two to three times daily; maximum daily dose should not exceed 150 mg.",
                "Alternatives": ["Ibuprofen", "Naproxen"]
            },
            {
                "Medicine_Name": "Ketoprofen",
                "ATC_Code": "M01AE03",
                "Description": "An NSAID used for its analgesic and anti-inflammatory properties.",
                "Uses": "Treats pain, fever, and inflammation in conditions like rheumatoid arthritis and osteoarthritis.",
                "Cautions": "May cause gastrointestinal discomfort and increase cardiovascular risk.",
                "Dosage": "50-100 mg up to three times daily; do not exceed 300 mg per day.",
                "Alternatives": ["Diclofenac", "Ibuprofen"]
            },
            {
                "Medicine_Name": "Naproxen",
                "ATC_Code": "M01AE02",
                "Description": "An NSAID effective in reducing pain, fever, inflammation, and stiffness.",
                "Uses": "Commonly used for arthritis, menstrual cramps, tendinitis, and gout.",
                "Cautions": "May increase the risk of heart attack or stroke; use the lowest effective dose for the shortest duration.",
                "Dosage": "250-500 mg twice daily; maximum daily dose is 1000 mg.",
                "Alternatives": ["Ibuprofen", "Diclofenac"]
            },
            {
                "Medicine_Name": "Ibuprofen",
                "ATC_Code": "M01AE01",
                "Description": "An NSAID used for treating pain, fever, and inflammation.",
                "Uses": "Effective for headaches, dental pain, menstrual cramps, muscle aches, or arthritis.",
                "Cautions": "Can increase the risk of heart attack or stroke, especially with prolonged use.",
                "Dosage": "200-400 mg every 4-6 hours as needed; maximum daily dose is 1200 mg.",
                "Alternatives": ["Naproxen", "Diclofenac"]
            },
            {
                "Medicine_Name": "Dexibuprofen",
                "ATC_Code": "M01AE14",
                "Description": "The active enantiomer of ibuprofen with anti-inflammatory and analgesic properties.",
                "Uses": "Used for pain relief in osteoarthritis, dysmenorrhea, and dental pain.",
                "Cautions": "Similar cardiovascular and gastrointestinal risks as other NSAIDs.",
                "Dosage": "200-400 mg three times daily; do not exceed 1200 mg per day.",
                "Alternatives": ["Ibuprofen", "Naproxen"]
            },
            {
                "Medicine_Name": "Aspirin",
                "ATC_Code": "N02BA01",
                "Description": "An NSAID and antiplatelet agent used to reduce pain, fever, or inflammation.",
                "Uses": "Treats mild to moderate pain, reduces fever, and used for its antiplatelet effects in cardiovascular conditions.",
                "Cautions": "Can cause gastrointestinal bleeding and should be used cautiously in patients with bleeding disorders.",
                "Dosage": "Pain/fever: 325-650 mg every 4-6 hours; Cardiovascular protection: 81-325 mg daily.",
                "Alternatives": ["Paracetamol", "Ibuprofen"]
            },
            {
                "Medicine_Name": "Diflunisal",
                "ATC_Code": "N02BA11",
                "Description": "A salicylate NSAID with analgesic and anti-inflammatory properties.",
                "Uses": "Used to relieve mild to moderate pain and inflammation in conditions like osteoarthritis.",
                "Cautions": "May increase the risk of serious cardiovascular events and gastrointestinal bleeding.",
                "Dosage": "500 mg initially, followed by 250-500 mg every 8-12 hours; maximum daily dose is 1500 mg.",
                "Alternatives": ["Naproxen", "Ibuprofen"]
            },
            {
                "Medicine_Name": "Paracetamol",
                "ATC_Code": "N02BE01",
                "Description": "Also known as acetaminophen, used to treat pain and fever.",
                "Uses": "Effective for mild to moderate pain relief such as headaches and muscle aches, and for reducing fever.",
                "Cautions": "Overdose can lead to severe liver damage; avoid combining with other products containing acetaminophen.",
                "Dosage": "500 mg to 1 g every 4-6 hours as needed; maximum daily dose should not exceed 4 g.",
                "Alternatives": ["Ibuprofen", "Aspirin"]
            },
            {
                "Medicine_Name": "Phenacetin",
                "ATC_Code": "N02BE04",
                "Description": "An analgesic and antipyretic agent, now largely withdrawn due to nephrotoxicity.",
                "Uses": "Historically used for pain and fever relief.",
                "Cautions": "Associated with kidney damage and carcinogenicity; no longer recommended.",
                "Dosage": "Not applicable due to withdrawal from the market.",
                "Alternatives": ["Paracetamol", "Ibuprofen"]
            },
            {
                "Medicine_Name": "Diazepam",
                "ATC_Code": "N05BA01",
                "Description": "A benzodiazepine used for its anxiolytic, anticonvulsant, and muscle relaxant properties.",
                "Uses": "Treats anxiety, seizures, muscle spasms, and alcohol withdrawal symptoms.",
                "Cautions": "Can cause sedation and dependence; use with caution in patients with a history of substance abuse.",
                "Dosage": "2-10 mg two to four times daily, depending on the condition being treated.",
                "Alternatives": ["Alprazolam", "Lorazepam"]
            },
            {
                "Medicine_Name": "Alprazolam",
                "ATC_Code": "N05BA12",
                "Description": "A short-acting benzodiazepine with anxiolytic effects.",
                "Uses": "Primarily used to manage anxiety disorders and panic attacks.",
                "Cautions": "Risk of dependence and withdrawal symptoms; should be used for short durations.",
                "Dosage": "0.25-0.5 mg three times daily; maximum daily dose is 4 mg.",
                "Alternatives": ["Diazepam", "Clonazepam"]
            },
            {
                "Medicine_Name": "Zolpidem",
                "ATC_Code": "N05CF02",
                "Description": "A sedative-hypnotic used for the short-term treatment of insomnia.",
                "Uses": "Helps individuals fall asleep faster and reduces nighttime awakenings.",
                "Cautions": "May cause dizziness, daytime drowsiness, and complex sleep behaviors like sleepwalking. Use with caution in patients with a history of substance abuse.",
                "Dosage": "5 mg for women and 5-10 mg for men immediately before bedtime; maximum daily dose is 10 mg.",
                "Alternatives": ["Eszopiclone", "Ramelteon"]
            },
            {
                "Medicine_Name": "Eszopiclone",
                "ATC_Code": "N05CF04",
                "Description": "A non-benzodiazepine hypnotic agent used for the treatment of insomnia.",
                "Uses": "Improves sleep onset and maintenance in individuals with insomnia.",
                "Cautions": "May cause next-day impairment, dizziness, and hallucinations. Risk of dependence with prolonged use.",
                "Dosage": "1 mg immediately before bedtime; may increase to 2-3 mg if clinically indicated.",
                "Alternatives": ["Zolpidem", "Ramelteon"]
            },
            {
                "Medicine_Name": "Salbutamol",
                "ATC_Code": "R03AC02",
                "Description": "A short-acting β₂-adrenergic receptor agonist used to relieve bronchospasm in conditions like asthma and COPD. Also known as Albuterol.",
                "Uses": "Provides quick relief from acute episodes of bronchospasm.",
                "Cautions": "May cause tachycardia, palpitations, and tremors. Use with caution in patients with cardiovascular disorders.",
                "Dosage": "Inhalation: 100-200 mcg every 4-6 hours as needed.",
                "Alternatives": ["Levalbuterol", "Terbutaline"]
            },
            {
                "Medicine_Name": "Formoterol",
                "ATC_Code": "R03AC13",
                "Description": "A long-acting β₂-adrenergic receptor agonist used in the management of asthma and COPD.",
                "Uses": "Provides long-term control of bronchospasm and improves lung function.",
                "Cautions": "Not for relief of acute bronchospasm. May increase the risk of asthma-related death; should be used with an inhaled corticosteroid.",
                "Dosage": "Inhalation: 12 mcg twice daily.",
                "Alternatives": ["Salmeterol", "Indacaterol"]
            },
            {
                "Medicine_Name": "Budesonide",
                "ATC_Code": "R03BA02",
                "Description": "An inhaled corticosteroid used for the maintenance treatment of asthma and COPD.",
                "Uses": "Reduces inflammation in the airways, leading to improved breathing.",
                "Cautions": "May cause oral thrush; rinse mouth after use. Use with caution in patients with tuberculosis or untreated infections.",
                "Dosage": "Inhalation: 200-400 mcg twice daily.",
                "Alternatives": ["Fluticasone", "Beclomethasone"]
            },
            {
                "Medicine_Name": "Loratadine",
                "ATC_Code": "R06AX13",
                "Description": "A second-generation antihistamine used to treat allergic rhinitis and urticaria.",
                "Uses": "Relieves symptoms of hay fever and other allergies, such as sneezing, runny nose, and itching.",
                "Cautions": "Generally well-tolerated; may cause headache or dry mouth.",
                "Dosage": "10 mg once daily.",
                "Alternatives": ["Cetirizine", "Fexofenadine"]
            },
            {
                "Medicine_Name": "Cetirizine",
                "ATC_Code": "R06AE07",
                "Description": "A second-generation antihistamine used to treat symptoms of allergies such as runny nose, red and watery eyes, and itchiness.",
                "Uses": "Effective for allergic rhinitis and chronic urticaria.",
                "Cautions": "May cause drowsiness in some individuals; use caution when driving or operating machinery.",
                "Dosage": "5-10 mg once daily.",
                "Alternatives": ["Loratadine", "Fexofenadine"]
            },
            {
                "Medicine_Name": "Diphenhydramine",
                "ATC_Code": "R06AA02",
                "Description": "A first-generation antihistamine with sedative and antiemetic properties.",
                "Uses": "Treats allergic reactions, motion sickness, and used as a sleep aid.",
                "Cautions": "Causes significant drowsiness; avoid alcohol and other CNS depressants.",
                "Dosage": "25-50 mg every 4-6 hours; maximum daily dose is 300 mg.",
                "Alternatives": ["Chlorpheniramine", "Promethazine"]
            },
            {
                "Medicine_Name": "Amoxicillin",
                "ATC_Code": "J01CA04",
                "Description": "A broad-spectrum β-lactam antibiotic used to treat various bacterial infections.",
                "Uses": "Effective against infections of the ear, nose, throat, skin, and urinary tract.",
                "Cautions": "May cause allergic reactions, including anaphylaxis; use with caution in patients with a history of penicillin allergy.",
                "Dosage": "500 mg every 8 hours or 875 mg every 12 hours.",
                "Alternatives": ["Ampicillin", "Cefuroxime"]
            },
            {
                "Medicine_Name": "Penicillin",
                "ATC_Code": "J01CE02",
                "Description": "A narrow-spectrum β-lactam antibiotic used to treat streptococcal infections. Also known as Penicillin V.",
                "Uses": "Treats mild to moderate infections like pharyngitis and tonsillitis.",
                "Cautions": "May cause hypersensitivity reactions; use with caution in patients with a history of allergies.",
                "Dosage": "500 mg every 6-8 hours.",
                "Alternatives": ["Amoxicillin", "Erythromycin"]
            },
            {
                "Medicine_Name": "Cefalexin",
                "ATC_Code": "J01DB01",
                "Description": "A first-generation cephalosporin antibiotic used to treat a variety of bacterial infections.",
                "Uses": "Effective against respiratory tract infections, skin infections, and urinary tract infections.",
                "Cautions": "Use with caution in patients with a history of penicillin allergy due to potential cross-reactivity.",
                "Dosage": "250-500 mg every 6 hours; maximum daily dose is 4 g.",
                "Alternatives": ["Cefadroxil", "Amoxicillin"]
            },
            {
                "Medicine_Name": "Ceftriaxone",
                "ATC_Code": "J01DD04",
                "Description": "A third-generation cephalosporin antibiotic used to treat a wide range of bacterial infections.",
                "Uses": "Effective against severe infections like meningitis, pneumonia, and infections of the abdomen, joints, and skin.",
                "Cautions": "May cause allergic reactions; use with caution in patients with a history of penicillin allergy. Can lead to pseudomembranous colitis.",
                "Dosage": "1-2 g once daily; maximum daily dose is 4 g.",
                "Alternatives": ["Cefotaxime", "Levofloxacin"]
            },
            {
                "Medicine_Name": "Cefixime",
                "ATC_Code": "J01DD08",
                "Description": "An oral third-generation cephalosporin antibiotic used to treat various bacterial infections.",
                "Uses": "Effective against respiratory tract infections, urinary tract infections, and gonorrhea.",
                "Cautions": "Use with caution in patients with a history of penicillin allergy. May cause gastrointestinal disturbances.",
                "Dosage": "200-400 mg once daily or divided into two doses.",
                "Alternatives": ["Cefpodoxime", "Amoxicillin-Clavulanate"]
            },
            {
                "Medicine_Name": "Atorvastatin",
                "ATC_Code": "C10AA05",
                "Description": "A statin medication used to lower blood cholesterol levels and reduce the risk of cardiovascular disease.",
                "Uses": "Treats hypercholesterolemia and prevents cardiovascular events in high-risk patients.",
                "Cautions": "May cause muscle pain or weakness; rare risk of rhabdomyolysis. Monitor liver enzymes before and during treatment.",
                "Dosage": "10-80 mg once daily.",
                "Alternatives": ["Rosuvastatin", "Simvastatin"]
            },
            {
                "Medicine_Name": "Rosuvastatin",
                "ATC_Code": "C10AA07",
                "Description": "A statin medication used to treat high cholesterol and prevent cardiovascular disease.",
                "Uses": "Lowers LDL cholesterol and triglycerides; increases HDL cholesterol.",
                "Cautions": "Similar to atorvastatin; monitor for muscle symptoms and liver enzyme abnormalities.",
                "Dosage": "5-40 mg once daily.",
                "Alternatives": ["Atorvastatin", "Pravastatin"]
            },
            {
                "Medicine_Name": "Metformin",
                "ATC_Code": "A10BA02",
                "Description": "An oral antidiabetic medication used to treat type 2 diabetes.",
                "Uses": "Improves blood sugar control in adults with type 2 diabetes.",
                "Cautions": "May cause gastrointestinal upset; rare risk of lactic acidosis, especially in renal impairment.",
                "Dosage": "500-1000 mg twice daily; maximum daily dose is 2000-2500 mg.",
                "Alternatives": ["Gliclazide", "Sitagliptin"]
            },
            {
                "Medicine_Name": "Gliclazide",
                "ATC_Code": "A10BB09",
                "Description": "A sulfonylurea antidiabetic medication used to treat type 2 diabetes.",
                "Uses": "Stimulates insulin secretion to lower blood sugar levels.",
                "Cautions": "Risk of hypoglycemia; use with caution in elderly patients and those with renal or hepatic impairment.",
                "Dosage": "40-320 mg daily, taken as a single dose or in divided doses.",
                "Alternatives": ["Glimepiride", "Metformin"]
            },
            {
                "Medicine_Name": "Sitagliptin",
                "ATC_Code": "A10BH01",
                "Description": "A DPP-4 inhibitor used to improve glycemic control in adults with type 2 diabetes.",
                "Uses": "Enhances incretin levels, increasing insulin release and decreasing glucagon levels.",
                "Cautions": "Generally well-tolerated; rare cases of pancreatitis reported.",
                "Dosage": "100 mg once daily.",
                "Alternatives": ["Vildagliptin", "Linagliptin"]
            },
            {
                "Medicine_Name": "Ethinylestradiol",
                "ATC_Code": "G03AA10",
                "Description": "A synthetic form of estrogen used in various hormonal contraceptives.",
                "Uses": "Component of combined oral contraceptive pills.",
                "Cautions": "Increases risk of thromboembolic events; contraindicated in smokers over 35 years and those with certain cardiovascular conditions.",
                "Dosage": "Typically 20-35 mcg daily in combination with a progestin.",
                "Alternatives": ["Estradiol", "Mestranol"]
            },
            {
                "Medicine_Name": "Levonorgestrel",
                "ATC_Code": "G03AC03",
                "Description": "A progestin used in various hormonal contraceptives and emergency contraception.",
                "Uses": "Prevents ovulation and fertilization; used in birth control pills and intrauterine devices.",
                "Cautions": "May cause menstrual irregularities; not recommended for routine contraception in women with certain health conditions.",
                "Dosage": "Emergency contraception: 1.5 mg as a single dose within 72 hours of unprotected intercourse.",
                "Alternatives": ["Ulipristal acetate", "Copper IUD"]
            },
            {
                "Medicine_Name": "Methotrexate",
                "ATC_Code": "L04AX03",
                "Description": "An antimetabolite and antifolate drug used to treat cancer and autoimmune diseases.",
                "Uses": "Treats certain types of cancer, severe psoriasis, and rheumatoid arthritis.",
                "Cautions": "Can cause severe toxicity; monitor blood counts, liver and kidney function. Contraindicated in pregnancy.",
                "Dosage": "Varies widely depending on indication; for rheumatoid arthritis, typically 7.5-25 mg once weekly.",
                "Alternatives": ["Leflunomide", "Sulfasalazine"]
            },
            {
                "Medicine_Name": "Cyclosporine",
                "ATC_Code": "L04AD01",
                "Description": "An immunosuppressant drug used to prevent organ rejection and treat autoimmune conditions.",
                "Uses": "Prevents rejection in organ transplantation; treats severe rheumatoid arthritis and psoriasis.",
                "Cautions": "May cause nephrotoxicity, hypertension, and increased risk of infections; monitor kidney function and blood pressure closely.",
                "Dosage": "Depends on indication and patient weight; for transplantation, typically 5-15 mg/kg/day in divided doses.",
                "Alternatives": ["Tacrolimus", "Azathioprine"]
            }
        ]
        
        # Fetch stock data
        stock_data = get_stock_data()

        # Create a dictionary for quick lookup of medicine information
        medicine_dict = {medicine["Medicine_Name"]: medicine for medicine in medicines_info}

        # Streamlit app
        st.title("Medicine Information")

        # Dropdown menu for medicine selection
        medicine_names = list(medicine_dict.keys())
        selected_medicine = st.selectbox("Select a medicine:", medicine_names)

        # Display medicine details
        if selected_medicine:
            medicine = medicine_dict[selected_medicine]
            st.subheader(medicine["Medicine_Name"])
            st.write(f"**ATC Code:** {medicine['ATC_Code']}")
            st.write(f"**Description:** {medicine['Description']}")
            st.write(f"**Uses:** {medicine['Uses']}")
            st.write(f"**Cautions:** {medicine['Cautions']}")
            st.write(f"**Dosage:** {medicine['Dosage']}")
            st.write("**Alternatives:**")
            for alt in medicine["Alternatives"]:
                st.write(f"- {alt}")
            
            # Fetch and display stock availability
            stock_info = stock_data[stock_data["Medicine_Name"] == selected_medicine]
            if not stock_info.empty:
                stock_available = int(stock_info["Stock_Available"].values[0])
                if stock_available > 0:
                    st.write(f"**Stock Available:** {stock_available} units")
                else:
                    st.write("**Stock Available:** Out of stock")
            else:
                st.write("**Stock Available:** Information not available")
                        
else:
    
    # --- Authentication Page Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style='display: flex; align-items: center; min-height: 70vh; padding: 20px;'>
            <h1 style='font-size: clamp(2.5em, 5vw, 5em); line-height: 1.1; font-weight: 800; 
                       color: #1a1a2e; white-space: nowrap;'>
                PharmaStock<br>Optimizer
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Use CSS spacing for vertical alignment
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        
        option = st.selectbox("Choose an option:", ["Login", "Register", "Manage Account"])

        if option == "Login":
            st.subheader("Login")
            st.text_input("Username", key='login_username')
            st.text_input("Password", type="password", key='login_password')
            st.button("Login", on_click=login_callback)

        elif option == "Register":
            st.subheader("Register")
            st.text_input("Choose a Username", key='register_username')
            st.text_input("Enter Your Email", key='register_email')
            st.text_input("Choose a Password", type="password", key='register_password')
            st.button("Register", on_click=register_callback)

        elif option == "Manage Account":
            st.subheader("Manage Account")
            action = st.selectbox("Select Action", ["Retrieve Username", "Delete Account"])

            if action == "Retrieve Username":
                st.text_input("Enter Your Email", key='retrieve_email')
                st.button("Retrieve Username", on_click=retrieve_username_callback)

            elif action == "Delete Account":
                st.text_input("Enter Your Username", key='delete_username')
                st.text_input("Enter Your Password", type="password", key='delete_password')
                st.button("Delete Account", on_click=delete_account_callback)
