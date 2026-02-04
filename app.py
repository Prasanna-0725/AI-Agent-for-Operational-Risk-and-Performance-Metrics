from flask import Flask, render_template, request, jsonify, url_for
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib 
import openai 
import requests 
import yfinance as yf # Used for financial metrics
from dotenv import load_dotenv 

# --- SETUP ---
load_dotenv()
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
except Exception:
    print("Warning: OPENAI_API_KEY not found or invalid. AI analysis will be unavailable.")
    client = None

app = Flask(__name__)
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# OperationalRiskAgent Class
# -----------------------------
class OperationalRiskAgent:
    """
    AI Agent for dynamic operational risk & financial performance insights.
    """
    def __init__(self, openai_client):
        self.client = openai_client
        print("🌐 AI Operational Risk Agent initialized (online mode).")

    def fetch_news(self, query: str) -> str:
        """Fetch recent news using DuckDuckGo (free API)."""
        try:
            search_query = query[:100]
            url = f"https://api.duckduckgo.com/?q={search_query}&format=json&t=ai_risk_agent"
            response = requests.get(url, timeout=5)
            data = response.json()
            summary = data.get("AbstractText") or data.get("Abstract") or "No relevant news found."
            return summary if summary else "No relevant news found."
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return f"Error fetching news: {str(e)}"

    def fetch_financials(self, ticker: str) -> dict:
        """Fetch financial data using Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Filter and return key data
            return {
                "company": info.get("shortName", "Unknown"),
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "beta": info.get("beta", "N/A"),
                "revenue": info.get("totalRevenue", "N/A"),
                "profit": info.get("grossProfits", "N/A"),
            }
        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            return {"error": f"Error fetching financials for {ticker} or ticker not found."}

    def generate_analysis(self, query: str, financial_data: dict = {}) -> str:
        """Combines live financial + news + AI analysis."""
        if not self.client:
            return "Cannot generate analysis: OpenAI key is not configured."
        
        # News data is fetched based on the general query
        news_data = self.fetch_news(query)

        prompt = f"""
        You are an expert in operational and financial risk analysis.
        Analyze the query using the provided context. If financial data and news are available, synthesize them into the analysis.
        
        {("The user is requesting an analysis of a specific bank." if financial_data else "")}

        Query: {query}
        Financial Data: {financial_data}
        News Summary: {news_data}

        Please provide a structured analysis including:
        1. Organization Overview (Name, Sector, Market Cap)
        2. Key Financial Performance (if financial data is available)
        3. Potential Operational Risks
        4. Recent News Highlights (if news is available)
        5. Recommendations
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial and risk intelligence assistant. Be concise and professional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating OpenAI analysis: {str(e)}"

risk_agent = OperationalRiskAgent(client)

# -----------------------------
# Utility: load Kaggle dataset & Compute metrics (Operational Risk)
# -----------------------------
def load_kaggle_data():
    """
    Attempt to load a CSV at data/kaggle_operational_risk.csv.
    If not present, generate a synthetic Kaggle-style dataset.
    """
    path = os.path.join("data", "kaggle_operational_risk.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            colmap = {}
            for c in df.columns:
                lc = c.lower()
                if "event" in lc and "type" in lc:
                    colmap[c] = "Event Type"
                elif "loss" in lc and "k" in lc:
                    colmap[c] = "Loss Amount (kUSD)"
                elif "loss" in lc:
                    colmap[c] = "Loss Amount (kUSD)"
                elif "year" in lc:
                    colmap[c] = "Year"

            df = df.rename(columns=colmap)
            if "Event Type" not in df.columns:
                df["Event Type"] = np.random.choice(
                    ["System Failure", "Data Breach", "Tech Failure", "Cyber-Fraud", "Phishing"], len(df)
                )
            if "Loss Amount (kUSD)" not in df.columns:
                df["Loss Amount (kUSD)"] = np.abs(np.random.uniform(10, 500, len(df)))
            if "Year" not in df.columns:
                df["Year"] = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.year.fillna(datetime.now().year).astype(int)

            df = df[["Event Type", "Loss Amount (kUSD)", "Year"]].copy()
            return df
        except Exception as e:
            print("Error reading CSV, generating synthetic dataset:", e)

    # Fallback: generate synthetic dataset
    np.random.seed(42)
    event_types = ['Phishing', 'Cyber-Fraud', 'Tech Failure', 'Data Breach', 'System Failure']
    n = 300
    df = pd.DataFrame({
        "Event Type": np.random.choice(event_types, n, p=[0.2, 0.2, 0.25, 0.2, 0.15]),
        "Loss Amount (kUSD)": np.random.exponential(scale=20, size=n) * np.random.randint(1,5,n),
        "Year": np.random.choice(range(2015, 2025), n)
    })
    return df


def compute_metrics(df):
    """Computes Operational Risk Metrics from the Kaggle DataFrame."""
    metrics = {}
    metrics["is_financial"] = False # Flag for frontend
    metrics["total_records"] = int(len(df))
    metrics["total_loss_all"] = float(df["Loss Amount (kUSD)"].sum())
    metrics["mean_loss"] = float(df["Loss Amount (kUSD)"].mean())
    metrics["median_loss"] = float(df["Loss Amount (kUSD)"].median())

    loss_by_event = df.groupby("Event Type")["Loss Amount (kUSD)"].sum().sort_values(ascending=False)
    metrics["total_loss_by_event"] = {k: float(round(v, 2)) for k, v in loss_by_event.items()}

    yearly = df.groupby("Year")["Loss Amount (kUSD)"].sum().sort_index()
    metrics["yearly_trend"] = {int(k): float(round(v, 2)) for k, v in yearly.items()}

    topN = df.sort_values("Loss Amount (kUSD)", ascending=False).head(10)
    metrics["top_losses"] = topN[["Event Type", "Loss Amount (kUSD)", "Year"]].to_dict(orient="records")

    return metrics

# -----------------------------
# NEW UTILITY: Compute Financial Metrics (from YFinance)
# -----------------------------

def compute_financial_metrics(financial_data):
    """
    Converts YFinance data into a simplified metrics structure for dashboard display.
    """
    if financial_data.get("error") or not financial_data:
        return None

    m_cap = financial_data.get("market_cap")
    revenue = financial_data.get("totalRevenue")
    profit = financial_data.get("grossProfits")

    def format_value(val):
        if val is None or val == "N/A": return "N/A"
        try:
            val = float(val)
            if val >= 1_000_000_000:
                return f"{val / 1_000_000_000:.2f} B"
            elif val >= 1_000_000:
                return f"{val / 1_000_000:.2f} M"
            return f"{val:,.0f}"
        except:
            return str(val)

    chart_df = pd.DataFrame({
        "Metric": ["P/E Ratio", "Beta", "Market Cap (Bn)", "Revenue (Bn)"],
        "Value": [
            financial_data.get("trailingPE", 0),
            financial_data.get("beta", 0),
            m_cap / 1e9 if m_cap else 0,
            revenue / 1e9 if revenue else 0
        ]
    })

    metrics = {
        "is_financial": True, 
        "company_name": financial_data.get("company"),
        "total_records": 4, 
        "total_loss_all": m_cap if m_cap else 0, 
        "mean_loss": financial_data.get("pe_ratio", 0), 
        "median_loss": financial_data.get("beta", 0), 
        
        "financial_summary": {
            "Market Cap": format_value(m_cap),
            "P/E Ratio": financial_data.get("pe_ratio", "N/A"),
            "Beta": financial_data.get("beta", "N/A"),
            "Total Revenue": format_value(revenue),
            "Gross Profit": format_value(profit)
        },
        
        # Storing the DataFrame for chart generation, will be converted later for JSON
        "chart_data_financial": chart_df
    }
    return metrics


# -----------------------------
# UPDATED CHART GENERATION
# -----------------------------

def generate_chart(data, chart_type="bar", is_financial=False):
    """Generates a chart saved to static/risk_chart.png and returns the static url."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))

    if is_financial:
        # Financial Metrics Chart: 'data' is a Pandas DataFrame
        grouped = data.sort_values("Value")
        
        ax.barh(grouped["Metric"], grouped["Value"], color='#4CAF50') 
        ax.set_title(f"{grouped['Metric'].iloc[-1].split('(')[0]} Key Financial Metrics", fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Metric")
        
        for p in ax.patches:
            val = p.get_width()
            label = f"{val:,.2f}"
            ax.annotate(label, (val, p.get_y() + p.get_height()/2),
                                    va="center", ha="left", fontsize=9)
    
    elif chart_type == "trend":
        # Operational Trend Chart: 'data' is a Pandas DataFrame
        grouped = data.groupby("Year")["Loss Amount (kUSD)"].sum().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        grouped.plot(kind="line", marker="o", ax=ax, color='#ff9800')
        ax.set_title("Yearly Operational Loss Trends", fontsize=14)
        ax.set_ylabel("Total Loss (kUSD)")
        ax.set_xlabel("Year")
    
    else:
        # Operational Bar Chart (Loss by Event Type): 'data' is a Pandas DataFrame
        grouped = data.groupby("Event Type")["Loss Amount (kUSD)"].sum().sort_values()
        ax.barh(grouped.index, grouped.values, color='#f44336')
        ax.set_title("Total Operational Loss by Event Type", fontsize=14)
        ax.set_xlabel("Total Loss (kUSD)")
        ax.set_ylabel("Event Type")
        
        for p in ax.patches:
            ax.annotate(f"{p.get_width():,.0f}", (p.get_width(), p.get_y() + p.get_height()/2),
                                     va="center", ha="left", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join("static", "risk_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return url_for('static', filename='risk_chart.png')


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    alert = {
        "status": "SAFE",
        "message": "AI Agent is active and monitoring real-time data.",
        "details": ""
    }
    return render_template(
        'index.html',
        alert=alert,
        metrics="No metrics available yet. Ask the AI assistant for metrics insights.",
        chart_url=None
    )

# --- EXPANDED BANK MAPPING ---
def extract_bank_name_and_ticker(query):
    # Expanded Indian bank ticker map (using common NSE tickers)
    ticker_map = {
        "HDFC": "HDFCBANK.NS", "HDFC BANK": "HDFCBANK.NS",
        "SBI": "SBIN.NS", "STATE BANK OF INDIA": "SBIN.NS",
        "ICICI": "ICICIBANK.NS", "ICICI BANK": "ICICIBANK.NS",
        "AXIS": "AXISBANK.NS", "AXIS BANK": "AXISBANK.NS",
        "KOTAK": "KOTAKBANK.NS", "KOTAK BANK": "KOTAKBANK.NS",
        "INDUSIND": "INDUSINDBK.NS", "INDUSIND BANK": "INDUSINDBK.NS",
        "YES BANK": "YESBANK.NS", "YES": "YESBANK.NS",
        "BANDHAN": "BANDHANBNK.NS", "BANDHAN BANK": "BANDHANBNK.NS",
        "PNB": "PNB.NS", "PUNJAB NATIONAL BANK": "PNB.NS",
        "UNION BANK": "UNIONBANK.NS",
        "BANK OF BARODA": "BANKBARODA.NS", "BOB": "BANKBARODA.NS",
    }
    for name, ticker in ticker_map.items():
        if name.lower() in query.lower():
            return name, ticker
    return None, None


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()
    user_query = data.get("message", "")
    response_text = ""
    chart_url = None
    metrics = None
    
    metrics_keywords = ["metrics", "metric", "visual", "chart", "plot", "graph", "trend"]
    bank_name, ticker = extract_bank_name_and_ticker(user_query)

    try:
        # --- PATH 1: LOCAL OPERATIONAL RISK METRICS (e.g., "chart", "loss by event") ---
        # Prioritize local metrics if specific local keywords are used AND no bank is mentioned.
        if (any(word in user_message for word in ["kaggle", "loss", "trend", "year"]) or 
           ("metrics" in user_message and not bank_name)):
            
            df = load_kaggle_data()
            metrics = compute_metrics(df)

            chart_type = "bar"
            if "trend" in user_message or "year" in user_message:
                chart_type = "trend"

            chart_url = generate_chart(df, chart_type=chart_type)
            response_text = f"✅ Operational Risk Metrics and a **{chart_type} chart** generated from local Kaggle-like data. Check the dashboard for updates."
            
            response_text += f"\n\n**Summary:** Total Loss: {metrics['total_loss_all']:,.0f} kUSD. Mean Loss: {metrics['mean_loss']:.2f} kUSD. Top Loss Event: {list(metrics['total_loss_by_event'].keys())[0]}."


        # --- PATH 2: BANK FINANCIAL METRICS/ANALYSIS (Any bank mentioned) ---
        elif bank_name:
            
            financial_data = risk_agent.fetch_financials(ticker)
            
            # Sub-Path 2a: Generate Metrics and Chart if metrics/chart is requested.
            if any(word in user_message for word in metrics_keywords):
                metrics = compute_financial_metrics(financial_data)
                
                if metrics and not metrics.get("error"):
                    # 1. Get the DataFrame needed for chart generation
                    chart_df = metrics["chart_data_financial"]
                    
                    # 2. Generate chart using the DataFrame
                    chart_url = generate_chart(chart_df, is_financial=True)
                    
                    # 3. FIX: Convert the DataFrame to JSON-safe list format 
                    #    before passing it to jsonify.
                    metrics["chart_data_financial"] = chart_df.to_dict(orient="records") 

                    response_text = f"✅ Financial Metrics and Chart for **{metrics['company_name']}** generated from live Yahoo Finance data."
                    
                    # Create chat summary
                    summary_data = metrics.get("financial_summary", {})
                    response_text += "\n\n**Financial Summary:**"
                    for k, v in list(summary_data.items())[:3]:
                        response_text += f" {k}: **{v}**."
                
                else:
                    response_text = f"⚠️ Could not fetch financial metrics for {bank_name} ({ticker}). Ticker may be invalid or data unavailable. Falling back to AI analysis."
                    # If fetching metrics failed, proceed to AI analysis (Sub-Path 2b)

            # Sub-Path 2b: AI Agent Analysis (for general questions about the bank)
            if not chart_url or financial_data.get("error"): 
                # Run general AI analysis, passing financial data if available
                if financial_data.get("error"): financial_data = {} # Clear error for AI prompt
                response_text = risk_agent.generate_analysis(user_query, financial_data=financial_data)


        # --- PATH 3: GENERAL AI AGENT ANALYSIS (Default) ---
        else:
            response_text = risk_agent.generate_analysis(user_query)

    except Exception as e:
        print(f"FATAL ERROR IN CHAT ROUTE: {e}")
        response_text = f"⚠️ Error during processing: {str(e)}"

    return jsonify({
        "response": response_text,
        "chart_url": chart_url,
        "metrics": metrics
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)