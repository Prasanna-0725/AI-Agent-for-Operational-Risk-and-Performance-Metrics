import os
import openai
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class OperationalRiskAgent:
    """
    AI Agent for dynamic operational risk & financial performance insights.
    Fetches online data, news, and financial metrics, then uses OpenAI for reasoning.
    """

    def __init__(self):
        print("🌐 AI Operational Risk Agent initialized (online mode).")

    # ----------------------------
    # FETCH NEWS ABOUT ORGANIZATION
    # ----------------------------
    def fetch_news(self, query: str) -> str:
        """
        Fetch recent news using DuckDuckGo (free API).
        """
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            data = response.json()
            summary = data.get("AbstractText") or "No relevant news found."
            return summary
        except Exception as e:
            return f"Error fetching news: {str(e)}"

    # ----------------------------
    # FETCH FINANCIAL DATA (Yahoo Finance)
    # ----------------------------
    def fetch_financials(self, ticker: str) -> dict:
        """
        Fetch financial data using Yahoo Finance.
        Example: ticker='HDFCBANK.NS' for HDFC Bank (NSE)
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "company": info.get("shortName", "Unknown"),
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "beta": info.get("beta"),
                "revenue": info.get("totalRevenue"),
                "profit": info.get("grossProfits"),
                "dividend_yield": info.get("dividendYield"),
            }
        except Exception as e:
            return {"error": f"Error fetching financials: {str(e)}"}

    # ----------------------------
    # COMBINE + ANALYZE WITH OPENAI
    # ----------------------------
    def generate_analysis(self, query: str) -> str:
        """
        Combines live financial + news + AI analysis.
        """
        # Try to guess ticker symbols for major Indian banks
        ticker_map = {
            "HDFC": "HDFCBANK.NS",
            "SBI": "SBIN.NS",
            "ICICI": "ICICIBANK.NS",
            "AXIS": "AXISBANK.NS",
            "KOTAK": "KOTAKBANK.NS",
        }

        ticker = None
        for key in ticker_map:
            if key.lower() in query.lower():
                ticker = ticker_map[key]

        # Fetch data
        financial_data = self.fetch_financials(ticker) if ticker else {}
        news_data = self.fetch_news(query)

        # Create AI prompt
        prompt = f"""
        You are an expert in operational and financial risk analysis.
        Below is data gathered from the web:

        Query: {query}
        Financial Data: {financial_data}
        News Summary: {news_data}

        Please provide a structured analysis including:
        1. Organization Overview
        2. Key Financial Performance
        3. Potential Operational Risks
        4. Recent News Highlights
        5. Recommendations
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial and risk intelligence assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating OpenAI analysis: {str(e)}"
