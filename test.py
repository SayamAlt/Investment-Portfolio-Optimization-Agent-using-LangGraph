# import requests
# import time
# from dotenv import load_dotenv
# import os
# import yfinance as yf

# load_dotenv()

# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# def get_stock_quote(symbol: str) -> dict:
#     """
#     Fetch real-time stock quote using Finnhub /quote endpoint.
#     This provides only current and previous close data.
#     """
#     url = "https://finnhub.io/api/v1/quote"
#     params = {
#         "symbol": symbol,
#         "token": FINNHUB_API_KEY
#     }

#     response = requests.get(url, params=params)
#     data = response.json()

#     if not data or data.get("c") is None:
#         raise ValueError(f"No quote data available for {symbol}")

#     return {
#         "symbol": symbol,
#         "current_price": data["c"],
#         "change": data["d"],
#         "percent_change": data["dp"],
#         "high": data["h"],
#         "low": data["l"],
#         "open": data["o"],
#         "previous_close": data["pc"]
#     }

# data = get_stock_quote("AAPL")
# print(data)

# ticker = yf.Ticker("AAPL")
# print(ticker.info)
# print(type(ticker.history(period="1y")))
import os
import pandas as pd
from finnhub import Client
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Define a simple NewsItem dataclass
@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    datetime: str

# Initialize Finnhub client
finnhub_api_key = os.environ.get("FINNHUB_API_KEY")  # Make sure your API key is set in environment
finnhub_client = Client(api_key=finnhub_api_key)

# Dummy historical data to simulate your 'history' DataFrame
dates = pd.date_range(end=datetime.today(), periods=10)
history = pd.DataFrame(index=dates, data={"close": range(10)})

# Symbol to test
symbol = "AAPL"

# Function to fetch combined news
def fetch_combined_news(finnhub_client, symbol, history):
    combined_news = []
    if finnhub_client:
        try:
            # General market news
            market_news = finnhub_client.general_news('general', min_id=0)
            for article in market_news:
                combined_news.append(NewsItem(
                    headline=article.get("headline", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    datetime=str(article.get("datetime"))
                ))

            # Company-specific news for the ticker (last 7 days)
            from_date = (history.index[-1] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            to_date = history.index[-1].strftime("%Y-%m-%d")
            company_news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
            for article in company_news:
                combined_news.append(NewsItem(
                    headline=article.get("headline", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    datetime=str(article.get("datetime"))
                ))

        except Exception as e:
            print("Error fetching news:", e)
            combined_news = []

    return combined_news

# Fetch and display news
news_items = fetch_combined_news(finnhub_client, symbol, history)

# Print results
print(f"Total news articles fetched: {len(news_items)}\n")
for i, news in enumerate(news_items[:5], 1):  # Print first 5 articles for brevity
    print(f"{i}. {news.headline} | Source: {news.source} | URL: {news.url} | DateTime: {news.datetime}")