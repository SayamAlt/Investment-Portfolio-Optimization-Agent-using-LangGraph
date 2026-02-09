from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Any
from langchain_community.cache import InMemoryCache
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os, json, finnhub
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import cvxpy as cp
import streamlit as st

load_dotenv()

if "OPENAI_API_KEY" in st.secrets['secrets']:
    OPENAI_API_KEY = st.secrets['secrets']['OPENAI_API_KEY']
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if "FINNHUB_API_KEY" in st.secrets['secrets']:
    FINNHUB_API_KEY = st.secrets['secrets']['FINNHUB_API_KEY']
else:
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
        
cache = InMemoryCache()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=OPENAI_API_KEY, cache=cache)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

# Unified Risk Targets for Leverage & Optimization
RISK_BUDGETS = {
    "conservative": 0.15,
    "moderate": 0.25,
    "aggressive": 0.40
}

# Initialize Finnhub API client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Create data models
class UserProfile(BaseModel):
    risk_appetite: Literal["conservative", "moderate", "aggressive"] = Field(..., title="Risk Appetite", description="The level of risk tolerance of the user.")
    investment_horizon: int = Field(..., title="Investment Horizon", gt=0, description="The number of years the user plans to invest.")
    preferences: Optional[List[str]] = Field(default_factory=list, title="Preferences", description="The preferences of the user.")
    constraints: Optional[List[str]] = Field(default_factory=list, title="Constraints", description="The constraints of the user.")
    capital: float = Field(..., title="Capital", gt=0, description="The amount of capital the user has available for investment.")
    use_leverage: bool = Field(default=False, title="Use Leverage", description="Whether the user wants to use leverage for risk management.")

class TickerSelection(BaseModel):
    tickers: List[str] = Field(..., title="Tickers", description="The list of tickers most suitable as per the user's profile and preferences.")
    
class NewsItem(BaseModel):
    headline: str = Field(..., description="News headline")
    source: str = Field(..., description="Source of the article")
    url: str = Field(..., description="URL to the full news article")
    datetime: Optional[str] = Field(None, description="Publish datetime (ISO or timestamp)")
    
class MarketSnapshot(BaseModel):
    ticker: str = Field(..., title="Ticker Symbol", description="The ticker symbol of the company.")
    current_price: float = Field(..., gt=0, description="Current market price.")
    previous_close: float = Field(..., gt=0, description="Previous close price.")
    open_price: float = Field(..., gt=0, description="Today's open price.")
    day_high: float = Field(..., gt=0, description="Today's high price.")
    day_low: float = Field(..., gt=0, description="Today's low price.")
    fifty_two_week_low: Optional[float] = Field(default=None, gt=0, description="52-week low price.")
    fifty_two_week_high: Optional[float] = Field(default=None, gt=0, description="52-week high price.")
    fifty_day_avg: Optional[float] = Field(default=None, gt=0, description="50-day moving average.")
    two_hundred_day_avg: Optional[float] = Field(default=None, gt=0, description="200-day moving average.")
    
    beta: Optional[float] = Field(default=None, description="Beta of the stock vs market.")
    alpha: Optional[float] = Field(default=None, description="Alpha based on CAPM model.")
    volatility: Optional[float] = Field(default=None, gt=0, description="Annualized volatility.")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio.")
    sortino_ratio: Optional[float] = Field(default=None, description="Sortino ratio.")
    calmar_ratio: Optional[float] = Field(default=None, description="Calmar ratio.")
    tail_ratio: Optional[float] = Field(default=None, description="Tail ratio.")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown.")
    downside_deviation: Optional[float] = Field(default=None, description="Downside deviation.")
    omega_ratio: Optional[float] = Field(default=None, description="Omega ratio.")
    skewness: Optional[float] = Field(default=None, description="Skewness.")
    kurtosis: Optional[float] = Field(default=None, description="Kurtosis.")
    information_ratio: Optional[float] = Field(default=None, description="Information ratio.")
    treynor_ratio: Optional[float] = Field(default=None, description="Treynor ratio.")
    value_at_risk_95: Optional[float] = Field(default=None, description="Value at Risk (95% confidence).")
    
    market_cap: Optional[float] = Field(default=None, gt=0, description="Market capitalization.")
    enterprise_value: Optional[float] = Field(default=None, gt=0, description="Enterprise value.")
    pe_ratio: Optional[float] = Field(default=None, gt=0, description="Trailing P/E ratio.")
    forward_pe: Optional[float] = Field(default=None, description="Forward P/E ratio.")
    peg_ratio: Optional[float] = Field(default=None, gt=0, description="PEG ratio.")
    price_to_book: Optional[float] = Field(default=None, description="Price-to-book ratio.")
    price_to_sales: Optional[float] = Field(default=None, gt=0, description="Price-to-sales ratio.")
    dividend_yield: Optional[float] = Field(default=None, ge=0, description="Dividend yield.")
    payout_ratio: Optional[float] = Field(default=None, description="Payout ratio.")
    
    return_on_assets: Optional[float] = Field(default=None, description="ROA (%)")
    return_on_equity: Optional[float] = Field(default=None, description="ROE (%)")
    revenue_growth: Optional[float] = Field(default=None, description="Revenue growth (%)")
    earnings_growth: Optional[float] = Field(default=None, description="Earnings growth (%)")
    free_cashflow: Optional[float] = Field(default=None, description="Free cashflow")
    operating_cashflow: Optional[float] = Field(default=None, description="Operating cashflow")
    gross_profits: Optional[float] = Field(default=None, description="Gross profits")
    ebitda: Optional[float] = Field(default=None, description="EBITDA")
    total_debt: Optional[float] = Field(default=None, description="Total debt")
    current_ratio: Optional[float] = Field(default=None, description="Current ratio")
    quick_ratio: Optional[float] = Field(default=None, description="Quick ratio")
    
    # Leverage & financing
    risk_free_rate: Optional[float] = Field(default=0.03, description="Risk-free rate")
    borrowing_rate: Optional[float] = Field(default=0.04, description="Effective borrowing rate")
    
    unlevered_expected_return: Optional[float] = Field(default=None, description="Unlevered expected return")
    unlevered_volatility: Optional[float] = Field(default=None, description="Unlevered volatility")
    levered_expected_return: Optional[float] = Field(default=None, description="Levered expected return")
    levered_volatility: Optional[float] = Field(default=None, description="Levered volatility")
    
    leverage_multiplier: Optional[float] = Field(default=1.0, ge=1.0, description="Leverage multiplier")
    
    historical_prices: Optional[pd.DataFrame] = Field(default=None, description="Historical OHLC prices.")
    daily_returns: Optional[pd.Series] = Field(default=None, description="Daily returns.")
    correlations: Optional[Dict[str, float]] = Field(default=None, description="Pairwise correlations with other tickers.")
    
    # Black-Scholes option metrics
    implied_volatility: Optional[float] = Field(default=None, description="Implied volatility for options.")
    option_price_call: Optional[float] = Field(default=None, description="Black-Scholes call option price.")
    option_price_put: Optional[float] = Field(default=None, description="Black-Scholes put option price.")
    
    latest_news: Optional[List[NewsItem]] = Field(default=None, description="Latest news about the company and market.")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
class MarketDataInference(BaseModel):
    ticker: str = Field(..., title="Ticker", description="The ticker symbol of the stock.")
    price_trend_insights: Optional[List[str]] = Field(default_factory=list, title="Price Trend Insights", description="Insights on the price trend of the stock.")
    volatility_and_risk_insights: Optional[List[str]] = Field(default_factory=list, title="Volatility and Risk Insights", description="Insights on the volatility and risk of the stock.")
    fundamental_and_valuation_insights: Optional[List[str]] = Field(default_factory=list, title="Fundamental and Valuation Insights", description="Insights on the fundamentals and valuations of the stock.")
    correlation_insights: Optional[List[str]] = Field(default_factory=list, title="Correlation Insights", description="Insights on the correlation of the stock with other stocks to assess market positioning, volatility, and risk.")
    news_sentiment_insights: Optional[List[str]] = Field(default_factory=list, title="News Sentiment Insights", description="Insights on the news sentiment and market impact of the stock.")
    actionable_recommendations: Optional[List[str]] = Field(default_factory=list, title="Actionable Recommendations", description="Recommendations on the stock based on insights and analysis.")
    
class PortfolioState(BaseModel):
    holdings: Dict[str, float] = Field(default_factory=dict, title="Holdings", description="The holdings of the portfolio.")
    cash: float = Field(..., title="Cash", ge=0, description="The amount of cash in the portfolio.")
    total_value: float = Field(..., title="Total Value", ge=0, description="The total value of the portfolio.")
    expected_return: float = Field(..., title="Expected Return", description="The expected return of the portfolio.")
    risk: Optional[float] = Field(default=None, title="Risk", description="The risk associated with the portfolio.")
    
class OptimizationResult(BaseModel):
    new_holdings: Dict[str, float] = Field(default_factory=dict, title="New Holdings", description="The new holdings of the portfolio.")
    expected_return: float = Field(..., title="Expected Return", description="The expected return of the portfolio.")
    portfolio_risk: float = Field(..., title="Portfolio Risk", description="The risk associated with the portfolio.")
    leverage_multiplier: float = Field(default=1.0, title="Leverage Multiplier", description="The leverage multiplier applied to the portfolio.")

class AgentState(BaseModel):
    user_description: str = Field(default="", title="User Description", description="A natural language description of the user's investment goals.")
    user_profile: UserProfile = Field(default=None, title="User Profile", description="The user profile of the agent.")
    relevant_tickers: TickerSelection = Field(default=None, title="Relevant Tickers", description="The most relevant tickets as per the user profile.")
    market_data: List[MarketSnapshot] = Field(default_factory=list, title="Market Data", description="The market data of the agent.")
    portfolio_state: PortfolioState = Field(default=None, title="Portfolio State", description="The portfolio state of the agent.")
    optimization_result: Optional[OptimizationResult] = Field(default=None, title="Optimization Result", description="The optimization result of the agent.")
    portfolio_assessment: Optional[Dict[str, Any]] = Field(default_factory=dict, title="Portfolio Assessment", description="A fair and unbiased assessment of the portfolio.")
    portfolio_action: Optional[Dict[str, Any]] = Field(default_factory=dict, title="Portfolio Action", description="The action performed on the portfolio for optimization.")
    portfolio_status: Literal["stable", "unstable"] = Field(default="unstable", title="Portfolio Status", description="The status of the portfolio whether it is stable or unstable.")
    recommendations: Optional[Dict[str, Any]] = Field(default_factory=dict, title="Recommendations", description="The recommendations of the agent.")
    market_insights: Optional[List[MarketDataInference]] = Field(default_factory=list, title="Market Insights", description="List of market insights for different stock options.")
    num_iterations: Optional[int] = Field(default=0, title="Number of Iterations", description="The count of iterations performed for portfolio optimization.")
    
# Initialize structured LLMs
llm_with_user_profile = llm.with_structured_output(UserProfile)
llm_with_tickers = llm.with_structured_output(TickerSelection)
llm_with_market_data_inference = llm.with_structured_output(MarketDataInference)

def fetch_market_data(state: AgentState) -> dict:
    """   
        Fetches historical market data for a list of tickers using yfinance and Finnhub APIs.
        Calculates historical returns, volatility, correlations, and latest news for each ticker.
    """
    tickers = state.relevant_tickers.tickers
    results = []
    all_hist = dict()
    
    # Fetch historical stock prices
    for symbol in tickers:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1y")
        
        if history.empty:
            continue
        
        all_hist[symbol] = history
        
        # Calculate returns and volatility
        daily_returns = history["Close"].pct_change().dropna()
        
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else None
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else None
        sortino = (daily_returns.mean() * 252) / (daily_returns[daily_returns<0].std() * np.sqrt(252)) if daily_returns[daily_returns<0].std() != 0 else None
        var95 = np.percentile(daily_returns, 5)  # 5% quantile
        
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        calmar_ratio = (daily_returns.mean() * 252) / abs(max_drawdown) if max_drawdown else None
        
        tail_ratio = abs(
            np.percentile(daily_returns, 95) / np.percentile(daily_returns, 5)
        ) if np.percentile(daily_returns, 5) != 0 else None
        
        omega_ratio = (
            daily_returns[daily_returns > 0].sum() /
            abs(daily_returns[daily_returns < 0].sum())
        ) if abs(daily_returns[daily_returns < 0].sum()) > 0 else None
        
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        
        risk_free_rate = 0.03
        market_return = 0.08 # market proxy
        
        beta = info.get("beta", 1.0)
        unlevered_expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        # Adjust expected return and volatility based on risk appetite
        risk_target = {
            "conservative": 0.12,
            "moderate": 0.18,
            "aggressive": 0.30
        }
        
        target_vol = RISK_BUDGETS.get(state.user_profile.risk_appetite, 0.25)
        leverage_multiplier = target_vol / max(annual_vol, 0.01)
        leverage_multiplier = np.clip(leverage_multiplier, 1.0, 4.0)
        
        levered_return = leverage_multiplier * unlevered_expected_return
        levered_volatility = leverage_multiplier * annual_vol
        
        treynor_ratio = ((daily_returns.mean() * 252) - 0.03) / beta if beta != 0 else None
        
        benchmark_return = 0.08 # market proxy
        
        information_ratio = (
            (daily_returns.mean() * 252 - benchmark_return) / annual_vol
        ) if annual_vol else None
        
        # Correlations with other tickers
        corr_dict = {}
        for other in tickers:
            if other in all_hist and symbol != other:
                corr_dict[other] = daily_returns.corr(all_hist[other]["Close"].pct_change().dropna())
        
        # Gather latest company and market news using FinnHub API
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
                # Company‑specific news for the ticker
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
            except Exception:
                combined_news = []
    
        snapshot = MarketSnapshot(
            ticker=symbol,
            current_price=info.get("currentPrice", history["Close"].iloc[-1]),
            previous_close=info.get("previousClose", history["Close"].iloc[-2]),
            open_price=info.get("open", history["Open"].iloc[-1]),
            day_high=info.get("dayHigh", history["High"].iloc[-1]),
            day_low=info.get("dayLow", history["Low"].iloc[-1]),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_day_avg=info.get("fiftyDayAverage"),
            two_hundred_day_avg=info.get("twoHundredDayAverage"),
            beta=info.get("beta"),
            volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            value_at_risk_95=var95,
            calmar_ratio=calmar_ratio,
            tail_ratio=tail_ratio,
            omega_ratio=omega_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            max_drawdown=max_drawdown,
            downside_deviation=downside_deviation,
            market_cap=info.get("marketCap"),
            enterprise_value=info.get("enterpriseValue"),
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            peg_ratio=info.get("pegRatio"),
            price_to_book=info.get("priceToBook"),
            price_to_sales=info.get("priceToSales"),
            dividend_yield=info.get("dividendYield") / 100.0 if info.get("dividendYield") and info.get("dividendYield") > 0.5 else info.get("dividendYield"),
            payout_ratio=info.get("payoutRatio"),
            return_on_assets=info.get("returnOnAssets"),
            return_on_equity=info.get("returnOnEquity"),
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsQuarterlyGrowth"),
            free_cashflow=info.get("freeCashflow"),
            operating_cashflow=info.get("operatingCashflow"),
            gross_profits=info.get("grossProfits"),
            ebitda=info.get("ebitda"),
            total_debt=info.get("totalDebt"),
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            risk_free_rate=risk_free_rate,
            borrowing_rate=0.04,
            unlevered_expected_return=unlevered_expected_return,
            unlevered_volatility=annual_vol,
            levered_expected_return=levered_return,
            levered_volatility=levered_volatility,
            leverage_multiplier=leverage_multiplier,
            historical_prices=history,
            daily_returns=daily_returns,
            correlations=corr_dict,
            latest_news=combined_news 
        )
        
        results.append(snapshot)
    
    return {"market_data": results}
        
def infer_user_profile(state: AgentState) -> dict:
    """   
        Makes inference about a user's profile and risk appetite based on the agent's state.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a financial investment advisor agent. Your task is to infer the user's profile and risk appetite based on the agent's state. Respond with a JSON object containing the user profile and risk appetite."),
        HumanMessagePromptTemplate.from_template(
            """
            User data: capital={capital}, preferences={preferences}, constraints={constraints}
            Suggest risk appetite (conservative/moderate/aggressive) and horizon in years.
            Return JSON: {{"risk_appetite": "...", "investment_horizon": ...}}
            """
        )
    ])
    
    formatted_prompt = prompt.format(
        capital=state.user_profile.capital,
        preferences=", ".join(state.user_profile.preferences) if state.user_profile.preferences else "None",
        constraints=", ".join(state.user_profile.constraints) if state.user_profile.constraints else "None"
    )
    
    inferred_data = llm_with_user_profile.invoke(formatted_prompt)
    
    # Create new profile starting from the state's profile to preserve all fields
    new_profile = state.user_profile.model_copy()
    
    # Only overwrite if the state's values are default (e.g. "moderate" or None)
    # or if you want the AI to always have the final say based on the description
    # For now, let's only set if them if they are missing or if the description is provided
    if state.user_description:
        new_profile.risk_appetite = inferred_data.risk_appetite
        new_profile.investment_horizon = inferred_data.investment_horizon
    
    return {"user_profile": new_profile}

def select_relevant_tickers(state: AgentState) -> dict:
    """
        Selects a list of tickers based on user preferences, constraints, and investment horizon.
        Uses LLM reasoning to prioritize sectors, stocks, and ETFs that match user risk profile.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a financial AI investment advisory agent. Your task is to selecting relevant stock tickers based on user profile."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            User Profile:
            - Risk Appetite: {risk_appetite}
            - Investment Horizon (years): {investment_horizon}
            - Preferences: {preferences}
            - Constraints: {constraints}

            Task:
            Generate a JSON list of 5-10 stock tickers that align with the user's risk profile and preferences.
            Consider diversification across sectors and risk exposure.
            Return JSON list: ["TICKER1", "TICKER2", ...]
            """
        )
    ])
    
    formatted = prompt.format(
        risk_appetite=state.user_profile.risk_appetite,
        investment_horizon=state.user_profile.investment_horizon,
        preferences=", ".join(state.user_profile.preferences) if state.user_profile.preferences else "none",
        constraints=", ".join(state.user_profile.constraints) if state.user_profile.constraints else "none"
    )
    
    selected_tickers = llm_with_tickers.invoke(formatted)
    
    if selected_tickers.tickers:
        tickers = selected_tickers.tickers
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # fallback
    
    return {"relevant_tickers": TickerSelection(tickers=tickers)}

def analyze_market_patterns(state: AgentState) -> dict:
    """   
        Analyzes market patterns based on the agent's state.
    """
    insights = []

    for ms in state.market_data:
        if ms.historical_prices is None or ms.daily_returns is None:
            continue

        # Compute quantitative indicators
        prices = ms.historical_prices["Close"]
        returns = ms.daily_returns

        # Moving averages
        ma_20 = prices.rolling(window=20).mean().iloc[-1]
        ma_50 = prices.rolling(window=50).mean().iloc[-1]
        ma_200 = prices.rolling(window=200).mean().iloc[-1]

        # Trend signal
        if prices.iloc[-1] > ma_50 and ma_50 > ma_200:
            trend_signal = "Strong uptrend"
        elif prices.iloc[-1] < ma_50 and ma_50 < ma_200:
            trend_signal = "Strong downtrend"
        else:
            trend_signal = "Sideways / mixed trend"

        # Volatility spike detection
        recent_vol = returns[-20:].std() * np.sqrt(252)
        vol_spike = "High" if recent_vol > ms.volatility else "Normal"

        # Momentum (simple: 20-day return)
        momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100

        # Correlation summary
        corr_str = ", ".join([f"{k}: {v:.2f}" for k, v in (ms.correlations or {}).items()])

        # Aggregate news
        news_text = " ".join([n.headline for n in (ms.latest_news or [])])

        prompt = f"""
        You are a financial AI analyst.

        Ticker: {ms.ticker}

        Price/Trend Metrics:
        - Latest Close: {prices.iloc[-1]:.2f}, 20-day MA: {ma_20:.2f}, 50-day MA: {ma_50:.2f}, 200-day MA: {ma_200:.2f}
        - Trend: {trend_signal}, Momentum (20-day %): {momentum:.2f}%
        - Recent Volatility: {recent_vol:.2f}, Volatility Spike: {vol_spike}

        Risk Metrics:
        - Beta: {ms.beta}, Sharpe: {ms.sharpe_ratio}, Sortino: {ms.sortino_ratio}, VaR95: {ms.value_at_risk_95}

        Fundamental & Valuation:
        - P/E: {ms.pe_ratio}, Forward PE: {ms.forward_pe}, PEG: {ms.peg_ratio}, Dividend Yield: {ms.dividend_yield}
        - ROA: {ms.return_on_assets}, ROE: {ms.return_on_equity}, Revenue Growth: {ms.revenue_growth}

        Correlations: {corr_str}

        Option Metrics:
        - Implied Volatility: {ms.implied_volatility}, Call: {ms.option_price_call}, Put: {ms.option_price_put}

        News Headlines: {news_text}

        TASK:
        1. Provide a short analysis of price trend and momentum.
        2. Assess risk exposure and volatility situation.
        3. Comment on fundamentals and valuation.
        4. Evaluate correlation exposures.
        5. Suggest 2-3 actionable recommendations for this ticker.

        Return JSON:
        {{
            "ticker": "{ms.ticker}",
            "trend_insight": "...",
            "risk_insight": "...",
            "fundamental_insight": "...",
            "correlation_insight": "...",
            "news_insight": "...",
            "recommendations": ["...", "...", "..."]
        }}
        """

        try:
            market_insights = llm_with_market_data_inference.invoke(prompt)
        except json.JSONDecodeError:
            # fallback
            market_insights = {
                "ticker": ms.ticker,
                "trend_insight": trend_signal,
                "risk_insight": f"Volatility: {vol_spike}",
                "fundamental_insight": "Fundamental metrics available.",
                "correlation_insight": corr_str or "No correlations",
                "news_insight": "News headlines analyzed.",
                "recommendations": ["Monitor trend and volatility.", "Diversify exposure."]
            }

        insights.append(MarketDataInference(**market_insights.model_dump()))

    return {"market_insights": insights}

def simulate_portfolio(state: AgentState) -> dict:
    """
        Advanced portfolio simulation:
        - Uses historical returns, CAPM, Black-Litterman adjusted returns
        - Integrates factor models (Fama-French 3-factor simplified)
        - Considers optionality effects via Black-Scholes call/put prices
        - Performs Monte Carlo simulations to estimate expected return, risk, and VaR
    """
    holdings = state.portfolio_state.holdings
    total_capital = state.portfolio_state.cash + sum(
        [ms.current_price * holdings[ms.ticker] for ms in state.market_data if ms.ticker in holdings]
    )

    # Prepare historical returns matrix
    returns_matrix = []
    tickers = []
    
    for ms in state.market_data:
        if ms.historical_prices is not None and ms.ticker in holdings:
            daily_returns = ms.daily_returns
            if daily_returns is not None and len(daily_returns) > 1:
                returns_matrix.append(daily_returns.values)
                tickers.append(ms.ticker)

    returns_dict = {ms.ticker: ms.daily_returns for ms in state.market_data if ms.daily_returns is not None}
    if not returns_dict:
        # Fallback if no data
        return PortfolioState(
            holdings=holdings,
            cash=state.portfolio_state.cash,
            total_value=total_capital,
            expected_return=0.08,
            risk=0.15
        )

    returns_df = pd.DataFrame(returns_dict).dropna()
    if returns_df.empty:
        # Fallback if no overlapping data
        return PortfolioState(
            holdings=holdings,
            cash=state.portfolio_state.cash,
            total_value=total_capital,
            expected_return=0.08,
            risk=0.15
        )

    # Final tickers that have valid overlapping data
    final_tickers = returns_df.columns.tolist()
    mean_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252

    # CAPM expected returns
    market_return = 0.08
    capm_returns = []
    
    for ticker, mu in zip(final_tickers, mean_returns):
        ms = next(m for m in state.market_data if m.ticker == ticker)
        beta = ms.beta if ms.beta else 1.0
        capm_r = 0.03 + beta * (market_return - 0.03)
        capm_returns.append(capm_r)
        
    capm_returns = np.array(capm_returns)

    # Black-Litterman adjusted returns
    bl_adjusted_returns = mean_returns * 0.6 + capm_returns * 0.4

    # Factor Model Adjustment (simplified Fama-French 3-factor)
    factor_exposures = np.array([0.3, 0.2, 0.1])  # Market, SMB, HML (example weights)
    factor_returns = np.array([0.08, 0.02, 0.015])  # annualized returns
    factor_adjustment = factor_exposures.dot(factor_returns)
    bl_adjusted_returns += factor_adjustment

    # Monte Carlo Simulation
    n_simulations = 5000
    # Values based on final_tickers only
    values = np.array([holdings.get(t, 0) * next(ms.current_price for ms in state.market_data if ms.ticker == t) for t in final_tickers])
    
    # Avoid div by zero
    total_val = values.sum()
    portfolio_weights = values / total_val if total_val > 0 else np.ones(len(final_tickers)) / len(final_tickers)
    simulated_returns = []
    
    for _ in range(n_simulations):
        sample = np.random.multivariate_normal(bl_adjusted_returns / 252, cov_matrix / 252)
        port_return = np.dot(sample, portfolio_weights)
        simulated_returns.append(port_return)
        
    simulated_returns = np.array(simulated_returns)

    expected_return = np.mean(simulated_returns) * 252 # Annualized return
    portfolio_risk = np.std(simulated_returns) * np.sqrt(252) # Annualized volatility
    
    target_vol = RISK_BUDGETS.get(state.user_profile.risk_appetite, 0.25)
    portfolio_leverage = target_vol / max(portfolio_risk, 0.01) if state.user_profile.use_leverage else 1.0
    portfolio_leverage = np.clip(portfolio_leverage, 1.0, 4.0)
    
    expected_return *= portfolio_leverage
    portfolio_risk *= portfolio_leverage
    
    var_95 = np.percentile(simulated_returns, 5) * np.sqrt(252) # Annualized 95% VaR

    # Black-Scholes effect (call options as partial hedge)
    for ms in state.market_data:
        if ms.option_price_call and ms.volatility:
            S = ms.current_price
            K = S  # At-the-money
            T = state.user_profile.investment_horizon
            r = 0.03  # risk-free rate
            sigma = ms.volatility
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            # Reduce effective risk by optionality hedge
            portfolio_risk *= max(0.85, 1 - 0.5 * (call_price / S))

    updated_portfolio = PortfolioState(
        holdings=holdings,
        cash=state.portfolio_state.cash,
        total_value=total_capital * portfolio_leverage,
        expected_return=float(expected_return),
        risk=float(portfolio_risk)
    )
    
    new_recommendations = {
        **state.recommendations,
        "Portfolio 95% VaR": f"{var_95:.4f}",
        "Expected Return": f"{expected_return:.4f}",
        "Portfolio Risk": f"{portfolio_risk:.4f}"
    }

    return {"portfolio_state": updated_portfolio, "recommendations": new_recommendations}

def optimize_portfolio(state: AgentState) -> dict:
    """
        Optimizes portfolio for risk-adjusted return using historical market data,
        correlations, Sharpe/Sortino ratios, and user-defined constraints/preferences.
        Allocations are scaled by user capital and risk tolerance is respected.
    """
    tickers = [ms.ticker for ms in state.market_data]
    capital = state.user_profile.capital
    risk_level = state.user_profile.risk_appetite
    max_risk = RISK_BUDGETS.get(risk_level, 0.25)

    target_vol = RISK_BUDGETS.get(risk_level, 0.25)
    portfolio_leverage = 1.0 # Will be calculated after optimization to reach target_vol
    
    # Compute expected returns and covariance matrix
    returns_dict = {}
    for ms in state.market_data:
        if ms.daily_returns is not None:
            returns_dict[ms.ticker] = ms.daily_returns
        else:
            returns_dict[ms.ticker] = pd.Series(np.zeros(len(ms.historical_prices)))  # fallback

    returns_df = pd.DataFrame(returns_dict).dropna()
    mean_returns = returns_df.mean() * 252  # annualized return
    cov_matrix = returns_df.cov() * 252     # annualized covariance

    num_assets = len(tickers)
    
    if num_assets == 0:
        fallback_tickers = ["GOOG", "AAPL", "MSFT"]
        num_assets = len(fallback_tickers)

    # Define optimization variables
    w = cp.Variable(num_assets)
    
    # Portfolio metrics in CVXPY terms
    portfolio_return = mean_returns.values @ w
    portfolio_variance = cp.quad_form(w, cov_matrix.values)

    if state.user_profile.use_leverage:
        # STRATEGY 1: Maximize Sharpe Ratio (Tangency Portfolio)
        # We solve this by minimizing variance subject to a fixed excess return
        # and then normalizing, which is equivalent to maximizing the Sharpe Ratio.
        # We'll use a standard risk-free rate of 4%
        rf_rate = 0.04
        
        # To avoid fractional programming, we minimize variance subject to (mu - rf) * w = 1
        # Then we will normalize the weights.
        w_sharp = cp.Variable(num_assets)
        excess_returns = (mean_returns.values - rf_rate)
        
        # Only solve if there's at least one positive excess return
        if np.any(excess_returns > 0):
            # Constraints: (mu - rf) * w = 1 and w >= 0
            constraints_sharp = [excess_returns @ w_sharp == 1, w_sharp >= 0]
            
            # Max allocation per asset if preferences exist
            if state.user_profile.preferences:
                # This is tricky in the transformed space, we'll keep it simple for Tangency
                pass

            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w_sharp, cov_matrix.values)),
                constraints_sharp
            )
            prob.solve(solver=cp.SCS)
            
            if w_sharp.value is not None:
                # Normalize weights to sum to 1
                weights = w_sharp.value / np.sum(w_sharp.value)
            else:
                # Fallback to equal weight if solver fails
                weights = np.ones(num_assets) / num_assets
        else:
            weights = np.ones(num_assets) / num_assets

        # Stable Leverage Calculation based on Risk Appetite limits
        # Caps: Conservative: 1.5x, Moderate: 2.0x, Aggressive: 2.5x
        LEVERAGE_CAPS = {
            "conservative": 1.5,
            "moderate": 2.0,
            "aggressive": 2.5
        }
        
        # Calculate Unlevered Risk of the Tangency Portfolio
        unlevered_risk = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
        
        # Scale to hit the target_vol, but respect the profile cap
        profile_cap = LEVERAGE_CAPS.get(risk_level, 2.0)
        target_leverage = target_vol / max(unlevered_risk, 0.01)
        
        # Ensure leverage is at least 1.1x if requested, but not above the cap
        portfolio_leverage = np.clip(target_leverage, 1.1, profile_cap)
        
    else:
        # STRATEGY 2: Standard Markowitz (Max Return s.t. Risk Budget)
        # Used when leverage is NOT allowed
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            portfolio_variance <= max_risk**2
        ]
        
        # Max allocation per asset for preferred tickers
        if state.user_profile.preferences:
            max_alloc = 0.5
            for i, ticker in enumerate(tickers):
                if ticker in state.user_profile.preferences:
                    constraints.append(w[i] <= max_alloc)

        prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
        prob.solve(solver=cp.SCS)
        
        if w.value is not None:
            weights = np.clip(w.value, 0, 1)
            weights /= weights.sum()
        else:
            weights = np.ones(num_assets) / num_assets
            
        unlevered_risk = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
        portfolio_leverage = 1.0

    # Compute Final Levered metrics
    scaled_holdings = {tickers[i]: weights[i] * capital * portfolio_leverage for i in range(num_assets)}
    portfolio_expected_return = float(mean_returns.values @ weights) * portfolio_leverage
    portfolio_risk = unlevered_risk * portfolio_leverage

    optimization_result = OptimizationResult(
        new_holdings=scaled_holdings,
        expected_return=portfolio_expected_return,
        portfolio_risk=portfolio_risk,
        leverage_multiplier=portfolio_leverage
    )
    
    updated_portfolio_state = PortfolioState(
        holdings=scaled_holdings,  
        cash=state.portfolio_state.cash, 
        total_value=state.user_profile.capital * portfolio_leverage,  
        expected_return=portfolio_expected_return, 
        risk=portfolio_risk  
    )

    return {"optimization_result": optimization_result, "portfolio_state": updated_portfolio_state}
    
def assess_portfolio(state: AgentState) -> dict:
    """
        Robustly evaluates whether the portfolio should be rebalanced, hedged, or held,
        using risk appetite, portfolio risk, asset correlations, volatility, market news, and performance ratios.
    """ 
    # Extract relevant metrics
    risk_appetite = state.user_profile.risk_appetite
    portfolio = state.portfolio_state
    optimization = state.optimization_result
    
    # Default action
    action = "hold"
    explanations = []

    action_scores = {"hold": 0, "rebalance": 0, "hedge": 0}
    correlations = []
    high_corr_count = 0
    volatilities = []
    vars_95 = []
    negative_news_count = 0
    
    # Risk targets for reference
    target_risk = RISK_BUDGETS.get(risk_appetite, 0.25)
    
    # Assess overall portfolio risk vs target
    if optimization:
        current_risk = optimization.portfolio_risk
        
        # Scoring logic based on deviations from target_risk
        risk_deviation = (current_risk - target_risk) / target_risk
        
        if abs(risk_deviation) < 0.15: # Within 15% of target
            action_scores["hold"] += 2
        elif risk_deviation > 0.2: # Significantly over risk budget
            action_scores["rebalance"] += 2
            explanations.append(f"Portfolio risk ({current_risk:.2f}) is significantly above target ({target_risk:.2f}).")
        
        # Check for high correlation between assets
        correlations = []
        for ms in state.market_data:
            if ms.correlations:
                correlations.extend([v for v in ms.correlations.values()])
        
        high_corr_count = sum(1 for c in correlations if abs(c) > 0.8)
        if high_corr_count > len(correlations) * 0.4:
            action_scores["rebalance"] += 1
            explanations.append("High asset correlations detected. Suggesting rebalance to improve diversification.")

        # Tail Risk Assessment (VaR)
        vars_95 = [ms.value_at_risk_95 for ms in state.market_data if ms.value_at_risk_95 is not None]
        avg_var = np.mean(vars_95) if vars_95 else 0
        
        # Risk-appetite sensitive VaR thresholds
        var_threshold = -0.10 if risk_appetite == "conservative" else -0.20
        if avg_var < var_threshold:
            action_scores["hedge"] += 1
            explanations.append(f"Significant tail risk detected (Avg VaR {avg_var:.2f}). Hedging recommended.")

        # Volatility check
        volatilities = [ms.volatility for ms in state.market_data if ms.volatility is not None]

        # Negative market news (impact > 50% of portfolio)
        negative_news_count = 0
        for ms in state.market_data:
            if ms.latest_news:
                for news in ms.latest_news:
                    if any(kw in news.headline.lower() for kw in ["crash", "scandal", "lawsuit", "bankruptcy"]):
                        negative_news_count += 1
                        break
        
        if negative_news_count >= max(1, len(state.market_data) // 2):
            action_scores["hedge"] += 2
            explanations.append(f"Broad negative news sentiment detected ({negative_news_count} assets). Hedging suggested.")

    else:
        # Fallback if no optimization
        action_scores["hold"] += 1
        explanations.append("Unable to compute optimized risk. Holding current state.")

    # Final decision with a slight bias towards 'hold' to prevent over-trading
    action_scores["hold"] += 0.5 
    action = max(action_scores, key=lambda k: action_scores[k])
        
    llm_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert fintech investment strategy advisor providing professional portfolio recommendations."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Portfolio Analysis Context:
            - Action suggested by rules: {action}
            - Risk Appetite: {risk_appetite}
            - Portfolio Risk: {portfolio_risk}
            - Optimization Risk: {optimization_risk}
            - Correlations: {high_corr_count} highly correlated pairs
            - Volatility Max: {max_volatility}
            - VaR 95% Average: {var95_avg}
            - Negative News Count: {negative_news_count}

            Insights from earlier evaluation:
            {explanations}

            Task: Provide a concise professional explanation for why this action ({action}) is recommended, 
            including the reasoning and key risk factors considered.
            """
        )
    ])
    
    formatted_prompt = llm_prompt.format(
        action=action,
        risk_appetite=risk_appetite,
        portfolio_risk=f"{portfolio.risk:.2f}" if portfolio.risk is not None else "0.00",
        optimization_risk=f"{optimization.portfolio_risk:.2f}" if optimization else "0.00",
        high_corr_count=high_corr_count,
        max_volatility=f"{max(volatilities):.2f}" if volatilities else "0.00",
        var95_avg=f"{np.mean(vars_95):.2f}" if vars_95 else "0.00",
        negative_news_count=negative_news_count,
        explanations="\n".join(explanations)
    )

    response = llm.invoke(formatted_prompt).content
    final_output = {
        "action": action,
        "explanation": response
    }
    return {"portfolio_assessment": final_output}

def calculate_total_value(holdings, cash, price_map):
    """   
        Calculates the total value of the portfolio based on holdings and cash.
    """
    total = cash
    for ticker, qty in holdings.items():
        total += qty * price_map.get(ticker, 0)
    return total
def execute_portfolio_action(state: AgentState) -> dict:
    """
        Executes portfolio actions (rebalance/hedge/hold) realistically using market data.

        - Rebalance: Apply optimized allocations using current market prices.
        - Hedge: Apply derivative-based hedge using option prices and volatility.
        - Hold: Maintain current holdings, log metrics.
    """
    action = state.portfolio_assessment["action"]
    
    if not state.market_data:
        return {"portfolio_action": {"action": "error", "explanation": "Error: No market data available to execute portfolio actions."}}

    # Map ticker to latest price for portfolio valuation
    price_map = {md.ticker: md.current_price for md in state.market_data}

    total_portfolio_value = calculate_total_value(state.portfolio_state.holdings, state.portfolio_state.cash, price_map)

    # Rebalance
    if action.lower().strip() == "rebalance":
        if not state.optimization_result or not state.optimization_result.new_holdings:
            return "Error: No optimization result available for rebalancing."

        new_holdings_value = state.optimization_result.new_holdings
        total_capital = state.portfolio_state.total_value
        updated_holdings = {}
        remaining_cash = total_capital

        # Convert allocation percentages to actual shares
        for ticker, allocation in new_holdings_value.items():
            price = price_map.get(ticker)
            
            if price and price > 0:
                qty = allocation / price
                updated_holdings[ticker] = round(qty, 4)
                remaining_cash -= qty * price

        updated_portfolio = PortfolioState(
            holdings=updated_holdings,
            cash=round(max(remaining_cash, 0), 2),
            total_value=calculate_total_value(updated_holdings, round(max(remaining_cash, 0), 2), price_map),
            expected_return=state.portfolio_state.expected_return,
            risk=state.portfolio_state.risk
        )
        
        rebalance_prompt = f"""
            You are a financial AI assistant. 
            The portfolio has been rebalanced based on market data and optimization results.
            Current holdings: {updated_holdings}
            Cash remaining: ${state.portfolio_state.cash}
            Total portfolio value: ${state.portfolio_state.total_value}
            Risk appetite: {state.user_profile.risk_appetite}
            Provide a concise summary highlighting the benefits of this rebalancing and recommendations for the user.
        """
        explanation = llm.invoke(rebalance_prompt).content
        return {"portfolio_state": updated_portfolio, "portfolio_action": {"action": "rebalance", "explanation": explanation}}

    # Hedge
    elif action.lower().strip() == "hedge":
        # Attempt a basic options hedge using implied volatility and option prices
        hedge_actions = []
        
        for md in state.market_data:
            if md.option_price_put and md.implied_volatility:
                hedge_qty = round((state.portfolio_state.total_value * 0.05) / md.option_price_put, 2)
                hedge_actions.append(f"Hedge {hedge_qty} PUT options for {md.ticker}")
        
        hedge_summary = hedge_actions if hedge_actions else ["No suitable derivatives found for hedging."]
        hedge_prompt = f"""
            You are a financial AI assistant. 
            The portfolio requires hedging based on risk exposure and market conditions.
            Proposed hedge actions: {hedge_summary}
            Current portfolio value: ${total_portfolio_value}
            Risk appetite: {state.user_profile.risk_appetite}
            Provide a clear, human-readable explanation of these hedge actions and their expected impact on portfolio risk.
        """
        explanation = llm.invoke(hedge_prompt).content
        return {"portfolio_action": {"action": "hedge", "explanation": explanation}}

    # Hold
    elif action.lower().strip() == "hold":
        # Evaluate risk vs expected return
        risk = state.portfolio_state.risk if state.portfolio_state.risk else 0.15
        expected_return = state.portfolio_state.expected_return
        holdings_summary = ", ".join([f"{k}: {v} shares" for k,v in state.portfolio_state.holdings.items()])
        
        hold_prompt = f"""
            You are a financial AI assistant.
            The portfolio is being held as is.
            Holdings: {holdings_summary}
            Cash: ${state.portfolio_state.cash}
            Total portfolio value: ${total_portfolio_value}
            Expected return: {expected_return*100:.2f}%
            Estimated risk: {risk*100:.2f}%
            Risk appetite: {state.user_profile.risk_appetite}
            Provide a detailed but concise explanation of why holding is appropriate and any considerations for the user.
        """
        explanation = llm.invoke(hold_prompt).content
        return {"portfolio_action": {"action": "hold", "explanation": explanation}}

    else:
        return {"portfolio_action": {"action": "invalid", "explanation": f"Invalid action '{action}'. Please use 'rebalance', 'hedge', or 'hold'."}}
    
def check_portfolio_stability(state: AgentState) -> dict:
    """
        Checks portfolio stability based on risk tolerance, news sentiment, and expected return.
        Uses weighted criteria to determine stability.

        Criteria weights sum to 1:
        - risk_weight: contribution of risk being within tolerance
        - news_weight: contribution of absence of negative news
        - return_weight: contribution of expected return convergence

        Returns a portfolio status update: {"portfolio_status": "stable" | "unstable"}
    """
    # Criteria weights
    return_tolerance, risk_weight, news_weight, return_weight = 0.005, 0.5, 0.3, 0.2

    # Risk tolerance
    risk_tolerance = {"conservative": 0.15, "moderate": 0.2, "aggressive": 0.3}
    max_risk = risk_tolerance.get(state.user_profile.risk_appetite, 0.2)

    # Risk value
    risk_value = state.portfolio_state.risk if state.portfolio_state.risk is not None else 0.0

    # Risk condition
    risk_score = 1.0 if risk_value <= max_risk else max(0, 1 - (risk_value - max_risk)/0.1)

    # Negative news condition
    negative_news_count = sum(
        1 for ms in state.market_data if ms.latest_news and any(
            kw in news.headline.lower() for news in ms.latest_news
            for kw in ["crash", "fraud", "lawsuit", "bankruptcy", "scandal"]
        )
    )
    news_score = 1.0 if negative_news_count == 0 else max(0, 1 - negative_news_count/5)

    # Expected return stability condition
    last_expected_return = None
    
    if state.recommendations and isinstance(state.recommendations, dict):
        last_expected_return = float(state.recommendations.get("Expected Return", None)) if "Expected Return" in state.recommendations else None
        
    delta_return = float("inf")
    
    if last_expected_return is not None:
        delta_return = abs(state.portfolio_state.expected_return - last_expected_return) if last_expected_return is not None else float("inf")
        
    return_score = 1.0 if delta_return < return_tolerance else max(0, 1 - delta_return / 0.05)

    # Weighted score
    weighted_score = (risk_weight * risk_score) + (news_weight * news_score) + (return_weight * return_score)

    # Stable if score >= 0.7
    portfolio_status =  "stable" if weighted_score >= 0.7 else "unstable"

    # Enforce max iterations
    current_iterations = state.num_iterations if state.num_iterations is not None else 0
    new_iterations = current_iterations + 1
    MAX_ITERATIONS = 3

    if new_iterations >= MAX_ITERATIONS:
        portfolio_status = "stable"

    return {"portfolio_status": portfolio_status, "num_iterations": new_iterations}

def portfolio_stability_router(state: AgentState) -> str:
    return str(state.portfolio_status).lower().strip() if state.portfolio_status else "unstable"
    
# Define state graph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("infer_user_profile", infer_user_profile)
graph.add_node("select_relevant_tickers", select_relevant_tickers)
graph.add_node("fetch_market_data", fetch_market_data)
graph.add_node("analyze_market_patterns", analyze_market_patterns)
graph.add_node("simulate_portfolio", simulate_portfolio)
graph.add_node("optimize_portfolio", optimize_portfolio)
graph.add_node("assess_portfolio", assess_portfolio)
graph.add_node("execute_portfolio_action", execute_portfolio_action)
graph.add_node("check_portfolio_stability", check_portfolio_stability)

# Add edges to the graph
graph.add_edge(START, "infer_user_profile")
graph.add_edge("infer_user_profile", "select_relevant_tickers")
graph.add_edge("select_relevant_tickers", "fetch_market_data")
graph.add_edge("fetch_market_data", "analyze_market_patterns")
graph.add_edge("analyze_market_patterns", "simulate_portfolio")
graph.add_edge("simulate_portfolio", "optimize_portfolio")
graph.add_edge("optimize_portfolio", "assess_portfolio")
graph.add_edge("assess_portfolio", "execute_portfolio_action")
# Simulate portfolio if portfolio is unstable and execute portfolio action - Feedback loop
graph.add_edge("execute_portfolio_action", "check_portfolio_stability")
graph.add_conditional_edges("check_portfolio_stability", portfolio_stability_router, {
    "unstable": "simulate_portfolio", # continue portfolio optimization
    "stable": END # terminate the workflow
})

# Compile the graph
workflow = graph.compile()

# Save the graph visualization
try:
    with open("portfolio_optimization_workflow.png", "wb") as f:
        f.write(workflow.get_graph().draw_mermaid_png())
    print("Workflow visualization saved to 'portfolio_optimization_workflow.png'")
except Exception as e:
    print(f"Could not save workflow visualization: {e}")