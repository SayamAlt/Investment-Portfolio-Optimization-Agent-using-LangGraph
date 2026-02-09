import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from portfolio_optimization_workflow import workflow, AgentState, UserProfile, PortfolioState, MarketSnapshot  
import pandas_ta_classic as ta
import warnings, os
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_community.cache import InMemoryCache

load_dotenv()

if "OPENAI_API_KEY" in st.secrets['secrets']:
    OPENAI_API_KEY = st.secrets['secrets']['OPENAI_API_KEY']
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

cache = InMemoryCache()

# Initialize the LLM
llm = OpenAI(temperature=0.6, api_key=OPENAI_API_KEY, cache=cache)

# Unified Risk Targets for Leverage & Optimization
RISK_BUDGETS = {
    "conservative": 0.15,
    "moderate": 0.25,
    "aggressive": 0.40
}

st.set_page_config(page_title="AI Portfolio Optimization", layout="wide")

st.title("📈 AI-Powered Investment Portfolio Optimizer")
st.markdown("""
This application leverages advanced financial AI to:
- Infer user risk profile
- Select relevant tickers
- Fetch market data
- Simulate and optimize portfolio
- Provide actionable recommendations
""")

# User Input
st.header("Define Your Investment Profile")

with st.form("user_profile_form"):
    capital = st.number_input("Available Capital ($)", min_value=1000.0, placeholder="Enter your available capital", step=1000.0)
    investment_horizon = st.slider("Investment Horizon (years)", min_value=1, max_value=50, value=5) 
    risk_appetite = st.selectbox("Risk Appetite", ["conservative", "moderate", "aggressive"])
    leverage_option = st.selectbox(
        "Use Leverage?", 
        ["No", "Yes"], 
        help="Select 'Yes' to allow leverage in your portfolio"
    )
    preferences = st.text_input("Preferences (comma-separated tickers/sectors)").split(",")
    preferences = [p.strip().upper() for p in preferences if p.strip()]
    constraints = st.text_input("Constraints (comma-separated, e.g., no tech stocks)").split(",")
    constraints = [c.strip() for c in constraints if c.strip()]
    submitted_profile = st.form_submit_button("Submit Profile")

if submitted_profile:
    st.success("Profile submitted successfully!")

    # Initialize AgentState
    agent_state = AgentState(
        user_profile=UserProfile(
            capital=capital,
            investment_horizon=investment_horizon,
            risk_appetite=risk_appetite,
            preferences=preferences,
            constraints=constraints,
            use_leverage=True if leverage_option == "Yes" else False
        ).model_dump(),
        portfolio_state=PortfolioState(
            holdings={}, 
            cash=capital, 
            total_value=capital, 
            expected_return=0.0
        ).model_dump()
    )

    st.session_state["agent_state"] = agent_state

def get_portfolio_weights(state: AgentState) -> dict:
    """
        Compute portfolio weights based on current holdings and current market prices.
        Returns weights as fractions of total portfolio value (sum to 1).
    """
    holdings = getattr(state.portfolio_state, "holdings", {}) or {}
    market_data = getattr(state, "market_data", [])

    if not holdings or not market_data:
        return {}

    # Map ticker to current price
    price_map = {md.ticker: md.current_price for md in market_data if md.current_price is not None}

    # Compute total portfolio value
    total_value = state.portfolio_state.cash
    
    for ticker, qty in holdings.items():
        price = price_map.get(ticker)
        
        if price is not None:
            total_value += qty * price
        else:
            # Skip tickers with missing market data
            continue

    if total_value == 0 or not holdings:
        return {}

    # Compute weights
    weights = {}
    
    for ticker, qty in holdings.items():
        price = price_map.get(ticker)
        
        if price is None:
            weights[ticker] = 0.0
        else:
            weights[ticker] = (qty * price) / total_value

    return weights

def aggregate_portfolio_metrics(state: AgentState) -> dict:
    """
        Computes portfolio-level metrics by aggregating all relevant market data variables.
        Weighted metrics (like beta, P/E, dividend yield) are weighted by current value allocation.
        Returns averages for metrics not dependent on portfolio weights.
    """

    # Get portfolio weights
    weights = get_portfolio_weights(state)

    # Initialize aggregated metrics
    agg = {
        "portfolio_beta": 0.0,
        "portfolio_sharpe": 0.0,
        "portfolio_sortino": 0.0,
        "portfolio_pe_ratio": 0.0,
        "portfolio_dividend_yield": 0.0,
        "portfolio_calmar_ratio": 0.0,
        "portfolio_tail_ratio": 0.0,
        "portfolio_omega_ratio": 0.0,
        "portfolio_treynor_ratio": 0.0,
        "portfolio_information_ratio": 0.0,
        "portfolio_skewness": 0.0,
        "portfolio_kurtosis": 0.0,
        "portfolio_var95": 0.0,
        "portfolio_max_drawdown": 0.0,
    }

    # Track weight sums for weighted metrics
    weight_sum = 0
    weighted_pe_sum = 0
    weighted_div_yield_sum = 0
    weighted_beta_sum = 0
    weighted_sharpe_sum = 0
    weighted_sortino_sum = 0
    weighted_calmar_sum = 0
    weighted_tail_sum = 0
    weighted_omega_sum = 0
    weighted_treynor_sum = 0
    weighted_info_ratio_sum = 0
    weighted_var95_sum = 0
    weighted_max_drawdown_sum = 0
    weighted_skew_sum = 0
    weighted_kurt_sum = 0

    for ms in getattr(state, "market_data", []):
        w = weights.get(ms.ticker, 0)
        
        if w == 0:
            continue
        
        weight_sum += w

        # Weighted metrics
        if ms.beta is not None:
            weighted_beta_sum += ms.beta * w
        if ms.sharpe_ratio is not None:
            weighted_sharpe_sum += ms.sharpe_ratio * w
        if ms.sortino_ratio is not None:
            weighted_sortino_sum += ms.sortino_ratio * w
        if ms.pe_ratio is not None:
            weighted_pe_sum += ms.pe_ratio * w
        if ms.dividend_yield is not None:
            weighted_div_yield_sum += ms.dividend_yield * w
        if ms.calmar_ratio is not None:
            weighted_calmar_sum += ms.calmar_ratio * w
        if ms.tail_ratio is not None:
            weighted_tail_sum += ms.tail_ratio * w
        if ms.omega_ratio is not None:
            weighted_omega_sum += ms.omega_ratio * w
        if ms.treynor_ratio is not None:
            weighted_treynor_sum += ms.treynor_ratio * w
        if ms.information_ratio is not None:
            weighted_info_ratio_sum += ms.information_ratio * w
        if ms.value_at_risk_95 is not None:
            weighted_var95_sum += ms.value_at_risk_95 * w
        if ms.max_drawdown is not None:
            weighted_max_drawdown_sum += ms.max_drawdown * w
        if ms.skewness is not None:
            weighted_skew_sum += ms.skewness * w
        if ms.kurtosis is not None:
            weighted_kurt_sum += ms.kurtosis * w

    # Assign weighted aggregates
    if weight_sum > 0:
        agg["portfolio_beta"] = weighted_beta_sum / weight_sum
        agg["portfolio_sharpe"] = weighted_sharpe_sum / weight_sum
        agg["portfolio_sortino"] = weighted_sortino_sum / weight_sum
        agg["portfolio_pe_ratio"] = weighted_pe_sum / weight_sum
        agg["portfolio_dividend_yield"] = weighted_div_yield_sum / weight_sum
        agg["portfolio_calmar_ratio"] = weighted_calmar_sum / weight_sum
        agg["portfolio_tail_ratio"] = weighted_tail_sum / weight_sum
        agg["portfolio_omega_ratio"] = weighted_omega_sum / weight_sum
        agg["portfolio_treynor_ratio"] = weighted_treynor_sum / weight_sum
        agg["portfolio_information_ratio"] = weighted_info_ratio_sum / weight_sum
        agg["portfolio_var95"] = weighted_var95_sum / weight_sum
        agg["portfolio_max_drawdown"] = weighted_max_drawdown_sum / weight_sum
        agg["portfolio_skewness"] = weighted_skew_sum / weight_sum
        agg["portfolio_kurtosis"] = weighted_kurt_sum / weight_sum

        if getattr(state.user_profile, "use_leverage", False):
            # Calculate portfolio leverage to hit target volatility
            target_vol = RISK_BUDGETS.get(getattr(state.user_profile, "risk_appetite", "moderate"), 0.25)
            pass
            
    return agg
    
def generate_technical_indicators(ms: MarketSnapshot):
    """
        Generates technical indicators for a portfolio of assets or stocks.
    """
    df = ms.historical_prices.copy()
    
    # Ensure the DataFrame has proper columns
    df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
    
    # SMA, EMA, RSI, MACD
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    
    return df

def plot_technical_chart(df, ticker):
    """  
        Plots a technical chart for a portfolio of assets or stocks.
    """
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name=f"{ticker} Close"))

    # SMA / EMA
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20"))

    # MACD subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="MACD Signal", yaxis="y2"))

    # RSI subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], mode="lines", name="RSI", yaxis="y3"))

    # Layout
    fig.update_layout(
        title=f"{ticker} Price & Technical Indicators",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="MACD", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="RSI", overlaying="y", side="right", position=0.85, showgrid=False),
        xaxis=dict(title="Date"),
        height=600
    )

    return fig

# Run Portfolio Optimization Workflow
if "agent_state" in st.session_state:
    agent_state = st.session_state["agent_state"]
    st.header("Execute Portfolio Optimization Workflow")

    if st.button("Run Portfolio Optimization"):
        with st.spinner("Processing investment workflow..."):
            # Execute LangGraph workflow
            final_state = workflow.invoke(agent_state)

        final_state = AgentState(**final_state)
        # Store final state
        st.session_state["final_state"] = final_state
        st.success("Workflow completed successfully!")

def render_list_as_bullets(lst, fallback="N/A"):
    if not lst:
        return fallback
    return "\n".join([f"- {item}" for item in lst])

def safe_avg(values):
    vals = [v for v in values if v is not None]
    return np.mean(vals) if vals else None

def compute_portfolio_leverage(final_state):
    holdings = getattr(final_state.portfolio_state, "holdings", {}) or {}
    market_data = getattr(final_state, "market_data", [])
    
    if not holdings or not market_data:
        return 1.0

    # Calculate portfolio volatility (unlevered)
    weights = get_portfolio_weights(final_state)
    tickers = [ms.ticker for ms in market_data if ms.ticker in weights]
    
    # Simple aggregation for UI display consistency
    vols = [ms.volatility for ms in market_data if ms.ticker in weights and ms.volatility]
    w_vals = [weights[ms.ticker] for ms in market_data if ms.ticker in weights and ms.volatility]
    
    if not w_vals: return 1.0
    
    avg_vol = np.average(vols, weights=w_vals)
    target_vol = RISK_BUDGETS.get(final_state.user_profile.risk_appetite, 0.25)
    
    leverage = target_vol / max(avg_vol, 0.01) if final_state.user_profile.use_leverage else 1.0
    return np.clip(leverage, 1.0, 4.0)

# Display Results
if "final_state" in st.session_state:
    final_state: AgentState = st.session_state["final_state"]

    st.header("Portfolio Summary")

    # Portfolio Metrics
    st.subheader("Portfolio Metrics")

    portfolio_metrics = aggregate_portfolio_metrics(final_state)
    
    market_data = getattr(final_state, "market_data", [])

    advanced_metrics = {
        # Risk & Return
        "Portfolio Beta": portfolio_metrics.get("portfolio_beta"),
        "Portfolio Sharpe Ratio": portfolio_metrics.get("portfolio_sharpe"),
        "Portfolio Sortino Ratio": portfolio_metrics.get("portfolio_sortino"),

        # Technical Indicators / Risk Metrics
        "Average Volatility": safe_avg([ms.volatility for ms in market_data]),
        "Average Value-at-Risk (95%)": safe_avg([ms.value_at_risk_95 for ms in market_data]),
        "Average Max Drawdown": safe_avg([ms.max_drawdown for ms in market_data]),
        "Average Calmar Ratio": safe_avg([ms.calmar_ratio for ms in market_data]),
        "Average Omega Ratio": safe_avg([ms.omega_ratio for ms in market_data]),
        "Average Tail Ratio": safe_avg([ms.tail_ratio for ms in market_data]),
        "Average Treynor Ratio": safe_avg([ms.treynor_ratio for ms in market_data]),
        "Average Information Ratio": safe_avg([ms.information_ratio for ms in market_data]),

        # Distribution
        "Average Skewness": safe_avg([ms.skewness for ms in market_data]),
        "Average Kurtosis": safe_avg([ms.kurtosis for ms in market_data]),

        # Valuation
        "Portfolio P/E Ratio": portfolio_metrics.get("portfolio_pe_ratio"),
        "Portfolio Dividend Yield": portfolio_metrics.get("portfolio_dividend_yield"),
    }

    # Display advanced metrics
    for metric, value in advanced_metrics.items():
        if value is None:
            st.markdown(f"- **{metric}:** N/A")
        else:
            if "Yield" in metric or "Return" in metric or "Volatility" in metric or "Drawdown" in metric:
                st.markdown(f"- **{metric}:** {value * 100:.2f}%")
            else:
                st.markdown(f"- **{metric}:** {value:.3f}")
    
    # Portfolio Optimization Results
    if hasattr(final_state, "optimization_result"):
        st.subheader("Portfolio Optimization Results")

        opt_res = final_state.optimization_result

        # Display new holdings
        if opt_res.new_holdings:
            opt_holdings_df = pd.DataFrame([
                {"Ticker": t, "Allocated ($)": v} for t, v in opt_res.new_holdings.items()
            ])
            st.markdown("**Optimized Portfolio Holdings:**")
            st.dataframe(opt_holdings_df)
        else:
            st.info("No optimized holdings available.")

    # Display key metrics
    st.markdown("**Optimization Metrics:**")
    
    portfolio_leverage = getattr(opt_res, "leverage_multiplier", 1.0)
    
    st.markdown(f"- **Expected Return (Levered):** {opt_res.expected_return*100:.2f}%")
    st.markdown(f"- **Portfolio Risk (Levered):** {opt_res.portfolio_risk*100:.2f}%")
    
    # Portfolio Assessment / Recommendation
    st.subheader("Portfolio Action Recommendation")
    assessment = getattr(final_state, "portfolio_assessment", {}) or {}
    st.markdown(f"**Action:** {assessment.get('action', 'N/A')}")
    st.markdown(f"**Explanation:** {assessment.get('explanation', 'N/A')}")

    # Portfolio Leverage Impact (if applicable)
    if hasattr(final_state.user_profile, "use_leverage") and final_state.user_profile.use_leverage:
        portfolio_leverage = getattr(opt_res, "leverage_multiplier", 1.0)
    
        st.subheader("Leverage Impact")
        levered_return = opt_res.expected_return
        unlevered_return = levered_return / portfolio_leverage if portfolio_leverage else levered_return

        st.markdown(f"""
            - **Portfolio Leverage:** {portfolio_leverage:.2f}
            - **Unlevered Expected Return:** {unlevered_return*100:.2f}%  
            - **Levered Expected Return:** {levered_return*100:.2f}%  
            - **Sharpe Ratio:** unchanged
        """)
        
    # Market Insights 
    if getattr(final_state, "market_insights", None):
        st.subheader("Market Insights per Ticker")
        for insight in final_state.market_insights:
            st.markdown(f"### {getattr(insight, 'ticker', 'N/A')}")
            # Display list attributes
            st.markdown(f"**Trend Insight:**\n{render_list_as_bullets(getattr(insight, 'price_trend_insights', getattr(insight, 'trend_insight', [])))}")
            st.markdown(f"**Risk Insight:**\n{render_list_as_bullets(getattr(insight, 'volatility_and_risk_insights', getattr(insight, 'risk_insight', [])))}")
            st.markdown(f"**Fundamental Insight:**\n{render_list_as_bullets(getattr(insight, 'fundamental_and_valuation_insights', getattr(insight, 'fundamental_insight', [])))}")
            st.markdown(f"**Correlation Insight:**\n{render_list_as_bullets(getattr(insight, 'correlation_insights', getattr(insight, 'correlation_insight', [])))}")
            st.markdown(f"**News Insight:**\n{render_list_as_bullets(getattr(insight, 'news_sentiment_insights', getattr(insight, 'news_insight', [])))}")

            # Recommendations: bullet points
            recommendations = getattr(insight, 'actionable_recommendations', getattr(insight, 'recommendations', [])) or []
            st.markdown(f"**Recommendations:**\n{render_list_as_bullets(recommendations)}")

            st.markdown("---")

    st.success("Portfolio optimization and analysis complete!")

# Portfolio Visualizations
if "final_state" in st.session_state:
    final_state: AgentState = st.session_state["final_state"]
    market_data = getattr(final_state, "market_data", [])
    st.header("Portfolio and Market Visualizations")

    # Plot historical prices safely
    for ms in getattr(final_state, "market_data", []):
        if getattr(ms, "historical_prices", None) is None:
            continue

        df = ms.historical_prices.copy()

        if "Close" not in df.columns:
            continue

        df = df.rename(columns={
            "Close": "close",
            "Volume": "volume"
        })

        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_50"] = df["close"].rolling(50).mean()

        st.subheader(f"Historical Price Trend — {ms.ticker}")

        fig = go.Figure()

        # Price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["close"],
            name="Close Price",
            mode="lines",
            line=dict(width=2)
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["SMA_20"],
            name="SMA 20",
            line=dict(dash="dot")
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["SMA_50"],
            name="SMA 50",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            height=450,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Price",
            title=f"{ms.ticker} — Historical Price with Trend",
            legend=dict(orientation="h", y=-0.25)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Portfolio allocation bar chart
    holdings = getattr(final_state.portfolio_state, "holdings", {}) or {}
    
    if holdings:
        st.subheader("Portfolio Allocation")
        allocation_df = pd.DataFrame({
            "Ticker": list(holdings.keys()),
            "Value": list(holdings.values())
        })
        
        if hasattr(final_state.user_profile, "use_leverage") and final_state.user_profile.use_leverage:
            portfolio_leverage = np.average(
                [ms.leverage_multiplier for ms in final_state.market_data],
                weights=[holdings.get(ms.ticker, 0) for ms in final_state.market_data]
            )
            allocation_df["Value"] = allocation_df["Value"] * portfolio_leverage  # adjust for leverage
        
        allocation_df = allocation_df[allocation_df["Value"] > 0] # Filter out assets with zero allocation
        allocation_df = allocation_df.sort_values("Value", ascending=False) 
        allocation_df["Percentage"] = allocation_df["Value"] / allocation_df["Value"].sum() * 100
        allocation_df["Label"] = allocation_df.apply(
            lambda row: f"{row['Ticker']} ({row['Percentage']:.1f}%)", axis=1
        )

        bar_chart = alt.Chart(allocation_df).mark_bar().encode(
            x=alt.X("Value:Q", title="Allocation ($)"),
            y=alt.Y("Ticker:N", sort=alt.SortField("Value", order="descending"), title="Ticker"),
            color=alt.Color("Ticker:N", legend=None),
            tooltip=["Ticker", "Value", alt.Tooltip("Percentage", format=".1f")]
        )

        text = alt.Chart(allocation_df).mark_text(
            align='left',
            baseline='middle',
            dx=3,  # small offset from bar
            fontWeight='bold',
            size=12
        ).encode(
            x='Value:Q',
            y=alt.Y('Ticker:N', sort=alt.SortField("Value", order="descending")),
            text='Label'
        )

        st.altair_chart(bar_chart + text, use_container_width=True)

    if market_data:
        # Create tabs for each ticker
        tabs = st.tabs([md.ticker for md in market_data])

        for idx, ms in enumerate(market_data):
            with tabs[idx]:
                st.subheader(f"{ms.ticker} Technical Chart")

                if getattr(ms, "historical_prices", None) is None:
                    st.info("No historical price data available.")
                    continue

                # Prepare historical prices and compute indicators
                df = ms.historical_prices.copy()
                df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
                
                # Add indicators using pandas-ta
                df["SMA_20"] = ta.sma(df["close"], length=20)
                df["SMA_50"] = ta.sma(df["close"], length=50)
                df["EMA_20"] = ta.ema(df["close"], length=20)
                df["RSI_14"] = ta.rsi(df["close"], length=14)
                macd = ta.macd(df["close"])
                df["MACD"] = macd["MACD_12_26_9"]
                df["MACD_Signal"] = macd["MACDs_12_26_9"]
                
                bb = ta.bbands(df['close'], length=20, std=2)
                df["BB_Upper"] = bb["BBU_20_2.0"]
                df["BB_Middle"] = bb["BBM_20_2.0"]
                df["BB_Lower"] = bb["BBL_20_2.0"]

                # Plotly chart
                fig = go.Figure()

                # Price
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["close"],
                    mode="lines", name="Close Price"
                ))

                # Moving Averages
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20"))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50"))
                fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20"))

                # Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["BB_Upper"],
                    name="BB Upper", line=dict(width=1, dash="dot")
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["BB_Lower"],
                    name="BB Lower", line=dict(width=1, dash="dot"),
                    fill="tonexty", opacity=0.1
                ))

                # Volume
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df["volume"],
                    name="Volume",
                    yaxis="y4",
                    opacity=0.3
                ))

                # MACD
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["MACD"],
                    name="MACD", yaxis="y2"
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["MACD_Signal"],
                    name="MACD Signal", yaxis="y2"
                ))

                # RSI
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["RSI_14"],
                    name="RSI", yaxis="y3"
                ))

                # RSI Thresholds
                fig.add_hline(y=70, line_dash="dash", yref="y3", opacity=0.4)
                fig.add_hline(y=30, line_dash="dash", yref="y3", opacity=0.4)

                # Layout
                fig.update_layout(
                    title=f"{ms.ticker} Price & Technical Indicators",
                    xaxis=dict(title="Date"),

                    # Price axis
                    yaxis=dict(title="Price"),

                    # MACD
                    yaxis2=dict(
                        title="MACD",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),

                    # RSI
                    yaxis3=dict(
                        title="RSI",
                        overlaying="y",
                        side="right",
                        position=0.85,
                        range=[0, 100],
                        showgrid=False
                    ),

                    # Volume axis
                    yaxis4=dict(
                        title="Volume",
                        overlaying="y",
                        side="left",
                        position=0.05,
                        showgrid=False
                    ),

                    height=650,
                    legend=dict(orientation="h", y=-0.25)
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 💡 AI Investment Guidance")

                latest = df.iloc[-1]

                latest_state = {
                    "price": float(latest["close"]),
                    "trend": "uptrend" if latest["close"] > latest["SMA_50"] else "downtrend",
                    "short_term_momentum": "positive" if latest["MACD"] > latest["MACD_Signal"] else "negative",
                    "rsi": round(float(latest["RSI_14"]), 2),
                    "volatility_state": (
                        "high"
                        if latest["close"] > latest["BB_Upper"] or latest["close"] < latest["BB_Lower"]
                        else "normal"
                    ),
                    "volume_confirmation": (
                        "strong"
                        if latest["volume"] > df["volume"].rolling(20).mean().iloc[-1]
                        else "weak"
                    )
                }

                investment_prompt = f"""
                    You are an experienced portfolio manager providing practical investment guidance.

                    Asset: {ms.ticker}

                    Technical state:
                    {latest_state}

                    Context:
                    - The investor is building a diversified portfolio.
                    - Guidance should be short- to medium-term (weeks to months).
                    - Provide a directional stance, not just indicator interpretation.

                    Instructions:
                    1. State a clear action: Buy, Hold, Sell, or Skip.
                    2. Justify the action using trend, momentum, volatility, and volume.
                    3. Highlight key risks or invalidation signals.
                    4. Keep it concise (max 5 bullet points).
                    5. Do NOT include disclaimers or generic explanations of indicators.
                """

                # Streamlit placeholder for streaming text
                response_placeholder = st.empty()

                streamed_text = ""

                # Streaming LLM response
                for chunk in llm.stream(investment_prompt):
                    streamed_text += chunk
                    response_placeholder.markdown(streamed_text)