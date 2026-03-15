import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Environment and app constants
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

MODEL_SMALL = "gpt-4o-mini"
MODEL_LARGE = "gpt-4o"

DB_PATH = "stocks.db"
CSV_PATH = "sp500_companies.csv"


@dataclass
class AgentResult:
    agent_name: str
    answer: str
    tools_called: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    confidence: float = 0.0
    issues_found: list = field(default_factory=list)
    reasoning: str = ""


def create_local_database(csv_path: str = CSV_PATH):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(
        columns={
            "symbol": "ticker",
            "shortname": "company",
            "sector": "sector",
            "industry": "industry",
            "exchange": "exchange",
            "marketcap": "market_cap_raw",
        }
    )

    def cap_bucket(v):
        try:
            v = float(v)
            if v >= 10_000_000_000:
                return "Large"
            if v >= 2_000_000_000:
                return "Mid"
            return "Small"
        except Exception:
            return "Unknown"

    df["market_cap"] = df["market_cap_raw"].apply(cap_bucket)
    df = (
        df.dropna(subset=["ticker", "company"])
        .drop_duplicates(subset=["ticker"])
        [["ticker", "company", "sector", "industry", "market_cap", "exchange"]]
    )
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
    conn.commit()
    conn.close()


def ensure_db_ready():
    if not os.path.exists(DB_PATH):
        create_local_database()


# -----------------------------
# Tool functions (from notebook)
# -----------------------------
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data - possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price": round(end, 2),
                "pct_change": round((end - start) / start * 100, 2),
                "period": period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_top_gainers_losers() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title": a.get("title"),
                "source": a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score": a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }


def query_local_db(sql: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


def get_company_overview(ticker: str) -> dict:
    url = (
        f"https://www.alphavantage.co/query?function=OVERVIEW"
        f"&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    data = requests.get(url, timeout=10).json()
    if "Name" not in data:
        return {"error": f"No overview data for {ticker}"}
    return {
        "ticker": ticker,
        "name": data.get("Name", ""),
        "sector": data.get("Sector", ""),
        "pe_ratio": data.get("PERatio", ""),
        "eps": data.get("EPS", ""),
        "market_cap": data.get("MarketCapitalization", ""),
        "52w_high": data.get("52WeekHigh", ""),
        "52w_low": data.get("52WeekLow", ""),
    }


def get_tickers_by_sector(sector: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT ticker, company, industry FROM stocks WHERE LOWER(sector) = LOWER(?)"
    df = pd.read_sql_query(query, conn, params=[sector])
    if df.empty:
        query = (
            "SELECT ticker, company, industry FROM stocks "
            "WHERE LOWER(industry) LIKE LOWER(?)"
        )
        df = pd.read_sql_query(query, conn, params=[f"%{sector}%"])
    conn.close()
    return {"sector": sector, "stocks": df.to_dict(orient="records")}


def _s(name, desc, props, req):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }


SCHEMA_TICKERS = _s(
    "get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database.",
    {"sector": {"type": "string", "description": "Sector or industry name"}},
    ["sector"],
)
SCHEMA_PRICE = _s(
    "get_price_performance",
    "Get % price change for a list of tickers over a time period.",
    {
        "tickers": {"type": "array", "items": {"type": "string"}},
        "period": {"type": "string", "default": "1y"},
    },
    ["tickers"],
)
SCHEMA_OVERVIEW = _s(
    "get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. AAPL"}},
    ["ticker"],
)
SCHEMA_STATUS = _s("get_market_status", "Check whether markets are open.", {}, [])
SCHEMA_MOVERS = _s(
    "get_top_gainers_losers",
    "Get today's top gaining, top losing, and most active stocks.",
    {},
    [],
)
SCHEMA_NEWS = _s(
    "get_news_sentiment",
    "Get latest headlines and sentiment for a stock.",
    {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
    ["ticker"],
)
SCHEMA_SQL = _s(
    "query_local_db",
    "Run a SQL SELECT on stocks.db.",
    {"sql": {"type": "string"}},
    ["sql"],
)

ALL_SCHEMAS = [
    SCHEMA_TICKERS,
    SCHEMA_PRICE,
    SCHEMA_OVERVIEW,
    SCHEMA_STATUS,
    SCHEMA_MOVERS,
    SCHEMA_NEWS,
    SCHEMA_SQL,
]

ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector": get_tickers_by_sector,
    "get_price_performance": get_price_performance,
    "get_company_overview": get_company_overview,
    "get_market_status": get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment": get_news_sentiment,
    "query_local_db": query_local_db,
}


SINGLE_AGENT_PROMPT = """
You are an expert financial analyst assistant with access to market-data tools and a local stock database.
Always use tool data for current information.
Never fabricate numbers. If a tool returns missing/error data, report that clearly.
When a question mentions a sector/industry, call get_tickers_by_sector first.
When multiple data types are needed (price + fundamentals + sentiment), call separate tools and then synthesize.
"""

ORCHESTRATOR_PROMPT = """You are an orchestrator for a financial analysis system.
Decide which specialists to activate for the user's request:
- market
- fundamentals
- sentiment
Return ONLY JSON:
{"market": true/false, "fundamentals": true/false, "sentiment": true/false, "plan": "short plan"}
"""

MARKET_SPECIALIST_PROMPT = """You are a Market Data Specialist.
Use only tool outputs. Never fabricate prices/returns.
For sector questions, call get_tickers_by_sector first.
"""

FUNDAMENTALS_SPECIALIST_PROMPT = """You are a Fundamentals Specialist.
Use only tool outputs. Never fabricate P/E, EPS, market cap, or 52-week values.
"""

SENTIMENT_SPECIALIST_PROMPT = """You are a Sentiment Specialist.
Use only tool outputs. Never fabricate headlines or sentiment scores.
"""

CRITIC_PROMPT = """You are a Critic that verifies specialist answers against raw tool outputs.
Return ONLY JSON:
{
  "reviews": [
    {"agent":"...", "confidence":0.0, "issues":["..."], "hallucination_detected": false}
  ]
}
"""

SYNTHESIZER_PROMPT = """You are a Synthesizer.
Combine verified specialist outputs into one final answer.
Use only provided specialist content and note uncertainty if issues are flagged.
"""

MARKET_TOOLS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS = [SCHEMA_NEWS, SCHEMA_SQL, SCHEMA_TICKERS]


def _parse_json_response(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def build_conversation_context(messages: List[Dict]) -> str:
    # Pass full conversation history on every turn.
    recent = messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return (
        "Conversation history (use this to resolve references like 'that' or 'the two'):\n"
        + "\n".join(lines)
    )


def run_specialist_agent(
    client: OpenAI,
    model_name: str,
    agent_name: str,
    system_prompt: str,
    task: str,
    tool_schemas: list,
    max_iters: int = 8,
) -> AgentResult:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    tools_called = []
    raw_data = {}
    msg = None

    for _ in range(max_iters):
        kwargs = {"model": model_name, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            fn = ALL_TOOL_FUNCTIONS[fn_name]
            result = fn(**fn_args)
            tools_called.append(fn_name)
            raw_data[fn_name] = result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    answer = (msg.content if msg else "") or ""
    return AgentResult(agent_name=agent_name, answer=answer, tools_called=tools_called, raw_data=raw_data)


def run_single_agent(client: OpenAI, model_name: str, conversation_messages: List[Dict]) -> AgentResult:
    context_task = build_conversation_context(conversation_messages)
    return run_specialist_agent(
        client=client,
        model_name=model_name,
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=context_task,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
    )


def run_multi_agent(client: OpenAI, model_name: str, conversation_messages: List[Dict]) -> dict:
    t_start = time.time()
    context_task = build_conversation_context(conversation_messages)
    all_agent_results = []

    orch_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": context_task},
        ],
    )
    plan = _parse_json_response(orch_response.choices[0].message.content or "")

    activate_market = plan.get("market", True)
    activate_fundamentals = plan.get("fundamentals", True)
    activate_sentiment = plan.get("sentiment", True)

    specialists = []
    if activate_market:
        specialists.append(("Market Specialist", MARKET_SPECIALIST_PROMPT, MARKET_TOOLS))
    if activate_fundamentals:
        specialists.append(("Fundamentals Specialist", FUNDAMENTALS_SPECIALIST_PROMPT, FUNDAMENTAL_TOOLS))
    if activate_sentiment:
        specialists.append(("Sentiment Specialist", SENTIMENT_SPECIALIST_PROMPT, SENTIMENT_TOOLS))
    if not specialists:
        specialists.append(("Market Specialist", MARKET_SPECIALIST_PROMPT, MARKET_TOOLS))

    def _run_one(name, prompt, tools):
        return run_specialist_agent(
            client=client,
            model_name=model_name,
            agent_name=name,
            system_prompt=prompt,
            task=context_task,
            tool_schemas=tools,
            max_iters=8,
        )

    with ThreadPoolExecutor(max_workers=len(specialists)) as executor:
        futures = {executor.submit(_run_one, name, prompt, tools): name for name, prompt, tools in specialists}
        for fut in as_completed(futures):
            all_agent_results.append(fut.result())

    critic_input_parts = []
    for r in all_agent_results:
        raw_summary = json.dumps(r.raw_data, default=str)[:2000]
        critic_input_parts.append(
            f"=== {r.agent_name} ===\n"
            f"Answer: {r.answer[:1000]}\n"
            f"Tools called: {r.tools_called}\n"
            f"Raw tool data (truncated): {raw_summary}\n"
        )
    critic_input = "\n".join(critic_input_parts)

    critic_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"{context_task}\n\nSpecialist outputs:\n{critic_input}"},
        ],
    )
    critic_result = _parse_json_response(critic_response.choices[0].message.content or "")
    for review in critic_result.get("reviews", []):
        reviewed_name = (review.get("agent") or "").lower()
        for r in all_agent_results:
            if r.agent_name.lower() == reviewed_name:
                r.confidence = review.get("confidence", 0.5)
                r.issues_found = review.get("issues", [])
                break

    synth_input_parts = []
    for r in all_agent_results:
        synth_input_parts.append(
            f"=== {r.agent_name} (confidence: {r.confidence:.0%}) ===\n"
            f"{r.answer}\n"
            f"Issues flagged: {r.issues_found if r.issues_found else 'none'}\n"
        )
    synth_input = "\n".join(synth_input_parts)

    synth_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{context_task}\n\nVerified specialist answers:\n{synth_input}\n\n"
                    "Provide the final response to the latest user question."
                ),
            },
        ],
    )

    return {
        "final_answer": synth_response.choices[0].message.content or "",
        "agent_results": all_agent_results,
        "elapsed_sec": round(time.time() - t_start, 2),
        "architecture": "orchestrator-parallel-critic",
    }


def main():
    st.set_page_config(page_title="FinTech Agent Chat", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("FinTech Agent Chat")
    st.caption("Single-Agent and Multi-Agent chat wrapper with multi-turn memory.")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment. Add it to your .env file.")
        st.stop()
    if not ALPHAVANTAGE_API_KEY:
        st.warning("ALPHAVANTAGE_API_KEY not found. Some tools may return empty/error data.")

    ensure_db_ready()

    with st.sidebar:
        st.header("Controls")
        architecture_choice = st.selectbox("Agent selector", ["Single Agent", "Multi-Agent"])
        model_choice = st.selectbox("Model selector", [MODEL_SMALL, MODEL_LARGE], index=0)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(
                    f"Architecture: {message.get('architecture', 'n/a')} | Model: {message.get('model', 'n/a')}"
                )

    user_input = st.chat_input("Ask a financial question...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    client = OpenAI(api_key=OPENAI_API_KEY)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if architecture_choice == "Single Agent":
                    result = run_single_agent(client, model_choice, st.session_state.messages)
                    assistant_text = result.answer
                    arch_used = "single-agent"
                else:
                    result = run_multi_agent(client, model_choice, st.session_state.messages)
                    assistant_text = result["final_answer"]
                    arch_used = result["architecture"]
            except Exception as e:
                assistant_text = f"Error: {e}"
                arch_used = architecture_choice.lower()

        st.markdown(assistant_text)
        st.caption(f"Architecture: {arch_used} | Model: {model_choice}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "architecture": arch_used,
            "model": model_choice,
        }
    )


if __name__ == "__main__":
    main()

