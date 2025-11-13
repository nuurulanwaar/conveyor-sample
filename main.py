import os
import sys
from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI, Request
from telegram import Bot
from starlette.responses import JSONResponse
import logging
from asyncio import TimeoutError
from collections import defaultdict
import re
import uvicorn
import re
from calendar import month_name, month_abbr
# --- IMPORT TOOLS ---
from firebase_setup import (
    get_latest_conveyor_status,
    get_yesterday_production,
    get_production_history,
    get_total_log_count,
    get_all_event_logs,
    get_count_only,
    analyze_production_insights,
    get_month_production,
    data_sorter  # ← ADDED: internal tool
)

# --- CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN or not os.environ["GOOGLE_API_KEY"]:
    logging.error("Missing TELEGRAM_BOT_TOKEN or GOOGLE_API_KEY")
    sys.exit(1)

# --- FASTAPI & BOT ---
app = FastAPI()
bot = Bot(token=TELEGRAM_BOT_TOKEN)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

CHAT_HISTORIES = defaultdict(list)

# --- TOOLS LIST: data_sorter REMOVED (internal only) ---
tools = [
    get_latest_conveyor_status,
    get_yesterday_production,
    get_production_history,
    get_total_log_count,
    get_all_event_logs,
    get_count_only,
    analyze_production_insights,
    get_month_production,
]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

system_prompt = (
    "You are the Conveyor AI Operator — a smart, factory-hardened assistant.\n"
    "Your mission: monitor real-time production, detect faults, and guide operators with clarity.\n\n"
    "also handle non related questions intelligently, such as Hi, hello, tell me story, just be intellegint, but stay within industrial context "
    "### LATEST STATUS HANDLING\n"
    "Use get_latest_conveyor_status() for any query with: 'latest', 'current', 'now', 'status','sensors(IR, proximity, load cell, infrared)' 'motor', 'count', 'weight'.\n\n"
    "AI RESPONSE RULES:\n"
    "- 'latest production' or 'latest' → Full status + motor analysis + 1-paragraph actionable conclusion\n"
    "- 'latest count' → Only: Count: X items\n"
    "- 'latest weight, load' → Only: Weight: X kg\n"
    "- 'latest motor' or 'motor status' → Only: Current: X.XA → STATUS\n\n"
    "Always include a Recommendation paragraph if motor is not NORMAL.\n"
    "Example: 'OVERLOADED at 31.13A — URGENT: Stop line, inspect belt and VFD.'\n\n"

    "### TIME-FILTERED & MONTHLY PRODUCTION (PARTIAL AI)\n"
    "- For 'yesterday production', 'monthly production', 'this month': Use data_sorter(time_filter=...)\n"
    "  • ≤30 logs → AI explains in 1 line\n"
    "  • >30 logs → bot sends raw batches (no AI)\n"
    "- For 'October production', 'Nov', 'January', 'Feb': Use get_month_production(month_name)\n"
    "  • Same partial AI rule: ≤30 → AI insight, >30 → batch send\n"
    "  • End AI response with 1 actionable insight\n\n"

    "### CUMULATIVE BATCH LOGIC (CRITICAL)\n"
    "- Count is cumulative — the last count in a sequence is the total batch size\n"
    "- A new batch starts when:\n"
    "  • Count drops (e.g., 578 → 488)\n"
    "  • Count becomes 1\n"
    "- Never say 'reset' — say: 'new batch started'\n"
    "- Example: 150 → 151 → 152 → 1 → 2 → 412 → 413\n"
    "  → Two batches: 152 items, then 413 items\n\n"

    "### COUNT REQUESTS\n"
    "- 'show X counts', 'X count', 'last X logs':\n"
    "  • X ≤ 30 → use get_count_only(X)\n"
    "  • X > 30 → use get_all_event_logs(X) → batch send\n"
    "  • Never format large lists. Let system batch.\n\n"

    "### INSIGHTS & ANALYSIS\n"
    "- For: 'how’s production?', 'analyze', 'summary', 'today', 'performance':\n"
    "  • Use analyze_production_insights(limit)\n"
    "  • Respond in plain, human language with actionable advice\n"
    "  • Example: '3 small batches with high current → check sensor at 11:20.'\n\n"

    "### GENERAL RULES\n"
    "- Be direct, calm, and production-first.\n"
    "- Use real data only — no guesses.\n"
    "- Detect:\n"
    "  • Jams → small count + high current\n"
    "  • New batch → count drops or goes to 1\n"
    "  • Overwork → current > 15A\n"
    "  • Underwork → current < 5A\n"
    "- Always include time, count, current (A), weight when relevant.\n"
    "- Negative current? Report: '-8.5A' → '8.5A load'\n"
    "- Off-topic? → 'I’m your conveyor expert. How can I help with production?'\n\n"

    "### EXAMPLE RESPONSES\n"
    "• User: 'latest production'\n"
    "  → 'Latest: 05-11-25 15:44:20 | 2 items | 0 kg | 31.13A → OVERLOADED\n\n"
    "    Recommendation: URGENT: Motor critically overloaded. Risk of burnout. Stop line immediately, reduce load, inspect belt slip, VFD, and motor bearings.'\n\n"
    "• User: 'latest count'\n"
    "  → 'Latest count: 2 items (start of new batch)'\n\n"
    "• User: 'yesterday production'\n"
    "  → 'Yesterday: 2 batches, 562 items. New batch started at 12:00 with 1 item. Motor normal.'\n\n"
    "• User: 'October production'\n"
    "  → (≤30 logs): 'October: 3,200 items in 12 batches. 2 new batches started. Motor normal. Check feeder alignment.'\n"
    "  → (>30 logs): bot sends batched logs → no AI\n\n"
    "• User: 'analyze 50'\n"
    "  → 'Last 50 logs: 1,102 items in 3 batches. New batch at 12:00. Motor normal at 8.5A avg. Good uptime.'"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# --- BACKGROUND (DISABLED) ---
async def background_data_monitor():
    while True:
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_data_monitor())
    logging.info("Server started.")

# --- PROCESS MESSAGE ---
async def process_message(user_query: str, chat_id: int):
    history = CHAT_HISTORIES[chat_id]
    lower_query = user_query.lower()

    # === 1. TIME-FILTERED REQUESTS (yesterday, hourly, month) ===
    time_keywords = {
    "yesterday": "yesterday",
    "hour": "hourly",
    "hourly": "hourly",
    "today": "hourly",
    "month": "month",
    "monthly": "month"
    }
    filter_used = None
    for word, filter_type in time_keywords.items():
        if word in lower_query:
            filter_used = filter_type
            break

    if filter_used:
        limit = 1000
        match = re.search(r'(\d+)', user_query)
        if match:
            limit = int(match.group(1))

    # DEBUG LINE — PRINTS TO TERMINAL    
        print(f"[DEBUG] Time filter: '{filter_used}' | limit: {limit} | query: '{user_query}'")
        try:
        # USE SORTER DIRECTLY
            sorted_data = data_sorter(limit=limit, time_filter=filter_used)
            logs = sorted_data.get("list", [])

            if len(logs) == 0:
            # SMART: Use the actual keyword from query
                used_keyword = next(word for word in time_keywords if word in lower_query)
                return f"There was no production {used_keyword}."
            
            # BYPASS IF >30 LOGS
            if len(logs) > 30:
                num_batches = (len(logs) + 39) // 40
                for i in range(num_batches):
                    batch = logs[i*40:(i+1)*40]
                    text = "\n".join([
                        f"• {log['time']} → count: {log['count']}"
                        for log in batch
                    ])
                    await bot.send_message(chat_id=chat_id, text=text)
                    await asyncio.sleep(0.5)
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=f"Sent {len(logs)} logs in {num_batches} message(s)."))
                CHAT_HISTORIES[chat_id] = history[-10:]
                return

            # SMALL DATA → AI EXPLAINS
            else:
                response = await asyncio.wait_for(
                    agent_executor.ainvoke({
                        "input": f"Explain in 1 line: {sorted_data}",
                        "chat_history": history
                    }),
                    timeout=20
                )
                ai_text = response['output']
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=ai_text))
                CHAT_HISTORIES[chat_id] = history[-10:]
                return ai_text

        except Exception as e:
            return "Error fetching data."

        # === 2. NAMED MONTH REQUESTS (October, January, etc.) — PARTIAL AI ===
            # === 2. SMART MONTH DETECTION (NO LIST, FULLY DYNAMIC) ===
    

    # Build regex: (january|jan|jan's|january's|...)
    months = [m.lower() for m in month_name[1:]] + [m.lower() for m in month_abbr[1:]]
    pattern = r'\b(' + '|'.join(re.escape(m) + r"'?s?" for m in months) + r')\b'
    match = re.search(pattern, lower_query)
    
    if match:
        month_detected = match.group(1)  # e.g., "november's"
        # Clean: remove 's or s
        clean_month = re.sub(r"'?s$", "", month_detected)
        
        try:
            result = get_month_production.invoke({"month_name": clean_month})
            if "message" in result:
                return result["message"]

            logs = result.get("logs", [])
            if not logs:
                return f"No data for {clean_month.capitalize()}."

            # === SAME AI / BATCH LOGIC AS BEFORE ===
            if len(logs) > 30:
                num_batches = (len(logs) + 39) // 40
                for i in range(num_batches):
                    batch = logs[i*40:(i+1)*40]
                    text = "\n".join([
                        f"• {log['time']} → count: {log['count']}, current: {log['current']}A"
                        for log in batch
                    ])
                    await bot.send_message(chat_id=chat_id, text=text)
                    await asyncio.sleep(0.5)
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=f"Sent {len(logs)} logs for {clean_month.capitalize()}."))
                CHAT_HISTORIES[chat_id] = history[-10:]
                return
            else:
                response = await asyncio.wait_for(
                    agent_executor.ainvoke({
                        "input": f"Summarize {clean_month} production: {result}",
                        "chat_history": history
                    }),
                    timeout=15
                )
                ai_text = response['output']
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=ai_text))
                CHAT_HISTORIES[chat_id] = history[-10:]
                return ai_text

        except Exception as e:
            return "Error fetching monthly data."

    # === 4. LARGE COUNT REQUESTS (>30) — BYPASS AI ===
    count_keywords = ["count", "counts", "log", "logs", "production"]
    if any(word in lower_query for word in count_keywords):
        match = re.search(r'(\d+)', user_query)
        if match:
            limit = int(match.group(1))
            if limit > 30:
                try:
                    tool_response = get_all_event_logs.invoke({"limit": limit})
                    logs = tool_response.get("logs", [])
                    if not logs:
                        await bot.send_message(chat_id=chat_id, text="No logs found.")
                        return

                    num_batches = (len(logs) + 39) // 40
                    for i in range(num_batches):
                        start = i * 40
                        end = min(start + 40, len(logs))
                        batch = logs[start:end]
                        text = "\n".join([
                            f"• {log.get('time_recorded','??:??:??')} → count: {log.get('count','?')}, current: {log.get('current','?')}A"
                            for log in batch
                        ])
                        await bot.send_message(chat_id=chat_id, text=text)
                        await asyncio.sleep(0.5)

                    history.append(HumanMessage(content=user_query))
                    history.append(AIMessage(content=f"Sent {len(logs)} logs in {num_batches} message(s)."))
                    CHAT_HISTORIES[chat_id] = history[-10:]
                    return

                except Exception as e:
                    await bot.send_message(chat_id=chat_id, text=f"Error: {e}")
                    return

    # === 5. NORMAL AI FLOW ===
    timeout = 1000 if "all" in lower_query else 25
    try:
        response = await asyncio.wait_for(
            agent_executor.ainvoke({
                "input": user_query,
                "chat_history": history
            }),
            timeout=timeout
        )
        ai_text = response['output']

        # FALLBACK BATCHING
        if isinstance(response.get("logs"), list):
            logs = response["logs"]
            if len(logs) > 30:
                num_batches = (len(logs) + 39) // 40
                for i in range(num_batches):
                    batch = logs[i*40:(i+1)*40]
                    text = "\n".join([
                        f"• {log.get('time_recorded','??:??:??')} → count: {log.get('count','?')}, current: {log.get('current','?')}A"
                        for log in batch
                    ])
                    await bot.send_message(chat_id=chat_id, text=text)
                    await asyncio.sleep(0.5)
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=f"Sent {len(logs)} logs in {num_batches} batches."))
                CHAT_HISTORIES[chat_id] = history[-10:]
                return

        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=ai_text))
        CHAT_HISTORIES[chat_id] = history[-10:]
        return ai_text

    except TimeoutError:
        return "Timed out. Ask about production only."
    except Exception as e:
        return f"Error: {e}"

# --- WEBHOOK ---
@app.post("/webhook")
async def webhook(request: Request):
    try:
        update = await request.json()
    except:
        return JSONResponse({"status": "invalid"}, 400)

    if 'message' not in update:
        return JSONResponse({"status": "ok"}, 200)

    msg = update['message']
    chat_id = msg['chat']['id']

    if 'text' not in msg:
        await bot.send_message(chat_id, "Please send text only.")
        return JSONResponse({"status": "ok200"}, 200)

    user_text = msg['text'].strip()

    # SEND "THINKING..." IMMEDIATELY
    #await bot.send_message(chat_id, "Thinking...")

    # PROCESS IN BACKGROUND
    asyncio.create_task(process_long_task(user_text, chat_id))

    return JSONResponse({"status": "ok"}, 200)


# BACKGROUND TASK
async def process_long_task(user_text: str, chat_id: int):
    try:
        reply = await asyncio.wait_for(process_message(user_text, chat_id), timeout=30)
        if reply:
            await bot.send_message(chat_id, reply)
    except asyncio.TimeoutError:
        await bot.send_message(chat_id, "Timed out. Ask about production only.")
    except Exception as e:
        await bot.send_message(chat_id, f"Error: {e}")




if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Railway sets PORT=8080
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")






