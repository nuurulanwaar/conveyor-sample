import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
from langchain_core.tools import tool
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
from datetime import datetime, timezone, timedelta
WAT = timezone(timedelta(hours=1))  # UTC+1
import calendar

# Load environment variables
load_dotenv()

# Initialize Firebase
service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
database_url = os.getenv("FIREBASE_DATABASE_URL")

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {'databaseURL': database_url})

db_ref = db.reference('/')

# Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- TOOL 1: LATEST STATUS ---
@tool
def get_latest_conveyor_status() -> Dict[str, Any]:
    """
    Returns structured latest event data for AI to analyze and respond.
    AI will decide: full status, count only, motor only, etc.
    """
    data = db_ref.child('latest_event').get()
    if not data:
        return {"error": "No current data. Conveyor offline."}

    # Extract
    count = data.get('count', 0)
    current = data.get('current', 0.0)
    weight = data.get('weight', 0.0)
    ts = data.get('timestamp_epoch')

    # Format time
    if ts and isinstance(ts, (int, float)):
        time_str = datetime.fromtimestamp(ts).strftime('%d-%m-%y %H:%M:%S')
    else:
        time_str = data.get('timestamp_formatted', '??:??:??')

    # Motor status logic (for AI to use)
    if current < 5.0:
        motor_status = "UNDERWORKING"
        suggestion = "Check material feed. Motor idle or blocked."
    elif 5.0 <= current <= 23.0:
        motor_status = "NORMAL"
        suggestion = "Optimal operation. Continue monitoring."
    elif 23.0 < current <= 35.0:
        motor_status = "HIGH LOAD"
        suggestion = "Heavy load detected. Inspect belt tension or material flow."
    else:
        motor_status = "OVERLOADED"
        suggestion = "URGENT: Risk of motor burnout. Stop line and inspect immediately."

    return {
        "time_recorded": time_str,
        "count": count,
        "weight": weight,
        "current": current,
        "motor_status": motor_status,
        "suggestion": suggestion,
        "raw": data
    }

# --- TOOL 2: YESTERDAY'S PRODUCTION ---
@tool
def get_yesterday_production() -> Dict[str, Any]:
    """Summarize yesterday using data_sorter."""
    data = data_sorter(limit=1000, time_filter="yesterday")
    logs = data.get("list", [])
    if not logs:
        return {"message": "There was no production yesterday."}

    total_items = sum(l['count'] for l in logs if isinstance(l['count'], (int, float)))
    return {
        "period": "Yesterday",
        "total_items": total_items,
        "updates": len(logs)
    }

# TOO 2A: MONTHLY TOOL
@tool
def get_month_production(month_name: str) -> Dict[str, Any]:
    """Fast monthly summary using data_sorter (supports Oct, Nov, etc.)."""
    # === CALL DATA SORTER DIRECTLY ===
    data = data_sorter(limit=1000, time_filter="specific_month", month_name=month_name)
    
    # === HANDLE ERRORS FROM DATA_SORTER ===
    if "error" in data:
        return {"message": f"Invalid month: {month_name}. Use full name or first 3 letters."}
    
    logs = data.get("list", [])
    if not logs:
        return {"message": f"No data for {month_name.capitalize()} {datetime.now(WAT).year}."}

    # === BATCH DETECTION (cumulative count) ===
    batches = []
    current_batch = []
    prev_count = 0
    for log in logs:
        c = log.get('count', 0)
        if isinstance(c, (int, float)) and c < prev_count and current_batch:
            batches.append(current_batch)
            current_batch = [log]
        else:
            current_batch.append(log)
        prev_count = c
    if current_batch:
        batches.append(current_batch)

    total_items = sum(b[-1].get('count', 0) for b in batches) if batches else 0
    total_batches = len(batches)
    new_batches = sum(1 for b in batches if b and b[0].get('count', 0) == 1)

    return {
        "period": f"{month_name.capitalize()} {datetime.now(WAT).year}",
        "total_items": total_items,
        "total_batches": total_batches,
        "new_batches": new_batches,
        "logs": logs[:30],  # AI only sees first 30
        "summary": f"{total_items} items in {total_batches} batch{'es' if total_batches != 1 else ''}."
    }
# --- TOOL 3: RECENT HISTORY ---
@tool
def get_production_history(limit: int = 5) -> List[Dict[str, Any]]:
    """Get last N production updates (max 200)."""
    limit = max(1, min(limit, 200))
    data = db_ref.child('events')\
                 .order_by_child('timestamp_epoch')\
                 .limit_to_last(limit)\
                 .get()

    if not data:
        return [{"message": "No history."}]

    logs = []
    for entry in data.values():
        ts = entry.get('timestamp_epoch')
        time_str = entry.get('timestamp_formatted', '??:??:??')
        if ts and isinstance(ts, (int, float)):
            time_str = datetime.fromtimestamp(ts).strftime('%d-%m-%y %H:%M:%S')
        logs.append({
            "time_recorded": time_str,
            "count": entry.get('count', '?'),
            "weight": entry.get('weight', '?'),
            "current": entry.get('current', '?')
        })
    logs.reverse()  # newest first
    return logs


# --- TOOL 4: TOTAL LOG COUNT ---
@tool
def get_total_log_count() -> Dict[str, Any]:
    """Count total events."""
    total = len(db_ref.child('events').get() or {})
    return {
        "total_logs": total,
        "message": f"{total} logs in database."
    }


# --- TOOL 5: ALL LOGS (FOR LARGE REQUESTS) ---
@tool
def get_all_event_logs(limit: int = 1000) -> Dict[str, Any]:
    """Get up to N logs (for 90, 100, 500 counts)."""
    limit = min(limit, 1000)
    query = db_ref.child('events')\
                  .order_by_child('timestamp_epoch')\
                  .limit_to_last(limit)
    data = query.get()

    if not data:
        return {"total_logs": 0, "logs": [], "message": "No logs."}

    logs = []
    for entry in data.values():
        ts = entry.get('timestamp_epoch')
        time_str = entry.get('timestamp_formatted', '??:??:??')
        if ts and isinstance(ts, (int, float)):
            time_str = datetime.fromtimestamp(ts).strftime('%d-%m-%y %H:%M:%S')
        logs.append({
            "time_recorded": time_str,
            "count": entry.get('count', '?'),
            "weight": entry.get('weight', '?'),
            "current": entry.get('current', '?')
        })
    logs.reverse()  # newest first

    return {
        "total_logs": len(logs),
        "logs": logs,
        "message": f"Retrieved {len(logs)} logs."
    }


# --- TOOL 6: COUNT + TIME ONLY (LIGHTWEIGHT) ---
@tool
def get_count_only(limit: int = 30) -> Dict[str, Any]:
    """Get last N counts with time only. For small requests."""
    limit = min(limit, 100)
    query = db_ref.child('events')\
                  .order_by_child('timestamp_epoch')\
                  .limit_to_last(limit)
    data = query.get()

    if not data:
        return {"message": "No logs."}

    logs = []
    for entry in data.values():
        time_str = entry.get('timestamp_formatted', '??:??:??')
        count = entry.get('count', '?')
        logs.append(f"• {time_str} → count: {count}")
    logs.reverse()

    return {
        "counts": "\n".join(logs),
        "total": len(logs),
        "message": f"Last {len(logs)} counts:"
    }


# ANALYZING TOOL
@tool
def analyze_production_insights(limit: int = 100) -> Dict:
    """Only: Total items + Number of batches. Uses data_sorter internally."""
    # === USE data_sorter DIRECTLY ===
    data = data_sorter(limit=limit, time_filter="all")
    logs = data.get("list", [])

    if not logs:
        return {"summary": "No data."}

    # === BATCH DETECTION ===
    batches = []
    current = []
    for log in logs:
        c = log.get('count', 0)
        if c == 1 and current and current[-1].get('count', 0) > 1:
            batches.append(current)
            current = [log]
        else:
            current.append(log)
    if current:
        batches.append(current)

    total_items = logs[-1].get('count', 0) if logs else 0
    total_batches = len(batches)

    return {
        "total_items": total_items,
        "total_batches": total_batches,
        "summary": f"{total_items} items in {total_batches} batch{'es' if total_batches != 1 else ''}."
    }


# data sorter — INTERNAL TOOL (NO @tool)
# data sorter — INTERNAL TOOL (NO @tool)
def data_sorter(limit: int = 1000, time_filter: str = 'all', month_name: str = None) -> Dict[str, Any]:
    limit = min(limit, 1000)
    
    query = db_ref.child('events').order_by_child('timestamp_epoch').limit_to_last(limit)
    data = query.get() or {}

    raw_logs = []
    for key, entry in data.items():
        ts = entry.get('timestamp_epoch')
        time_str = entry.get('timestamp_formatted', '??:??:??')
        if ts and isinstance(ts, (int, float)):
            # Convert epoch (WAT) to readable string
            time_str = datetime.fromtimestamp(ts, tz=WAT).strftime('%d-%m-%y %H:%M:%S')
        raw_logs.append({
            "timestamp_epoch": ts or 0,
            "timestamp_formatted": time_str,
            "count": entry.get('count', '?'),
            "current": entry.get('current', '?'),   # Added for future use
            "weight": entry.get('weight', '?')      # Added for future use
        })

    # Sort: oldest → newest
    sorted_logs = sorted(raw_logs, key=lambda x: x.get('timestamp_epoch', 0))

    # USE WAT TIME
    now = datetime.now(WAT)

    filtered = []
    if time_filter == 'yesterday':
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        filtered = [l for l in sorted_logs if start <= l.get('timestamp_epoch', 0) <= end]

    elif time_filter == 'hourly':
        hour_start = now.replace(minute=0, second=0, microsecond=0).timestamp()
        filtered = [l for l in sorted_logs if l.get('timestamp_epoch', 0) >= hour_start]

    elif time_filter == 'month':
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        filtered = [l for l in sorted_logs if l.get('timestamp_epoch', 0) >= month_start]

    elif time_filter == 'specific_month' and month_name:
        # === SUPPORT FOR OCTOBER, JANUARY, NOV, etc. ===
        month_name = month_name.strip().lower().capitalize()
        month_map = {name: idx for idx, name in enumerate(calendar.month_name) if name}
        month_map.update({name[:3]: idx for idx, name in enumerate(calendar.month_name) if name})
        
        if month_name not in month_map:
            return {"error": "Invalid month name.", "list": [], "total": 0, "filter": time_filter}

        target_month = month_map[month_name]
        current_year = now.year
        start_dt = datetime(current_year, target_month, 1, 0, 0, 0, tzinfo=WAT)
        last_day = calendar.monthrange(current_year, target_month)[1]
        end_dt = datetime(current_year, target_month, last_day, 23, 59, 59, tzinfo=WAT)

        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())

        filtered = [
            l for l in sorted_logs 
            if start_epoch <= l.get('timestamp_epoch', 0) <= end_epoch
        ]

    else:
        filtered = sorted_logs

    concise = [
        {
            'time': l.get('timestamp_formatted', '??:??:??'),
            'count': l.get('count', '?'),
            'current': l.get('current', '?'),
            'weight': l.get('weight', '?')
        }
        for l in filtered
    ]

    return {
        "list": concise,
        "total": len(concise),
        "filter": time_filter,
        "month": month_name  # Optional: for AI context
    }
## --- FUNCTION FOR DATA UPDATES (DISABLED) ---
def log_data_entry(db_ref, use_mock_data=False):
    """
    Updates are disabled to rely on real production data only.
    """
    logging.info("Production updates are disabled. Relying on existing data.")
    return  # Explicitly do nothing to avoid writes