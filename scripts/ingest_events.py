import os, requests, time, json
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()
TM_API = os.getenv("TICKETMASTER_API_KEY")

def fetch_ticketmaster(city, start_date, end_date, page=0):
    """Fetch events from Ticketmaster API"""
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "apikey": TM_API,
        "city": city,
        "startDateTime": start_date,
        "endDateTime": end_date,
        "size": 100,  # max allowed per request
        "page": page
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def extract_event(record):
    """Extract clean fields from Ticketmaster event JSON"""
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "description": record.get("info") or record.get("description") or "",
        "date": record.get("dates", {}).get("start", {}).get("dateTime"),
        "url": record.get("url"),
        "venue": record.get("_embedded", {}).get("venues", [{}])[0].get("name"),
        "city": record.get("_embedded", {}).get("venues", [{}])[0].get("city", {}).get("name"),
        "price_ranges": record.get("priceRanges"),
        "classification": record.get("classifications", [{}])[0].get("segment", {}).get("name")
    }

if __name__ == "__main__":
    out = []
    today = datetime.utcnow()
    future = today + timedelta(days=30)

    start_date = today.strftime("%Y-%m-%dT00:00:00Z")
    end_date = future.strftime("%Y-%m-%dT23:59:59Z")

    city = "New York"  # change as needed
    page = 0

    while True:
        res = fetch_ticketmaster(city, start_date, end_date, page=page)
        events = res.get("_embedded", {}).get("events", [])
        if not events:
            break
        for e in events:
            out.append(extract_event(e))
        page += 1
        print(f"Fetched page {page}, total {len(out)} events so far")
        if page > 4:  # limit to ~500 events while testing
            break
        time.sleep(0.5)

    os.makedirs("data", exist_ok=True)
    with open("data/events.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Saved {len(out)} events to data/events.json")
