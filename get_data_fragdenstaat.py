# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (analyze_m_topics)
#     language: python
#     name: analyze-topics-venv
# ---

import requests
from datetime import datetime
import pandas as pd
from pathlib import Path
import time
import json
from pprint import pprint


def main():
    local_dir = Path("data/")
    local_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://fragdenstaat.de/api/v1/request/"
    num_entries = 1000
    sel_state = 92  # Bayern

    # Kamagnen die ignoriert werden sollen
    # Vorabfilterung da sonst zu 90% der Daten nur anfragen über die Abiturprüfungsergebnisse diverser Jahre enthalten und keine sinvolle Themenprüfung stattfinden kann.
    excluded_campaigns = {
        "https://fragdenstaat.de/api/v1/campaign/15/", # Kampagne: "Verschlusssache Prüfung"
        "https://fragdenstaat.de/api/v1/campaign/5/" # Kampagne: "Frag sie Abi!"
    }

    fsd_path_raw = local_dir / "fragdenstaat_messages_raw.json"
    yelp_path_raw = local_dir / "yelp_academic_dataset_review.json"
    path_csv = local_dir / "combined_reviews.csv"

    combined_rows = []

    # FragDenStaat laden oder herunterladen
    if not fsd_path_raw.exists():
        print(f"{fsd_path_raw} existiert noch nicht. Daten werden geladen...")
        fsd_data = fetch_fsd_entries(base_url, num_entries, sel_state, excluded_campaigns)
        save_raw_json(fsd_data, fsd_path_raw)
    else:
        print(f"{fsd_path_raw} existiert bereits. Daten werden geladen...")
        fsd_data = load_json(fsd_path_raw)

    combined_rows.extend(extract_rows("fsd", fsd_data))

    # Yelp Datensatz laden
    if not yelp_path_raw.exists():
        print(
            f"{yelp_path_raw} nicht gefunden.\n"
            "Bitte Yelp Dataset herunterladen, entpacken und unter data/ ablegen."
        )
    else:
        print(f"{yelp_path_raw} existiert. Daten werden extrahiert...")
        yelp_data = load_json(yelp_path_raw, num_entries)
        combined_rows.extend(extract_rows("yelp", yelp_data))

    
    # CSV Datei mit Daten von FragdenStaat und YELP unter data/ erzeugen
    df = pd.DataFrame(combined_rows)
    df.to_csv(path_csv, index=False, encoding="utf-8")
    
    print(f"CSV gespeichert unter: {path_csv}")
    print(f"{len(df)} Zeilen geschrieben.")


# +
def fetch_fsd_entries(url, num_entries, state=None, excluded_campaigns=None):
    """
    Fetches up to num_entries filtered entries from FragDenStaat API.
    Uses pagination via 'meta.next'.
    """

    results = []

    params = {
        "limit": 50 
    }

    # robust: None → leeres Set
    excluded_campaigns = set(excluded_campaigns or [])

    if state:
        params["jurisdiction"] = state

    print(f"Starte Download von bis zu {num_entries} Einträgen...")

    while url:
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"API Fehler {r.status_code} bei URL: {url}")
            break

        data = r.json()
        objects = data.get("objects", [])

        if not objects:
            print("Keine weiteren Daten gefunden.")
            break

        for entry in objects:
            campaign = entry.get("campaign")

            # funktioniert jetzt auch bei leerem excluded_campaigns
            if campaign not in excluded_campaigns:
                results.append(entry)

        if len(results) >= num_entries:
            break

        url = data.get("meta", {}).get("next")
        params = None

        print(f"Current Entries: {len(results)} -> going to Next page")

        time.sleep(0.2)

    print(f"{len(results[:num_entries])} Einträge heruntergeladen.")
    return results[:num_entries]

def save_raw_json(data, json_path):
    """
    Saves the raw downloaded JSON list to disk.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"RAW JSON gespeichert unter: {json_path}")

    
def format_created(ts):
    if not ts:
        return None

    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return ts

def load_json(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            return data[:limit] if limit is not None else data

        results = []
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            line = line.strip()
            if line:
                results.append(json.loads(line))

        return results

def extract_rows(origin, data):
    mapping = {
        "fsd": ("FragdenStaat", "id", "created_at", "description"),
        "yelp": ("YELP", "review_id", "date", "text"),
    }

    try:
        origin_name, id_f, created_f, text_f = mapping[origin]
    except KeyError:
        raise ValueError(f"Unbekannter origin-Wert: {origin}")

    return [
        {
            "origin": origin_name,
            "id": d.get(id_f),
            "created": format_created(d.get(created_f)),
            "review": (d.get(text_f) or "").strip(),
        }
        for d in data
    ]



# -

if __name__ == "__main__":
    main()


