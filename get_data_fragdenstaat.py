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
    num_entries = 1000 #Anzahl Einträge die gesammelt werden sollen
    sel_state = 92 #Bayern vgl. https://fragdenstaat.de/api/v1/jurisdiction/
    
    path_raw = local_dir / "fragdenstaat_messages_raw.json"
    path_csv = local_dir / "fragdenstaat_messages.csv"

    data = fetch_entries(base_url, num_entries, sel_state)
    save_raw_json(data, path_raw)
    extract_to_csv(data, path_csv)
  


# +
def fetch_entries(url, num_entries, state=None):
    """
    Fetches up to num_entries filtered entries from FragDenStaat API.
    Uses pagination via 'meta.next'.
    """

    results = []

    params = {
        "limit": 100  # maximale Seitengröße aus Performancegründen
    }

    # Filterung nach Bundesland, wenn state nicht übergeben wird, keine Filterung
    if state:
        params["jurisdiction"] = state

    # Kamagnen die ignoriert werden sollen
    # Vorabfilterung da sonst zu 90% der Daten nur anfragen über die Abiturprüfungsergebnisse diverser Jahre enthalten und keine sinvolle Themenprüfung stattfinden kann.
    excluded_campaigns = {
        "https://fragdenstaat.de/api/v1/campaign/15/", # Kampagne: "Verschlusssache Prüfung"
        "https://fragdenstaat.de/api/v1/campaign/5/" # Kampagne: "Frag sie Abi!"
    }

    print(f"Starte Download von bis zu {num_entries} Einträgen...")

    # 
    while url:
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"API Fehler {r.status_code} bei URL: {url}")
            break

        data = r.json()
        objects = data.get("objects", [])

        # Abbruch, wenn keine weiteren Daten in der API gefunden, bzw. auf der letzten Seite angekommen.
        if not objects:
            print("Keine weiteren Daten gefunden.")
            break

        # Eingelesene Daten nach Kampagnen filtern
        for entry in objects:
            if entry.get("campaign") not in excluded_campaigns:
                results.append(entry)

        # Abbruch, wenn Anzahl gewünschter Ergebnisse erreicht ist
        if len(results) >= num_entries:
            break

        # Auf die nächte API Seite wechseln, da Anfragen auf mehrere Siten aufgeteilt sind.
        url = data.get("meta", {}).get("next")
        params = None  # Parameter werden nur im ersten Request benötigt
        
        print(f"Current Entries: {len(results)} -> going to Next page")

        time.sleep(0.2)

    print(f"{len(results[:num_entries])} Einträge heruntergeladen.")
    return results[:num_entries]

def format_created(ts):
    if not ts:
        return None
    dt = datetime.fromisoformat(ts)
    return dt.strftime("%d.%m.%Y %H:%M")

def extract_to_csv(data, csv_path):
    """
    Extracts ID and description and writes them to a CSV file.
    """
    rows = [
        {
            "origin": "FragdenStaat",
            "id": d.get("id"),
            "created": format_created(d.get("created_at")),
            "title": (d.get("title") or "").strip(),
            "description": (d.get("description") or "").strip()
        }
        for d in data
    ]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"CSV gespeichert unter: {csv_path}")


def save_raw_json(data, json_path):
    """
    Saves the raw downloaded JSON list to disk.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"RAW JSON gespeichert unter: {json_path}")


# -

if __name__ == "__main__":
    main()
    

