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
import pandas as pd
from pathlib import Path
import time
import json


# +
def fetch_100_entries(base_url):
    """
    Fetches 100 request entries from FragDenStaat using pagination.
    Only available objects are collected until the limit is reached.
    """
    results = []
    page = 1

    print("Starte Download von bis zu 100 Einträgen...")

    while len(results) < 100:
        url = f"{base_url}?page={page}&format=json"

        r = requests.get(url)
        if r.status_code != 200:
            print(f"API Fehler {r.status_code} bei URL: {url}")
            break

        objects = r.json().get("objects", [])
        if not objects:
            print("Keine weiteren Daten gefunden.")
            break

        results.extend(objects)
        page += 1
        time.sleep(0.3)

    print(f"{len(results[:100])} Einträge heruntergeladen.")
    return results[:100]


def extract_to_csv(data, csv_path):
    """
    Extracts ID and description and writes them to a CSV file.
    """
    rows = [
        {
            "id": d.get("id"),
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



# +
def main():

    local_dir = Path("data/")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://fragdenstaat.de/api/v1/request/"
    
    path_raw = local_dir / "fragdenstaat_messages_raw.json"
    path_csv = local_dir / "fragdenstaat_messages.csv"

    data = fetch_100_entries(base_url)
    save_raw_json(data, path_raw)
    extract_to_csv(data, path_csv)


if __name__ == "__main__":
    main()
# -


