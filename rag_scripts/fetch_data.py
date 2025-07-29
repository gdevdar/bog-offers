import requests
import json
import math
import time

def data_grab(n, page, retries = 3, delay = 2):
    url = f"https://bankofgeorgia.ge/api/bog-b/offers-hub/get-offers?pageInfo=true&pageNumber={page}&pageSize={n}&"
    for attempt in range(retries):
        try:
            response = requests.post(url)
            data = response.json()
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)

    raise Exception(f"Failed after {retries} retries")

def fetch_offers():
    num_of_items = data_grab(1,0)["result"]["totalItemCount"]
    k = 100 # Parameter determining the number of items per page per request
    loops = math.ceil(num_of_items/k)
    all_offers = []
    for i in range(loops):
        item = data_grab(k,i)["result"]["offers"]
        all_offers = all_offers + item
    with open("generated_files/offers.json", "w", encoding="utf-8") as f:
        json.dump(all_offers, f, ensure_ascii=False, indent=2)