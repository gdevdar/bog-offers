import faiss
import pickle
import json
from rag_scripts.embed_data import create_chunk
from rag_scripts.offer_recommender import embed_query
from rag_scripts.offer_recommender import generate_answer

def load_index():
    index = faiss.read_index("generated_files/faiss_indexv2.bin")

    with open("generated_files/chunks_metadatav2.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("generated_files/offer_idsv2.pkl", "rb") as f:
        offer_ids = pickle.load(f)

    with open("generated_files/offers.json", "r", encoding="utf-8") as f:
        offers_list = json.load(f)
        offers_by_id = {offer["campaignId"]: offer for offer in offers_list}

    return index, chunks, offer_ids, offers_by_id

from collections import defaultdict

def retrieve_top_offers(index, chunks, offer_ids, offers_by_id, query_vector, top_k=10):
    distances, indices = index.search(query_vector, top_k * 6)  # retrieve more chunks to increase match quality

    offer_scores = defaultdict(float)
    offer_counts = defaultdict(int)
    for i, dist in zip(indices[0], distances[0]):
        offer_id = offer_ids[i]
        score = -dist  # closer = better
        offer_scores[offer_id] += score
        offer_counts[offer_id] += 1
        #offer_scores[offer_id] = max(offer_scores[offer_id], score)

    mean_scores = {oid: offer_scores[oid] / offer_counts[oid] for oid in offer_scores}
    top_offer_ids = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    #top_offer_ids = sorted(offer_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_offers = [offers_by_id[oid] for oid, _ in top_offer_ids]
    # matches = [chunk for chunk in chunks if "ბელუქსი" in chunk]
    # print(f"Chunks with ბელუქსი: {len(matches)}")

    return top_offers


def grab_top_offers(API_KEY, query, index, chunks, offer_ids, offers_by_id):

    
    query_vector = embed_query(API_KEY, query)
    top_offers = retrieve_top_offers(index, chunks, offer_ids, offers_by_id, query_vector)

    formatted = ""
    for i, offer in enumerate(top_offers, 1):
        formatted += f"--- შეთავაზება {i} ---\n{create_chunk(offer)}\n"
    return formatted

def rag_query(API_KEY, history, index, chunks, offer_ids, offers_by_id ):
    prompt = input("შენ: ")
    context = grab_top_offers(API_KEY, prompt, index, chunks, offer_ids, offers_by_id )
    answer = generate_answer(API_KEY, prompt, context, history)
    history.append((prompt, answer))
    return answer

def rag_system(API_KEY):
    index, chunks, offer_ids, offers_by_id = load_index()
    history = []
    while True:
        answer = rag_query(API_KEY, history, index, chunks, offer_ids, offers_by_id )
        print("ასისტენტი: "+answer)

# Wrapper for API use
index_cache = None
chunks_cache = None
offer_ids_cache = None
offers_by_id_cache = None

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

def chat_with_model(message, history=None):
    global index_cache, chunks_cache, offer_ids_cache, offers_by_id_cache
    if index_cache is None or chunks_cache is None or offer_ids_cache is None or offers_by_id_cache is None:
        index_cache, chunks_cache, offer_ids_cache, offers_by_id_cache = load_index()
    if history is None:
        history = []
    context = grab_top_offers(API_KEY, message, index_cache, chunks_cache, offer_ids_cache, offers_by_id_cache)
    answer = generate_answer(API_KEY, message, context, history)
    history.append((message, answer))
    return answer

def main():
    index, chunks, offer_ids, offers_by_id = load_index()
    import os
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    history = []
    while True:
        answer = rag_query(API_KEY, history, index, chunks, offer_ids, offers_by_id)
        print("ასისტენტი: "+answer)

if __name__ == "__main__":
    main()

