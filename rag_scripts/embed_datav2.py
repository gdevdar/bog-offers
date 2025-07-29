import json
from bs4 import BeautifulSoup
from rag_scripts.embed_data import get_gemini_embeddings
import numpy as np
import faiss
import os
import pickle
import google.generativeai as genai

def create_smaller_chunks(offer):
    long_desc = offer["longDesc"]
    long_desc = BeautifulSoup(long_desc, 'html.parser')
    long_desc = long_desc.get_text(separator="\n", strip=True)

    mini_chunks = []

    description_chunk = f"""აღწერა: {long_desc}."""
    title_chunk =f"""სათაური: '{offer['title']} - {offer['shortDesc']}'.
    შეთავაზების კატეგორია: '{offer['categoryDesc']}'."""
    offer_period_chunk = f"""{offer['generatedCampaignDesc']}'
    შეთავაზების დასრულებამდე დარჩენილი დღეები: '{offer['daysLeft']}'."""
    brand_chunk = f"""ბრენდები: '{offer['brandNames']}'.
ბრენდების აღწერა: '{offer['brandDesc']}'.
ეს შეთავაზება ეხება: '{offer['brandNames']}'"""
    benefit_chunk = f"""ბენეფიტი: '{offer['benefitName']} - {offer['benefText']} - კოდი {offer['productCodes']}'."""
    other_stuff_chunk = f"""საკონტაქტო ინფორმაცია: 'მისამართი - {offer['address']}, მობილურის ნომერი - {offer['phoneNumber']}, Instagram - {offer['instagram']}, Facebook - {offer['facebook']}, ვებსაიტი - {offer['website']}'.\n\
სხვა ინფორმაცია: 'solo Campaign - {offer['soloCampaign']}, activating Campaign - {offer['activatingCampaign']}, activation Period In Days - {offer['activationPeriodInDays']}, section Types - {offer['sectionTypes']}, segment Types - {offer['segmentTypes']}, is Offer Activated - {offer['isOfferActivated']}'.\n\
url/ლინკი: https://bankofgeorgia.ge/ka/offers-hub/details/{offer['campaignId']}"""

    mini_chunks.append(description_chunk)
    mini_chunks.append(title_chunk)
    mini_chunks.append(offer_period_chunk)
    mini_chunks.append(brand_chunk)
    mini_chunks.append(benefit_chunk)
    mini_chunks.append(other_stuff_chunk)
    return mini_chunks

def chunk_prep():
    with open('generated_files/offers.json', 'r',encoding='utf-8') as f:
        data = json.load(f)
    chunks = []
    offer_ids = []
    for offer in data:
        small_chunks = create_smaller_chunks(offer)
        for chunk in small_chunks:
            chunks.append(chunk)
            offer_ids.append(offer["campaignId"])
    return chunks, offer_ids

def embed(API_KEY):
    genai.configure(api_key=API_KEY)
    chunks, offer_ids = chunk_prep()
    embeddings = get_gemini_embeddings(chunks, "RETRIEVAL_DOCUMENT")
    embeddings_np = np.array(embeddings, dtype='float32')

    FAISS_INDEX_PATH = "generated_files/faiss_indexv2.bin"
    CHUNKS_METADATA_PATH = "generated_files/chunks_metadatav2.pkl"
    OFFER_IDS_PATH = "generated_files/offer_idsv2.pkl"

    EMBEDDING_DIMENSIONALITY = 768  # Gemini's default

    index = faiss.IndexFlatL2(EMBEDDING_DIMENSIONALITY)
    index.add(embeddings_np)

    os.makedirs("generated_files", exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save chunk texts and offer IDs
    with open(CHUNKS_METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    with open(OFFER_IDS_PATH, "wb") as f:
        pickle.dump(offer_ids, f)

    print("✅ FAISS index and metadata saved.")

def main():
    API_KEY = os.getenv("API_KEY")
    print(API_KEY)
    embed(API_KEY)

if __name__ == "__main__":
    main()