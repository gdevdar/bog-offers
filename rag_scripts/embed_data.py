import json
from bs4 import BeautifulSoup
import google.generativeai as genai
import faiss
import numpy as np
import pickle

def create_chunk(offer):
    long_desc = offer["longDesc"]
    long_desc = BeautifulSoup(long_desc, 'html.parser')
    long_desc = long_desc.get_text(separator="\n", strip=True)
    chunk = \
        f"სათაური: '{offer['title']} - {offer['shortDesc']}'.\n\
აღწერა: '{long_desc}'.\n\
ბრენდები: '{offer['brandNames']}'.\n\
ბრენდების აღწერა: '{offer['brandDesc']}'.\n\
{offer['generatedCampaignDesc']}'.\n\
შეთავაზების დასრულებამდე დარჩენილი დღეები: '{offer['daysLeft']}'.\n\
შეთავაზების კატეგორია: '{offer['categoryDesc']}'.\n\
ქალაქი: '{offer['cityNames']}'.\n\
ბენეფიტი: '{offer['benefitName']} - {offer['benefText']} - კოდი {offer['productCodes']}'.\n\
საკონტაქტო ინფორმაცია: 'მისამართი - {offer['address']}, მობილურის ნომერი - {offer['phoneNumber']}, Instagram - {offer['instagram']}, Facebook - {offer['facebook']}, ვებსაიტი - {offer['website']}'.\n\
სხვა ინფორმაცია: 'solo Campaign - {offer['soloCampaign']}, activating Campaign - {offer['activatingCampaign']}, activation Period In Days - {offer['activationPeriodInDays']}, section Types - {offer['sectionTypes']}, segment Types - {offer['segmentTypes']}, is Offer Activated - {offer['isOfferActivated']}'.\n\
url/ლინკი: https://bankofgeorgia.ge/ka/offers-hub/details/{offer['campaignId']}\
"
    return chunk

def full_chunking(data):
    chunks = []
    for offer in data:
        chunk = create_chunk(offer)
        chunks.append(chunk)
    return chunks


def get_gemini_embeddings(texts, task_type, output_dimensionality=None):
    """
    Generates embeddings for a list of text chunks using gemini-embedding-001.

    Args:
        texts (list): A list of strings, where each string is a text chunk.
        task_type (str): The task type for which the embeddings will be used
                         ("RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for queries).
        output_dimensionality (int, optional): Desired output dimension.

    Returns:
        list: A list of embedding vectors (lists of floats).
    """
    try:
        embeddings = [] # რეალურად ემბედინგების მთავარი კოდი 50-57 და 61-62-ია.
        for text in texts:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type,
                output_dimensionality=output_dimensionality
            )
            
            # The response is a dictionary with 'embedding' key
            if isinstance(response, dict) and 'embedding' in response:
                embedding_values = response['embedding']
                embeddings.append(embedding_values)
            else:
                print(f"Unexpected response structure: {response}")
                continue
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings with Gemini API: {e}")
        return []

def embedding(API_KEY, chunks):
    genai.configure(api_key=API_KEY)
    TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
    EMBEDDING_DIMENSIONALITY = 768
    FAISS_INDEX_PATH = "generated_files/faiss_index.bin"
    CHUNKS_METADATA_PATH = "generated_files/chunks_metadata.pkl"

    # Get embeddings
    embeddings = get_gemini_embeddings(chunks, TASK_TYPE_DOCUMENT, EMBEDDING_DIMENSIONALITY)
    if not embeddings:
        print("No embeddings generated.")
        return

    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Build FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSIONALITY)
    index.add(embeddings_np)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata (the chunks)
    with open(CHUNKS_METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to {FAISS_INDEX_PATH} and metadata to {CHUNKS_METADATA_PATH}")

def full_embed(API_KEY):
    with open('generated_files/offers.json', 'r',encoding='utf-8') as f:
        data = json.load(f)
    chunks = full_chunking(data)
    embedding(API_KEY, chunks)

def test_embedding_api(API_KEY):
    """
    Test function to understand the API response structure
    """
    try:
        genai.configure(api_key=API_KEY)
        
        # Test with a simple text
        test_text = "Hello world"
        
        response = genai.embed_content(
            model="models/embedding-001",
            content=test_text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response content: {response}")
        
        if isinstance(response, dict):
            print(f"Response keys: {response.keys()}")
        else:
            print(f"Response attributes: {dir(response)}")
            
    except Exception as e:
        print(f"Test error: {e}")