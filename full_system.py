from rag_scripts import offer_recommender as offrec
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    API_KEY = os.getenv("API_KEY")
    a1 = input("Update offers data? (Y/N):\n")
    # scripts for updating the data
    if a1 == "Y":
        from rag_scripts import fetch_data as fd
        fd.fetch_offers()
        #API_KEY = input("Provide valid gemini API key: \n")
        from rag_scripts import embed_data as ed
        #ed.test_embedding_api(API_KEY)
        ed.full_embed(API_KEY)
    # scripts for running the RAG
    a2 = input("Start the rag system? (Y/N):\n")

    if a2 == "Y":
        offrec.rag_system(API_KEY)

if __name__ == "__main__":
    main()