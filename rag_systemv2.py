from rag_scripts import embed_datav2 as ed
from rag_scripts import offer_recommenderv2 as offrec
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    API_KEY = os.getenv("API_KEY")
    #ed.embed(API_KEY)
    offrec.rag_system(API_KEY)


if __name__ == "__main__":
    main()