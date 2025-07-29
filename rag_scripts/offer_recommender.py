import faiss
import pickle
import google.generativeai as genai
import numpy as np

# First build index loader
def load_index():
    index = faiss.read_index("generated_files/faiss_index.bin")
    with open("generated_files/chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# This code will be used for embedding the prompts
def embed_query(API_KEY, query):
    genai.configure(api_key=API_KEY)
    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    )
    return np.array(response['embedding'], dtype='float32').reshape(1, -1)

# This code let's us find the similar chunks
def retrieve_similar_chunks(index, chunks, query_vector, top_k=5):
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

def format_chunks_as_offers(chunks):
    formatted = ""
    for i, chunk in enumerate(chunks, 1):
        formatted += f"--- შეთავაზება {i} ---\n{chunk.strip()}\n\n"
    return formatted

def generate_answer(API_KEY, query, context, history):
    genai.configure(api_key=API_KEY)
    history_text = "\n".join([f"მომხმარებელი: {q}\nასისტენტი: {a}" for q, a in history])
    prompt = f"""
წინა საუბარი:
{history_text}

ინსტრუქცია მოდელისთვის:

შენ ხარ ინტელექტუალური ასისტენტი, რომლის მიზანია მომხმარებლის მოთხოვნის საფუძველზე, მოძებნოს ყველაზე შესაფერისი შეთავაზებები და გასცეს სარწმუნო, დეტალური და მკაფიო რეკომენდაციები. ქვემოთ მოცემულია ერთზე მეტი შეთავაზება, თითოეული მათგანი მკაფიოდ გამოყოფილია სათაურით --- შეთავაზება N ---.
მნიშვნელობების აღქმის წესები:

    შეთავაზება წარმოადგენს კონკრეტულ ფასდაკლებას ან შეთავაზებას კომპანიებისგან და შეიცავს ინფორმაციას, როგორიცაა: სათაური, აღწერა, ბრენდები, ადგილმდებარეობა, ბენეფიტი, კატეგორია და საკონტაქტო დეტალები.

    "None" არ ნიშნავს რომ ინფორმაცია არ არსებობს – ბევრ შემთხვევაში შესაბამისი დეტალები მოცემულია აღწერაში ან სხვა ველებში. აუცილებელია ტექსტის სრულად გაანალიზება.

    ზოგიერთ შეთავაზებაში კონკრეტული ველები შეიძლება იყოს ცარიელი, მაგრამ იგივე ინფორმაცია შეიძლება აღწერილობაში იყოს ახსნილი (მაგ., საიტი ან ფასდაკლების კოდი).

    თუ ინფორმაცია სადმე გამოტოვებულია, ნუ გააკეთებ დაშვებას რომ ის არ არსებობს — მოძებნე იგი აღწერასა და სხვა ველებში.

პასუხის სტილი:

    გამოარჩიე მხოლოდ ის შეთავაზებები, რომლებიც ყველაზე მეტად შეესაბამება მომხმარებლის მოთხოვნას.

    აღწერე თითოეული შერჩეული შეთავაზება მოკლედ და გასაგებად.

    გამოიყენე მხოლოდ ის ინფორმაცია, რაც მოცემულია, მაგრამ ახსენე ყველაფერი რაც მომხმარებლის ინტერესს შეიძლება შეესაბამებოდეს.

    თუ შესაძლებელია, მიუთითე ქალაქი, ბრენდი, ფასდაკლების პროცენტი ან სპეციფიური კოდი, რომ მომხმარებელმა მარტივად ისარგებლოს შეთავაზებით.

    აუცილებლად უთხარი მომხმარებელს შეთავაზების url/ლინკი, რომ შეძლოს შეთავაზების ნახვა

    მიუთითე, რატომ სთავაზობ თითოეულ შეთავაზებას მომხმარებელს — დააკავშირე შეთავაზების მახასიათებლები მომხმარებლის მოთხოვნასთან.

მომხმარებლის მოთხოვნა: {query}

შესაძლო შეთავაზებები:
{context}
"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    #chat = model.start_chat(history=[])
    response = model.generate_content(prompt)
    #response = chat.send_message(prompt)
    return response.text

def rag_query(API_KEY, index, chunks, history):
    prompt = input("შენ: ")
    embedded_prompt = embed_query(API_KEY,prompt)
    similar_chunks = retrieve_similar_chunks(index, chunks,embedded_prompt, top_k = 5)
    context = format_chunks_as_offers(similar_chunks)
    answer = generate_answer(API_KEY, prompt, context, history)
    history.append((prompt, answer))
    return answer

def rag_system(API_KEY):
    index, chunks = load_index()
    history = []
    while True:
        answer = rag_query(API_KEY, index, chunks, history)
        print("ასისტენტი: "+answer)
        #print(len(history))

# Wrapper for API use
index_cache = None
chunks_cache = None

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

def chat_with_model(message, history=None):
    global index_cache, chunks_cache
    if index_cache is None or chunks_cache is None:
        index_cache, chunks_cache = load_index()
    if history is None:
        history = []
    embedded_prompt = embed_query(API_KEY, message)
    similar_chunks = retrieve_similar_chunks(index_cache, chunks_cache, embedded_prompt, top_k=5)
    context = format_chunks_as_offers(similar_chunks)
    answer = generate_answer(API_KEY, message, context, history)
    history.append((message, answer))
    return answer

def main():
    index, chunks = load_index()
    #random_prompt = "ბათუმში მივდივარ ივენთზე, მჭირდება შესაბამისი ტანსაცმელი და სასტუმრო"
    import os
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    history = []
    while True:
        answer = rag_query(API_KEY, index, chunks, history)
        print("ასისტენტი: "+answer)
        #print(len(history))

if __name__ == "__main__":
    main()
