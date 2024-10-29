
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import requests

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
def chat_completion_request(client, messages, model="gpt-4o",
                            **kwargs):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def get_weather(city):
    api_key = "5714d13e8736c9573db1249716bd9c6d"
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    # Create the request URL
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"Current weather in {city}: {weather}, {temp}°C"
    else:
        return "Sorry, couldn't get the weather details."




class Copilot:
    def __init__(self):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en"
        )
        self.index = VectorStoreIndex.from_documents(docs, embed_model = embedding_model,
                                                     show_progress=True)
        self.retriever = self.index.as_retriever(
                        similarity_top_k=3
                        )
        
        self.system_prompt = """
            You are an expert on Columbia Business School HPC grid and your job is to answer questions 
            about this grid. Sometimes you will also give information about weather.
        """

    def ask(self, question, messages, openai_key=None):
        ### initialize the llm client
        self.llm_client = OpenAI(api_key = openai_key)
        if "weather" in question.lower():
            city = question.split("in")[-1].strip()  # Extract city name
            retrieved_info = get_weather(city)

        else:
            ### use the retriever to get the answer
            nodes = self.retriever.retrieve(question)
            ### make answer a string with "1. <>, 2. <>, 3. <>"
            retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])
            

        processed_query_prompt = """
            The user is asking a question: {question}

            The retrived information is: {retrieved_info}

            Please answer the question based on the retrieved information or the weather information. If the question is not related to Columbia Business School HPC grid or weather, 
            please tell the user and ask for a question related to Columbia Business School HPC grid or weather of some city.

            Please highlight the information with bold text and bullet points.
        """
        
        processed_query = processed_query_prompt.format(question=question, 
                                                        retrieved_info=retrieved_info)
        
        messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query}]
        response = chat_completion_request(self.llm_client, 
                                        messages = messages, 
                                        stream=True)
        
        return retrieved_info, response

if __name__ == "__main__":
    ### get openai key from user input
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = input("Please enter your OpenAI API Key (or set it as an environment variable OPENAI_API_KEY): ")
    copilot = Copilot()
    messages = []
    while True:
        question = input("Please ask a question: ")
        retrived_info, answer = copilot.ask(question, messages=messages, openai_key=openai_api_key)
        ### answer can be a generator or a string

        #print(retrived_info)
        if isinstance(answer, str):
            print(answer)
        else:
            answer_str = ""
            for chunk in answer:
                content = chunk.choices[0].delta.content
                if content:
                    answer_str += content
                    print(content, end="", flush=True)
            print()
            answer = answer_str

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
