# Import necessary classes and functions from the llama_index and langchain libraries
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
# from llama_index import LLMPredictor, load_index_from_storage
from langchain.chat_models import ChatOpenAI

# Import the openai library and os module to set the API key
import openai
import os

# SECURITY ALERT: Never reveal your API keys directly in code. Use environment variables or other secure means.
# Here, we're setting the OpenAI API key both using an environment variable and directly (demonstration purposes only)
os.environ['OPENAI_API_KEY'] = 'YOU-API-KEY'
openai.api_key = 'YOU-API-KEY'

# Notify the user that the document loading process has begun
print("started the loading document process...")

# Read the data from the specified directory. Change './boiler_docs/' to your desired path.
documents = SimpleDirectoryReader('arxiv-paper').load_data()

# Initialize the LLMPredictor with the desired GPT-3.5-turbo model and temperature setting
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# Create a ServiceContext using the initialized predictor
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Notify the user that the indexing process has begun
print("started the indexing process...")

# Create an index using the loaded documents and the created service context
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

