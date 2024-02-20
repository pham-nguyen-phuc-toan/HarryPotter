import os
import glob
import textwrap
import time

import langchain

# loaders
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

# prompts
from langchain import PromptTemplate, LLMChain

# vector stores
from langchain.vectorstores import FAISS

# models
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings

# retrievers
from langchain.chains import RetrievalQA

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print('langchain:', langchain.__version__)
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)

sorted(glob.glob('/kaggle/input/harry-potter-books-in-pdf-1-7/HP books/*'))

"""# CFG

- CFG class enables easy and organized experimentation
"""

class CFG:
    # LLMs
    model_name = 'llama2-13b-chat' # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
    temperature = 0,
    top_p = 0.95,
    repetition_penalty = 1.15

    # splitting
    split_chunk_size = 800
    split_overlap = 0

    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

    # similar passages
    k = 3

    # paths
    PDFs_path = '/kaggle/input/harry-potter-books-in-pdf-1-7/HP books/'
    Embeddings_path =  '/kaggle/input/faiss-hp-sentence-transformers'
    Persist_directory = './harry-potter-vectordb'

"""# Define model"""

def get_model(model = CFG.model_name):

    print('\nDownloading model: ', model, '\n\n')

    if model == 'wizardlm':
        model_repo = 'TheBloke/wizardLM-7B-HF'

        tokenizer = AutoTokenizer.from_pretrained(model_repo)

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            load_in_4bit = True,
            device_map = 'auto',
            torch_dtype = torch.float16,
            low_cpu_mem_usage = True
        )

        max_len = 1024

    elif model == 'llama2-7b-chat':
        model_repo = 'daryl149/llama-2-7b-chat-hf'

        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            load_in_4bit = True,
            device_map = 'auto',
            torch_dtype = torch.float16,
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )

        max_len = 2048

    elif model == 'llama2-13b-chat':
        model_repo = 'daryl149/llama-2-13b-chat-hf'

        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            load_in_4bit = True,
            device_map = 'auto',
            torch_dtype = torch.float16,
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )

        max_len = 2048 # 8192

    elif model == 'mistral-7B':
        model_repo = 'mistralai/Mistral-7B-v0.1'

        tokenizer = AutoTokenizer.from_pretrained(model_repo)

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            load_in_4bit = True,
            device_map = 'auto',
            torch_dtype = torch.float16,
            low_cpu_mem_usage = True,
        )

        max_len = 1024

    else:
        print("Not implemented model (tokenizer and backbone)")

    return tokenizer, model, max_len

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# tokenizer, model, max_len = get_model(model = CFG.model_name)

"""# 🤗 pipeline

- Hugging Face pipeline
"""

pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    pad_token_id = tokenizer.eos_token_id,
    max_length = max_len,
    temperature = CFG.temperature,
    top_p = CFG.top_p,
    repetition_penalty = CFG.repetition_penalty
)

llm = HuggingFacePipeline(pipeline = pipe)

"""# 🦜🔗 Langchain

- Multiple document retriever with LangChain
"""

CFG.model_name

"""# Loader

- [Directory loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory) for multiple files
- This step is not necessary if you are just loading the vector database
- This step is necessary if you are creating embeddings. In this case you need to:
    - load de PDF files
    - split into chunks
    - create embeddings
    - save the embeddings in a vector store
    - After that you can just load the saved embeddings to do similarity search with the user query, and then use the LLM to answer the question
    
You can comment out this section if you use the embeddings I already created.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# loader = DirectoryLoader(
#     CFG.PDFs_path,
#     glob="./*.pdf",
#     loader_cls=PyPDFLoader,
#     show_progress=True,
#     use_multithreading=True
# )
# 
# documents = loader.load()

"""# Splitter

- Splitting the text into chunks so its passages are easily searchable for similarity
- This step is also only necessary if you are creating the embeddings
- [RecursiveCharacterTextSplitter](https://python.langchain.com/en/latest/reference/modules/document_loaders.html?highlight=RecursiveCharacterTextSplitter#langchain.document_loaders.MWDumpLoader)
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CFG.split_chunk_size,
    chunk_overlap = CFG.split_overlap
)

texts = text_splitter.split_documents(documents)

print(f'We have created {len(texts)} chunks from {len(documents)} pages')

"""# Create Embeddings


- Embedd and store the texts in a Vector database (FAISS)
- [LangChain Vector Stores docs](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [FAISS - langchain](https://python.langchain.com/docs/integrations/vectorstores/faiss)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks - paper Aug/2019](https://arxiv.org/pdf/1908.10084.pdf)
- [This is a nice 4 minutes video about vector stores](https://www.youtube.com/watch?v=dN0lsF2cvm4)
- [Chroma - Persist and load the vector database](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html)

___

- If you use Chroma vector store it will take ~35 min to create embeddings
- If you use FAISS vector store on GPU it will take just ~3 min

___

We need to create the embeddings only once, and then we can just load the vector store and query the database using similarity search.

Loading the embeddings takes only a few seconds.

I uploaded the embeddings to a Kaggle Dataset so we just load it from [here](https://www.kaggle.com/datasets/hinepo/faiss-hp-sentence-transformers).
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# ### download embeddings model
# embeddings = HuggingFaceInstructEmbeddings(
#     model_name = CFG.embeddings_model_repo,
#     model_kwargs = {"device": "cuda"}
# )
# 
# ### create embeddings and DB
# vectordb = FAISS.from_documents(
#     documents = texts,
#     embedding = embeddings
# )
# 
# ### persist vector database
# vectordb.save_local("faiss_index_hp")

"""# Load vector database

- After saving the vector database, we just load it from the Kaggle Dataset I mentioned
- Obviously, the embeddings function to load the embeddings must be the same as the one used to create the embeddings
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# ### download embeddings model
# embeddings = HuggingFaceInstructEmbeddings(
#     model_name = CFG.embeddings_model_repo,
#     model_kwargs = {"device": "cuda"}
# )
# 
# ### load vector DB embeddings
# vectordb = FAISS.load_local(
#     CFG.Embeddings_path,
#     embeddings
# )

"""# Prompt Template

- Custom prompt
"""

prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

llm_chain = LLMChain(prompt=PROMPT, llm=llm)

"""# Retriever chain

- Retriever to retrieve relevant passages
- Chain to answer questions
- [RetrievalQA: Chain for question-answering](https://python.langchain.com/docs/modules/data_connection/retrievers/)
"""

retriever = vectordb.as_retriever(search_kwargs = {"k": CFG.k, "search_type" : "similarity"})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever,
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)

"""# Post-process outputs

- Format llm response
- Cite sources (PDFs)
- Change `width` parameter to format the output
"""

def wrap_text_preserve_newlines(text, width=700):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])

    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )

    ans = ans + '\n\nSources: \n' + sources_used
    return ans

def llm_ans(query):
    start = time.time()

    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str

"""# Ask questions

- Question Answering from multiple documents
- Invoke QA Chain
- Talk to your data
"""

CFG.model_name

query = "Which challenges does Harry face during the Triwizard Tournament?"
print(llm_ans(query))

query = "Who is Hagrid?"
print(llm_ans(query))

query = "Is Malfoy an ally of Voldemort?"
print(llm_ans(query))

query = "What are horcrux?"
print(llm_ans(query))

query = "Give me 5 examples of cool potions and explain what they do"
print(llm_ans(query))

! pip install streamlit

! pip show streamlit

import streamlit

"""# Gradio Chat UI

- **<font color='orange'>At the moment this part only works on Google Colab. Gradio and Kaggle started having compatibility issues recently.</font>**
- If you plan to use the interface, it is preferable to do so in Google Colab
- I'll leave this section commented out for now
- Chat UI prints below

___

- Create a chat UI with [Gradio](https://www.gradio.app/guides/quickstart)
- [ChatInterface docs](https://www.gradio.app/docs/chatinterface)
- The notebook should be running if you want to use the chat interface
"""

import locale
locale.getpreferredencoding = lambda: "UTF-8"

! pip install --upgrade gradio -qq
clear_output()

! pip install typing_extensions --upgrade

import gradio as gr

def predict(message, history):
    # output = message # debug mode

    output = str(llm_ans(message)).replace("\n", "<br/>")
    return output

demo = gr.ChatInterface(
    predict,
    title = f' Open-Source LLM ({CFG.model_name}) for Harry Potter Question Answering'
)

demo.queue()
demo.launch()

"""![image.png](attachment:413fe7a3-6534-45b5-b6e3-7fc86e982cf1.png)

![image.png](attachment:976f4bf4-7626-4d4a-b773-3eebd7e9f000.png)

# Conclusions

- Feel free to fork and optimize the code. Lots of things can be improved.

- Things I found had the most impact on models output quality in my experiments:
    - Prompt engineering
    - Bigger models
    - Other models families
    - Splitting: chunk size, overlap
    - Search: Similarity, MMR, k
    - Pipeline parameters (temperature, top_p, penalty)
    - Embeddings function
    - LLM parameters (max len)


- LangChain, Hugging Face and Gradio are awesome libs!

- **<font color='orange'>Upvote if you liked it or want me to keep updating it with new models and functionalities</font>**

- If you are interested in **<font color='blue'>Instruction Finetuning for LLMs</font>**, you might also want to check my other [notebook](https://www.kaggle.com/code/hinepo/llm-instruction-finetuning-wandb)

🦜🔗🤗

![image.png](attachment:68773819-4358-4ded-be3e-f1d275103171.png)
"""
