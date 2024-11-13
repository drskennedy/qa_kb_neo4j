from llama_index.core import (
    Document,
    PropertyGraphIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import TokenTextSplitter
import requests
from bs4 import BeautifulSoup
import re
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import timeit
import datetime
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

import logging
import sys


def scrape_kbs():
    # TO-DO: customize this code function to scrape your KB system
    cookies = {
        'SESSIONID': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/116.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    }
    
    last_page = 0 
    kb_set = set()
    # find the KB solution numbers across the pages
    for page in range(last_page,last_page+1):
        print(f'finding KBs on page {page}...')
        data = {
            'cat': 'AppResponse',
            'offset': str(15*page),
        }
        response = requests.post(
            'https://kb.example.com/content/c_list.jsp',
            cookies=cookies,
            headers=headers,
            data=data,
        )
        kb_set.update(re.findall(r"\bS\d{5,6}\b", response.content.decode('utf-8')))
    docs = []
    url = 'https://kb.example.com/index?id='
    for kb in kb_set:
        print(f'scraping {kb}...')
        resp = requests.get(url+kb, headers=headers, cookies=cookies)
        soup = BeautifulSoup(resp.content, "html.parser")
        kb_doc = soup.find('h1', {'id' : 'contenttitle'}).text
        issue_div = soup.find('div', {'id' : 'ISSUE'})
        if issue_div:
            kb_doc += issue_div.text
        kb_doc += soup.find('div', {'id' : 'SOLUTION'}).text
        date_modify = soup.find_all('div', {'class' : 'row searchDetailInput'}).pop().text.split('\n')[-3].strip()
        metadata = {
            'source': kb,
            'created_at': date_modify,
        }
        docs += create_documents_from_text(kb_doc,metadata)
        print(docs[-1])

    print(f'Split done! Total docs => {len(docs)}')
    return docs

def create_documents_from_text(text, metadata=None, chunk_size=1024, chunk_overlap=20):
    """
    Create LlamaIndex Document objects from text using token chunking strategies.

    Args:
        text (str): Input text to be chunked
        chunk_size (int): Size of chunks (interpreted differently based on strategy)
        chunk_overlap (int): Overlap between chunks

    Returns:
        list: List of Document objects
    """
    node_parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents([Document(text=text)])
    documents = [Document(text=node.text,metadata=metadata) for node in nodes]
    return documents

llm = LlamaCPP(
    model_path='./models/mistral-7b-instruct-v0.3.Q2_K.gguf',
    temperature=0.1,
    max_new_tokens=256,
    context_window=12000,
    # kwargs to pass to __init__()
    model_kwargs={"n_gpu_layers": 1},
    verbose=False
)
embed_model = HuggingFaceEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

url = "bolt://localhost:7687"
username = "neo4j"
password = "password"
database = "neo4j"
graph_store = Neo4jPropertyGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

# load from Neo4j if graphs already done
try:
    kg_index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)
except:
    sys.exit(1)

'''
# Otherwise, uncomment this section to scrape, ingest and generate KGs
documents = scrape_kbs()
gstorage_context = StorageContext.from_defaults(graph_store=graph_store)
start = timeit.default_timer()
# ingest and generate KGs
kg_index = PropertyGraphIndex.from_documents(
    documents,
    storage_context=gstorage_context,
    property_graph_store=graph_store,
)
kg_gen_time = timeit.default_timer() - start # seconds
gstorage_context.persist()
kg_index.storage_context.persist(persist_dir="./storage_n4")
print(f'KG generation completed in: {datetime.timedelta(seconds=kg_gen_time)}')
'''
kg_keyword_query_engine = kg_index.as_query_engine(
    include_text=True,
    similarity_top_k=2,
)

print(f'{"/"*80}   ASKING QUESTION   {"/"*80}')
# read questions from file & iterate
with open('sample_qs.txt') as qfile:
    for i,query in enumerate(qfile):
        # KG query
        start = timeit.default_timer()
        response = kg_keyword_query_engine.query(query.rstrip())
        kg_qa_resp_time = timeit.default_timer() - start # seconds
        print(f'Query: {query}\nResponse: {response.response}\nTime: {kg_qa_resp_time:.2f}\n{"="*80}')
        # display all sources used by query engine
        for j,src in enumerate(response.source_nodes):
            print(f'Context: ### source {j} ###:\n{src.node.text}\nMetadata: {src.node.metadata}')
