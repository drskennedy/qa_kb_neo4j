# KB question-answering app using a local LLM and Neo4j-powered knowledge graphs from LlamaIndex

**Step-by-step guide on Medium**: [Using Neo4j features to Explain the Impact of Staged Addition to Knowledge Graph Construction](https://medium.com/ai-advances/impact-of-staged-addition-to-knowledge-graph-construction-and-querying-af944ea15329?sk=5074141de349a9a096faa6ecd3f023ef)
___
## Context
Knowledge graphs are expected to improve adaptation of LLMs to niche domains, as they capture semantics or relationships underlying entities from text documents unlike the approach used by the RAG method. To help with robustness and scalability, the knowledge graphs are stored in a Neo4j database.

In this project, we develop a QA system by scraping a web-based KB site and ingested using `LlamaIndex`'s module `PropertyGraphIndex` powered by a locally hosted LLM loaded using `llama-cpp-python` and backed by the graph-native Neo4j database. 
Its overall architecture is as shown below:
<br><br>
![System Architecture](/assets/architecture.png)
<br><br>
After injesting 75 knowledge-base articles, the resulting knowledge graphs looked like this:
<br><br>
![Knowledge Graph](/assets/graphs.png)
<br><br>
Neo4j cypher allows to query the database to understand how a specific context gets picked for a query. To find out the 1-hop nodes from entity "Scheduled report", for example, we could query using `MATCH (a)-[r]-(b) WHERE a.name =~ 'Scheduled report.*' RETURN r, a, b`. The following shows the returned graph:
<br><br>
![1-hop nodes](/assets/1-hop.png)
<br><br>
Check the linked Medium article for additional examples.
___
## How to Setup Python Virtual Environment to Support this QA system
- Create and activate the environment:
```
$ python3.11 -m venv kg_qa
$ source kg_qa/bin/activate
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download Mistral-7B-Instruct-v0.3.Q2_K.gguf from [MaziyarPanahi HF repo](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF) to directory `models`.

- Modify function `scrape_kbs` in `qa_neo4j_kb.py` to scrape your KB site:

- Start your Neo4j instance (to perform a fresh Neo4j install, check [repo qa-kg-neo4j](https://github.com/drskennedy/qa-kg-neo4j)):
```
$ $NEO4J_HOME/bin/neo4j console
```
- To start the QA app, run script `qa_neo4j_kb.py`:
```
$ python qa_neo4j_kb.py
```
___
## Quickstart
- To start the QA app, launch Terminal from the project directory and run the following command:
```
$ source bin/activate
$ python qa_neo4j_kb.py
```
- Here is a sample run:
```
$ python qa_neo4j_kb.py
Query: What is the correct BPF filter to use for capture VLAN-tagged packets for a specific subnet, 10.1.1.0/24?

Response:  not((10.1.1.0/24) or (vlan and (net 10.1.1.0/24)))
Time: 186.33
================================================================================
Query: In AppResponse, is it possible to decrypt any TLS traffic that supports Perfect Forward Secrecy using their private key?

Response:  No, it is not possible to decrypt any TLS traffic that supports Perfect Forward Secrecy using their private key in AppResponse as the AR11 appliance's decryption module requires the association of Client Random and Master Secret values which must be provided to AR11 from an external source through REST API.
Time: 588.11
================================================================================
Query: What shell command to use to obtain AppResponse SAML metadata?

Response:  npm_curl https://127.0.0.1/saml/metadata
Time: 398.03
================================================================================
```
___
## Key Libraries
- **LlamaIndex**: Framework for developing applications powered by LLM
- **llama-cpp-python**: Library to load GGUF-formatted LLM from a local directory
- **llama-index-graph-stores-neo4j**: Library to support Neo4j integration

___
## Files and Content
- `models`: Directory hosting the downloaded LLM in GGUF format
- `qa_neo4j_kb.py`: Main Python script to launch the QA app
- `sample_qs.txt`: Sample list of questions
- `requirements.txt`: List of Python dependencies (and version)
___

## References
- https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/#retrieval-and-querying
