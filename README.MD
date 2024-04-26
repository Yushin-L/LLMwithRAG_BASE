LLM : "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF" with llama_cpp(0.2.64) / GPT-4

Embedding model : "BAAI/bge-m3"

VectorDB : milvus(2.4.0)

Web demo : gradio(4.28)

RAG data : 한국어 대학 강의 데이터(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71627)

Data path
1. USER -> Gradio : Question

2. Gradio -> BGE-M3 server : Question

3. BGE-M3 server -> Gradio : Embedding vector

4. Gradio -> Milvus server : Query with embedding vector

5. Milvus server -> Gradio : Top-k documents related with Question

6. Gradio -> USER : Answer


Deploy path
1. create milvus standalone server with docker : "wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml"

2. 한국어 대학 강의 데이터(JSON->CSV) -> embedded by bge-m3 / stored in milvus collection

3. create OpenAI Compatible server with llama-cpp/Llama-3

4. create a Gradio server with BGE-M3 to create embedding vectors
