LLM : "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF" with llama_cpp(0.2.64) / GPT-4

Embedding model : "BAAI/bge-m3"

VectorDB : milvus(2.4.0)

Web demo : gradio(4.28)

RAG data : 한국어 대학 강의 데이터(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71627)

![캡처](https://github.com/Yusin-Lee/LLMwithRAG_BASE/assets/98385516/9d8aa95d-5453-41d4-b8e9-e6a229c3b47f)
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


** 같은 IP 위의 Docker container는 network로 묶어서 사용하면 편합니다.
** llama-cpp-python 설치 시, gpu 사용을 위해서 환경변수 설정이 필요합니다.
** config로 서버 올리는 것은 https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support 를 참고하세요.
