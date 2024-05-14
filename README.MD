# LLM with RAG BaseLine

```
LLM : "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF" with llama_cpp(0.2.64)

Embedding model : "BAAI/bge-m3"

VectorDB : milvus(2.4.0)

Web demo : gradio(4.28)

RAG data : 한국어 대학 강의 데이터
(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71627)
```
## Data path
![캡처](https://github.com/Yusin-Lee/LLMwithRAG_BASE/assets/98385516/9d8aa95d-5453-41d4-b8e9-e6a229c3b47f)
```
1. USER -> Gradio : Question

2. Gradio : Embedding question uing BGE-M3

3. Gradio -> Milvus server : Search context most related with question

4. Milvus server -> Gradio : context related with question

5. Gradio -> LLaMA3 server : Question with RAG context

6. LLaMA3 server -> Gradio : Answer

7. Gradio -> USER : Answer
```

## Deploy path
```
1. create milvus standalone server with docker : "wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml"

2. 한국어 대학 강의 데이터(JSON->CSV) -> embedded by bge-m3 -> create milvus collection & store the data in milvus collection

3. Create OpenAI Compatible server with llama-cpp & Llama-3

4. Create a Gradio server with BGE-M3 to create embedding vectors
```

** 같은 IP 위의 Docker container는 network로 묶어서 사용하면 편합니다.

** llama-cpp-python 설치 시, gpu 사용을 위해서 환경변수 설정이 필요할 수 있습니다.

** config로 llama-cpp서버 올리는 것은 https://llama-cpp-python.readthedocs.io/en/latest/server/#configuration-and-multi-model-support 를 참고하세요.
