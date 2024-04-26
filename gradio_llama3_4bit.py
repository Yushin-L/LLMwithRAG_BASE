import gradio as gr
import pandas as pd
from openai import OpenAI
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel
from datetime import datetime, timezone, timedelta
KST = timezone(timedelta(hours=9))

title = 'llama3-8b-gguf-4q-1'

llm_client = OpenAI(base_url="Local_llm_api_url", api_key="api_key")
rag_client = MilvusClient("milvus_server_url")
emb_model = BGEM3FlagModel("BAAI/bge-m3",use_fp16=True,device = "cuda")
search_params = {"metric_type": "COSINE", "params": {}}

# 질문지 : HAERAE-HUB/KMMLU 중 Health 영역
test = pd.read_csv('../질문지.csv')

# 데이터는 질문과 정답으로 구성되어 있으며 Gradio 화면에서 질문에 대한 정답을 확인하기 위해 처리
test["prompt"] = test.question + "\n\n" + "A : " + test.A + "\nB : " + test.B  + "\nC : " + test.C  + "\nD : " + test.D + "정답: " + test.answer.astype(str)

# Dropdown으로 질문지 선택하기 위해 리스트로 처리
prompts = [text for text in test.prompt]

def answering_with_chatcomplate(prompt):
        answer = prompt.split('정답: ')[1]
        prompt = prompt.split('정답: ')[0]

        # 질문 Embedding 처리
        embeddings = emb_model.encode([prompt],batch_size=1, max_length=512)['dense_vecs']

        # 질문과 관련된 문서 탐색
        res = rag_client.search(collection_name="lecture_data", data = embeddings, limit=1, output_fields=['org_text','medium_class'],search_params=search_params)
        rag_context = "\n\n".join([f'관련 문서 {index+1} : ' + result['entity']['org_text'] for index, result in enumerate(res[0])])
        rag_distance = "\n\n".join([f'관련 문서 {index+1} : ' + str(result['distance']) for index, result in enumerate(res[0])])

        # 질문 + 관련 문서로 프롬프트 작성
        prompt += "\n\n" + rag_context
        messages = [{"role": "system", "content": "당신은 헬스케어 전문가 AI입니다. 모든 질문은 헬스케어와 관련된 내용이며, 참고 문서를 제공하니, 질문과 참고문서를 잘 읽고 대답하세요."},{"role": "user", "content": prompt}]

        # LLM 서버에 요청 및 반환
        output = llm_client.chat.completions.create(
                model="llama-3-8b-it-q4",
                messages = messages)
        return prompt, output.choices[0].message.content, answer, rag_distance

demo = gr.Interface(
    fn=answering_with_chatcomplate,
    title = title,
    inputs = [gr.Dropdown(prompts,label='질문')],
    outputs = [gr.Textbox(label="Full Prompt"), gr.Textbox(label="Response"), gr.Textbox(label="GT"), gr.Textbox(label="Rag_distance")],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) # share=True