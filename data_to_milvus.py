from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    utility, MilvusClient,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import CharacterTextSplitter
import random
from tqdm import tqdm

connections.connect("default", host="localhost", port="19530")
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR,
                is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="large_class", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="medium_class", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="org_text", dtype=DataType.VARCHAR, max_length=65535),
    # FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR), # use_fp16=True 시 sparse_vector error issue 발생
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                dim=1024),
]
schema = CollectionSchema(fields, "")
col_name = "lecture_data"
col = Collection(col_name, schema, consistency_level="Strong")
# sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
# col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "COSINE"}
col.create_index("dense_vector", dense_index)

df = pd.read_csv('../test.csv')
df = df[['video_name','large_class','medium_class','raw_text']]

# model = BGEM3EmbeddingFunction(use_fp16=False, device='cuda')
model = BGEM3FlagModel("BAAI/bge-m3",use_fp16=True, device='cuda')
dense_dim=1024
text_splitter = CharacterTextSplitter(separator='.', chunk_size=4096, chunk_overlap=200, length_function=len, is_separator_regex=False)
for index in tqdm(range(len(df))):
    text = df.raw_text[index]
    docs = text_splitter.split_text(text)
    embeddings = model.encode(docs,batch_size=1, max_length=5120)
    pk_list = []
    lc_list = []
    mc_list = []
    for i in range(len(embeddings['dense_vecs'])):
        pk_list.append(df.video_name[index] + '_' + str(i))
        lc_list.append(df.large_class[index])
        mc_list.append(df.medium_class[index])
    entities = [pk_list, lc_list, mc_list, docs, embeddings['dense_vecs']]
    col.insert(entities)
col.flush()
col.load()