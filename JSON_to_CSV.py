import json
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

lc = glob.glob("/root/dev/JSONFILE/*")
mc = []
json_path = []

for path in glob.glob("/root/dev/JSONFILE/*") :
    for path_2 in glob.glob(f"{path}/*"):
        json_path += glob.glob(f"{path_2}/*")

        
def work(json_folder):
    raw_text = str()
    for name in glob.glob(f"{json_folder}/*.json"):
        with open(name, 'r') as f:
            json_data = json.load(f)
            raw_text += " " + json_data['06_transcription']['1_text']
    large_class = json_folder.split('/')[-3]
    medium_class = json_folder.split('/')[-2]
    video_name = json_folder.split('/')[-1]
    return {"raw_text":raw_text[1:], "large_class":large_class, "medium_class":medium_class, "video_name":video_name}

def main():
    all_data = []
    # ProcessPoolExecutor 초기화, 프로세스 수는 기본적으로 CPU 코어 수에 따라 결정됩니다.
    with ProcessPoolExecutor() as executor:
        # 모든 작업을 실행하고, as_completed로 각 작업의 완료를 기다립니다.
        futures = {executor.submit(work, json_folder): json_folder for json_folder in json_path}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()  # 각 작업의 결과를 가져옵니다.
            all_data.append(result)
    return all_data

all_data = main()
df = pd.DataFrame(all_data)
df.to_csv('./Lecture_text.csv',index=False)