import copy
import pandas as pd
import numpy as np

from typing import List, Dict, Tuple, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def position_of_fashion_item(item: str) -> int:
    prefix = item[0:2]
    if prefix=='JK' or prefix=='JP' or prefix=='CT' or prefix=='CD' \
        or prefix=='VT' or item=='NONE-OUTER':
        idx = 0 
    elif prefix=='KN' or prefix=='SW' or prefix=='SH' or prefix=='BL' \
        or item=='NONE-TOP':
        idx = 1
    elif prefix=='SK' or prefix=='PT' or prefix=='OP' or item=='NONE-BOTTOM':
        idx = 2
    elif prefix=='SE' or item=='NONE-SHOES':
        idx = 3
    else:
        raise ValueError('{} do not exists.'.format(item))
    return idx

class MetaLoader:
    def __init__(self, path: str, swer: Any):
        self.path = path
        self.swer = swer

    def get_dataset(self) -> Tuple[Dict, Dict, Dict]:
        meta_dataset = self._load()
        img2id, id2img, img_similarity = self._mapping(meta_dataset)

        return img2id, id2img, img_similarity

    def _load(self) -> List:
        with open(self.path, encoding='euc-kr', mode='r') as f:
            lines = f.readlines()

        meta_dataset = []
        prev_name = ""
        for i in range(len(lines)):
            line = lines[i].split("\t")
            line = [l.strip() for l in line]
            
            # 새로운 아이템
            if line[0] != prev_name:
                meta_data = {"name": line[0], "category": line[1], "type": line[2], "characteristic": [line[3]], "description": [line[4]]}
                prev_name = line[0]
                meta_dataset.append(meta_data)
            
            # 기존과 동일한 아이템
            else:
                meta_data["characteristic"].append(line[3])
                meta_data["description"].append(line[4])
        
        return meta_dataset

    def _mapping(self, meta_dataset: List) -> Tuple[Dict, Dict, Dict]:
        img2desc = {meta_data["name"]: meta_data["description"] for meta_data in meta_dataset}
        
        # 카테고리별 분류(O, T, B, S)
        img_category = defaultdict(list)
        for img in img2desc.keys():
            c = position_of_fashion_item(img)
            img_category[c].append(img)

        # 임베딩
        img_vectors = defaultdict(dict)
        for c in img_category:
            sub_list = img_category[c]

            for img in sub_list:
                desc = img2desc[img]
                embed = [self.swer.get_sent_emb(d) for d in desc]
                vector = np.mean(embed, axis=0)
                img_vectors[c][img] = vector

        # 각 카테고리 별 유사도 계산
        img_similarity = defaultdict(list)            
        for c in img_vectors:
            vectors = list(img_vectors[c].values())
            src_array = np.array(vectors) # 카테고리 별 임베딩 벡터 배열

            for i in range(len(vectors)):
                trg_vector = vectors[i]
                trg_vector = np.expand_dims(trg_vector, axis=0)
                similarity = cosine_similarity(trg_vector, src_array)
                img_similarity[c].append(similarity[0])
                
        # 매핑
        img2id, id2img = defaultdict(dict), defaultdict(dict)
        for c in img_category:
            mapping = img_category[c]
            for idx, img in enumerate(mapping):
                img2id[c][img] = idx
                id2img[c][idx] = img

        # 더미 라벨 추가
        img2id[0]["NONE-OUTER"] = len(img2id[0])
        img2id[1]["NONE-TOP"] = len(img2id[1])
        img2id[2]["NONE-BOTTOM"] = len(img2id[2])
        img2id[3]["NONE-SHOES"] = len(img2id[3])

        id2img[0][len(id2img[0])] = "NONE-OUTER"
        id2img[1][len(id2img[1])] = "NONE-TOP"
        id2img[2][len(id2img[2])] = "NONE-BOTTOM"
        id2img[3][len(id2img[3])] = "NONE-SHOES"

        return img2id, id2img, img_similarity
    
class TrainLoader:
    def __init__(self, path: str):
        self.path = path

    def get_dataset(self) -> List:
        train_df = self._load()
        stories = self._split(train_df)

        dataset = []
        for i, story in enumerate(stories):
            desc, coordi, reward = self._extract_coordi(story)
            data = {"description": desc, "coordi": coordi, "reward": reward}
            dataset.append(data)

        return dataset

    def _load(self) -> pd.DataFrame:
        with open(self.path, encoding='euc-kr', mode='r') as f:
            lines = f.readlines()

        id_list, utter_list, desc_list, tag_list = [], [], [], []
        for i in range(len(lines)):
            line = lines[i].split('\t')
            line = [l.strip() for l in line]

            # 인덱스, 발화자, 설명, 태그 
            id_list.append(line[0])
            utter_list.append(line[1])
            desc_list.append(line[2])
            tag_list.append(line[3] if len(line) > 3 else "")

        return pd.DataFrame({"id": id_list, "utterance": utter_list, "description": desc_list, "tag": tag_list})

    def _split(self, df: pd.DataFrame) -> List:
        # INTRO TAG를 포함한 문장 인덱스 저장
        intro_ids = []
        for i in range(len(df)):
            if df.iloc[i]["tag"] == "INTRO":
                intro_ids.append(i)

        # INTRO TAG 기준으로 각 대화 구분
        stories = []
        for i in range(len(intro_ids) - 1):
            prev, cur = intro_ids[i], intro_ids[i+1]
            stories.append(df[prev:cur])
        
        # 마지막 남은 대화 추가
        stories.append(df[intro_ids[i+1]:])

        return stories

    def _add_dummy(self, item: Dict) -> Dict:
        for i, label in enumerate(["NONE-OUTER", "NONE-TOP", "NONE-BOTTOM", "NONE-SHOES"]):
            if i not in item:
                item[i] = label
        
        return item
    
    def _extract_coordi(self, story: pd.DataFrame) -> Tuple[List, List, List]:
        description, coordi, reward = [], [], []
        
        # 설명 부분만 추출
        for i in range(len(story)):
            if story.iloc[i]["utterance"] != "<AC>":
                description.append(story.iloc[i]["description"])

        # 추천 코디
        idx, clothes = 0, None
        while idx < len(story):
            line = story.iloc[idx]

            # USER TAG를 통해 만족도 체크
            if "USER" in line["tag"]:
                coordi.append(clothes)
                reward.append(line["tag"])
            
            # 코디 추천 발화인 경우
            if line["utterance"] == "<AC>":
                desc = line["description"]
                if clothes == None:
                    clothes = {position_of_fashion_item(c): c for c in desc.split(" ")}
                    clothes = self._add_dummy(clothes)
                else:
                    copied_clothes = copy.deepcopy(clothes)
                    for c in desc.split(" "):
                        copied_clothes[position_of_fashion_item(c)] = c
                    clothes = copied_clothes
            
            idx += 1    

        return description, coordi, reward