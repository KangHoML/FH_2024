import copy
import random
import numpy as np
from typing import List, Dict

def replace_coordi(source: Dict, num_coordi: int, img2id: Dict, id2img: Dict, img_similarity: Dict, top_k: int):
    # 더미가 아닌 카테고리 추출
    trg = [i for i in range(num_coordi) if "NONE" not in source[i]]

    # 카테고리 중 하나 랜덤 선택
    trg_id = random.choice(trg)
    trg_img = source[trg_id]

    # 유사도
    img_id = img2id[trg_id][trg_img]
    img_sim = img_similarity[trg_id][img_id]

    # 유사도 높은 상위 top_k개 중 하나 랜덤 선택 (0은 자기 자신이므로 제외)
    sorted_sim = np.argsort(img_sim)[::-1][1:]
    sel_id = sorted_sim[np.random.randint(top_k)]
    sel_img = id2img[trg_id][sel_id]

    # 교체
    add = copy.deepcopy(source)
    add[trg_id] = sel_img

    return add

class Preprocessor:
    def __init__(self, num_rank: int, num_coordi: int, top_k: int):
        self.num_rank = num_rank
        self.num_coordi = num_coordi
        self.top_k = top_k

    def __call__(self, train_dataset: List, img2id: Dict, id2img: Dict, img_similarity: Dict) -> List:
        dataset = []
        for i in range(len(train_dataset)):
            desc = train_dataset[i]["description"]
            coordi = train_dataset[i]["coordi"]
            reward = train_dataset[i]["reward"]

            if len(coordi) == 0: continue

            coordi_unique, reward_unique = [], []
            idx = len(coordi) - 1 # 최근 추천 코디부터 추가
            prev_coordi = None
            
            # 추천 코디 중복 없이 추가
            while idx >= 0:
                if prev_coordi == None:
                    coordi_unique.append(coordi[idx])
                    reward_unique.append(reward[idx])
                    prev_coordi = coordi[idx]
                else:
                    if not self._is_equal(prev_coordi, coordi[idx]):
                        coordi_unique.append(coordi[idx])
                        reward_unique.append(reward[idx])
                        prev_coordi = coordi[idx]
                
                idx -= 1
            
            # 추천된 코디 개수가 num_rank보다 많은 경우, 최근 추천 코디 3개 추가
            if len(coordi_unique) > self.num_rank:
                data = {"description": desc, "coordi": coordi_unique[:3], "reward": 0}
            
            # 추천된 코디 개수가 num_rank보다 작은 경우, 유사도를 기반으로 추가
            else:
                add_size = self.num_rank - len(coordi_unique)
                add_coordi = []
                for _ in range(add_size):
                    # 기존 코디 중 랜덤으로 하나 선택
                    source_idx = np.random.randint(len(coordi_unique))
                    source = coordi_unique[source_idx]

                    # 유사도 기반으로 대체된 코디 추가
                    add = replace_coordi(source, self.num_coordi, img2id, id2img, img_similarity, self.top_k)
                    add_coordi.append(add)    
                data = {"description": desc, "coordi": coordi_unique + add_coordi, "reward": 0}            
            dataset.append(data)
        
        return dataset            
    
    def _is_equal(self, a: Dict, b: Dict) -> bool:
        for key, value in a.items():
            if key not in b: return False # a_key != b_key
            if value != b[key]: return False # a_value != b_value
        
        return True

class Augmentation:
    def __init__(self, num_aug: int, num_rank: int, num_coordi: int, top_k: int):
        self.num_aug = num_aug
        self.num_rank = num_rank
        self.num_coordi = num_coordi
        self.top_k = top_k
    
    def __call__(self, train_dataset: List, img2id: Dict, id2img: Dict, img_similarity: Dict) -> List:
        dataset = []
        for data in train_dataset:
            dataset.append(data)

            desc = data["description"]
            coordi = data["coordi"]
            reward = data["reward"]

            for _ in range(self.num_aug):
                # 기존 코디 중 랜덤으로 하나 선택
                source_idx = np.random.randint(3)
                source = coordi[source_idx]

                # 유사도 기반으로 대체된 코디 추가
                add = replace_coordi(source, self.num_coordi, img2id, id2img, img_similarity, self.top_k)

                # 첫 번째 추천 코디는 유지
                if source_idx == 0:
                    data = {"desc": desc, "coordi": [coordi[0]] + [coordi[np.random.randint(1,3)]] + [add], "reward": reward}
                else:
                    data = {"desc": desc, "coordi": coordi[:source_idx] + [add] + coordi[source_idx + 1:], "reward": reward}
                dataset.append(data)
        
        # rank를 무작위로 선택하여 데이터 증강
        shuffled_dataset = []
        for data in dataset:
            shuffled_dataset.append(self._shuffle(data))
        
        return shuffled_dataset
    
    def _shuffle(self, data: Dict) -> Dict:
        desc = data["description"]
        coordi = data["coordi"]
        reward = data["reward"]

        ranks = list(range(self.num_rank))
        random.shuffle(ranks)

        coordi = [coordi[r] for r in ranks]
        reward = ranks.index(0)

        return {"description": desc, "coordi": coordi, "reward": reward}


        