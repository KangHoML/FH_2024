import copy
import random
import numpy as np

class Preprocessor:
    def __init__(self, num_rank: int, num_coordi: int):
        self.num_rank = num_rank
        self.num_coordi = num_coordi

    def preprocess(self, train_dataset, img2id, id2img, img_similarity):
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
                    # 기존 코디 중 하나 랜덤 선택
                    rand_idx = np.random.randint(len(coordi_unique))
                    ori = coordi_unique[rand_idx]

                    # 더미가 아닌 카테고리 추출
                    trg = [i for i in range(self.num_coordi) if "NONE" not in ori[i]]

                    # 카테고리 중 하나 랜덤 선택
                    trg_id = random.choice(trg)
                    trg_img = ori[trg_id]

                    # 유사도
                    img_id = img2id[trg_id][trg_img]
                    img_sim = img_similarity[trg_id][img_id]

                    # 유사도 높은 상위 50개 중 하나 랜덤 선택 (0은 자기 자신이므로 제외)
                    sorted_sim = np.argsort(img_sim)[::-1][1:]
                    sel_id = sorted_sim[np.random.randint(50)]
                    sel_img = id2img[trg_id][sel_id]

                    # 교체 후 추가
                    add = copy.deepcopy(ori)
                    add[trg_id] = sel_img
                    add_coordi.append(add)    
                data = {"description": desc, "coordi": coordi_unique + add_coordi, "reward": 0}            
            dataset.append(data)
        
        return dataset            
    
    def _is_equal(self, a, b):
        for key, value in a.items():
            if key not in b: return False # a_key != b_key
            if value != b[key]: return False # a_value != b_value
        
        return True
