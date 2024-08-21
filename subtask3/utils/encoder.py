import numpy as np
from typing import Any, List, Dict

class Encoder:
    def __init__(self, swer: Any, img2id: Dict, num_coordi: int, mem_size: int):
        self.swer = swer
        self.img2id = img2id
        self.num_coordi = num_coordi
        self.mem_size = mem_size

    def __call__(self, dataset: List) -> Dict:
        desc_list, coordi_list, rank_list = [], [], []

        for data in dataset:
            # 대화 임베딩
            desc = data["description"]
            encoded_desc = [self.swer.get_sent_emb(sent).tolist() for sent in desc] 
            if len(encoded_desc) >= self.mem_size:
                encoded_desc = encoded_desc[:self.mem_size]
            else:
                encoded_desc = [np.zeros(self.swer.get_emb_size()).tolist() for _ in range(self.mem_size - len(encoded_desc))] + encoded_desc

            # 추천 코디를 인덱스 형태로 변환
            coordi = data["coordi"]
            encoded_coordi = []
            for c in coordi:
                imgs = [self.img2id[pos][c[pos]] for pos in range(self.num_coordi)]
                encoded_coordi.append(imgs)

            desc_list.append(encoded_desc)
            coordi_list.append(encoded_coordi)
            if "reward" in data: rank_list.append(data["reward"])

        # 딕셔너리 형태로 변환
        if "reward" in dataset[0]:
            encoded_dataset = {"description" : desc_list, "coordi" : coordi_list, "rank" : rank_list}
        else :
            encoded_dataset = {"description" : desc_list, "coordi" : coordi_list}

        return encoded_dataset