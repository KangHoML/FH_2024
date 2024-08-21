import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Dict

class FH2024Dataset(Dataset):
    def __init__(self, dataset: Dict):
        self.desc = dataset["description"]
        self.coordi = dataset["coordi"]
        self.rank = dataset["rank"] if "rank" in dataset else None

    def __len__(self):
        return len(self.desc)
    
    def __getitem__(self, idx: int) -> Any:
        desc = self.desc[idx]
        coordi = self.coordi[idx]
        
        if self.rank == None: 
            return desc, coordi
        else:
            rank = self.rank[idx]
            return desc, coordi, rank

def collate_fn(batch):
    desc_tensor, coordi_tensor, rank_tensor = [], [], []
    
    for data in batch:
        # PyTorch Dataset 클래스로부터 데이터 가져오기
        rank = None
        if len(data) == 3: desc, coordi, rank = data
        else: desc, coordi = data

        # 텐서로 변환할 리스트에 추가
        desc_tensor.append(desc)
        coordi_tensor.append(coordi)
        if rank != None: rank_tensor.append(rank)

    # 텐서로 변환
    desc_tensor = torch.tensor(np.array(desc_tensor), dtype=torch.float32)
    coordi_tensor = torch.tensor(np.array(coordi_tensor), dtype=torch.float32)
    # coordi_tensor = torch.tensor(coordi_tensor, dtype=torch.long)
    
    if rank == None:
        batch_tensor = {"description": desc_tensor, "coordi": coordi_tensor}
    else:
        rank_tensor = torch.tensor(rank_tensor, dtype=torch.long)
        batch_tensor = {"description": desc_tensor, "coordi": coordi_tensor, "rank": rank_tensor}

    return batch_tensor