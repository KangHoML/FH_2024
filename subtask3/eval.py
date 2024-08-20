import os
import torch
import random
import argparse
import numpy as np

# 전처리 관련 라이브러리
from utils.load import MetaLoader, DialogueTestLoader
from utils.encoder import Encoder

# data 관련 라이브러리
from torch.utils.data import DataLoader
from data import FH2024Dataset, collate_fn

# 모델 관련 라이브러리
from net.tokenizer import SubWordEmbReaderUtil
from net.model import Model

# 평가 관련 라이브러리
from tqdm import tqdm
from scipy import stats

def str2bool(v):
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Argument Parser
parser = argparse.ArgumentParser(description="Fashion-How 2024 Evaluator")

# -- seed
parser.add_argument('--seed', type=int, default=42)

# -- path
parser.add_argument('--swer_path', type=str, default='./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat')
parser.add_argument('--meta_path', type=str, default='subtask3/mdata.wst.txt.2023.08.23')
parser.add_argument('--val_diag_path', type=str, default='subtask3/cl_eval_task1.wst.dev')

# -- data
parser.add_argument('--batch_size', type=int, default=16)

# -- model
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--key_size', type=int, default=300)
parser.add_argument('--mem_size', type=int, default=16)
parser.add_argument('--hops', type=int, default=3)
parser.add_argument('--eval_node', type=str, default='[6000,6000,6000,200][2000,2000]')
parser.add_argument('--drop_prob', type=float, default=0.0)

# -- bool type
parser.add_argument('--use_batch_norm', type=str2bool, default=False)
parser.add_argument('--use_dropout', type=str2bool, default=False)
parser.add_argument('--use_multimodal', type=str2bool, default=False)
parser.add_argument('--use_cl', type=str2bool, default=True)

def get_udevice():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()
    else:    
        device = torch.device('cpu')
    print('Using device: {}'.format(device))

    if torch.cuda.is_available():
        print('# of GPU: {}'.format(num_gpu))
    
    return device

def set_seed(seed):
    # seed 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def evaluate(args):
    # seed
    set_seed(args.seed)

    # Device
    device = get_udevice()
    print(f"\nDevice: {device}")

    # Subword Embedding
    swer = SubWordEmbReaderUtil(args.swer_path)

    # Meta Data
    meta_loader = MetaLoader(path=args.meta_path, swer=swer)
    img2id, _, _ = meta_loader.get_dataset()

    # Validation Dialogue
    val_diag_loader = DialogueTestLoader(path=args.val_diag_path, eval=True, num_rank=3)
    val_raw_dataset = val_diag_loader.get_dataset()
    label_ranks = [data.pop("reward") for data in  val_raw_dataset]

    # Encoder
    encoder = Encoder(swer=swer, img2id=img2id, num_coordi=4, mem_size=args.mem_size)
    encoded_val_dataset = encoder(val_raw_dataset)

    # Dataset & DataLoader
    val_dataset = FH2024Dataset(dataset=encoded_val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model Initialize
    item_size = [len(img2id[i]) for i in range(4)]
    net = Model(emb_size=swer.get_emb_size(), 
                key_size=args.key_size, 
                mem_size=args.mem_size,
                meta_size=4, 
                coordi_size=4, 
                num_rnk=3, 
                hops=args.hops, 
                eval_node=args.eval_node,
                item_size=item_size, 
                use_batch_norm=args.use_batch_norm, 
                use_dropout=args.use_dropout, 
                zero_prob=args.drop_prob,
                use_multimodal=args.use_multimodal,
                img_feat_size=4096)
    
    # checkpoint
    if args.ckpt is not None:
        ckpt_path = os.path.join('./pth', args.ckpt)
        net.load_state_dict(torch.load(ckpt_path))
        for n, p in net.named_parameters():
            n = n.replace('.', '__')
            net.register_buffer('{}_mean'.format(n), p.data.clone(), persistent=False)

    # evaluate
    net.to(device)
    pred_ranks = []

    with torch.no_grad():
        net.eval()
        for batch in tqdm(val_loader):
            desc = batch["description"].to(device)
            coordi = batch["coordi"].to(device)
            
            logits = net(desc, coordi)
            preds = torch.argsort(logits, -1, descending=True).detach().cpu().numpy()

            ranks = []
            for pred in preds:
                rank = [0, 0, 0] # 초기 순위
                for idx, r in enumerate(pred):
                    rank[r] = idx
                ranks.append(rank)

            pred_ranks.extend(ranks)
        
    # WKT 평가 지표 계산
    corr = _calculate_weighted_kendall_tau(pred_ranks, label_ranks)
    print(f"WKT: {corr: .4f}")

def _calculate_weighted_kendall_tau(pred, label):
    total_count = 0
    total_corr = 0.0

    for p, l in zip(pred, label):
        corr, _ = stats.weightedtau(p, l)
        total_corr += corr
        total_count += 1
    
    return (total_corr / total_count)