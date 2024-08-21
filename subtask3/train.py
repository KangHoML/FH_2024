import os
import torch
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

# 전처리 관련 라이브러리
from utils.load import MetaLoader, DialogueTrainLoader
from utils.preprocess import Preprocessor, Augmentation
from utils.encoder import Encoder

# data 관련 라이브러리
from torch.utils.data import DataLoader
from data import FH2024Dataset, collate_fn

# 모델 관련 라이브러리
from net.tokenizer import SubWordEmbReaderUtil
from net.model import Model

# 훈련 관련 라이브러리
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.nn.utils.clip_grad import clip_grad_norm_
from si import update_omega, surrogate_loss

# device setting
cores = os.cpu_count()
torch.set_num_threads(cores)

# SI parameter
si_c = 0.1
epsilon = 0.001

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
parser = argparse.ArgumentParser(description="Fashion-How 2024 Trainer")

# -- seed
parser.add_argument('--seed', type=int, default=42)

# -- path
parser.add_argument('--swer_path', type=str, default='./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat')
parser.add_argument('--meta_path', type=str, default='subtask3/mdata.wst.txt.2023.08.23')
parser.add_argument('--train_diag_path', type=str, default='subtask3/task1.ddata.wst.txt')

# -- data
parser.add_argument('--num_aug', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.7)
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

# -- optimizer
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-3)

# -- train
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--max_grad_norm', type=float, default=40.0)
parser.add_argument('--save_freq', type=int, default=2)

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

def get_optimizer():
    # optimizer type 설정
    if args.optimizer == 'SGD':
        return SGD
    elif args.optimizer == 'Adam':
        return Adam
    elif args.optimizer == 'AdamW':
        return AdamW
    else:
        raise ValueError(args.optimizer)

def set_seed(seed):
    # seed 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def plot_loss(train_losses, ce_losses, si_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(ce_losses)+1), ce_losses, label='CE Loss', marker='s')
    plt.plot(range(1, len(si_losses)+1), si_losses, label='SI Loss', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title(f"model_{args.seed} Loss Graph")
    plt.legend()
    plt.grid()
    plt.savefig(f'./loss/model_{args.seed}.png')
        
def train():
    # seed
    set_seed(args.seed)

    # Device
    device = get_udevice()
    print(f"\nDevice: {device}")

    # Subword Embedding
    swer = SubWordEmbReaderUtil(args.swer_path)

    # Meta Data
    meta_loader = MetaLoader(path=args.meta_path, swer=swer)
    img2id, id2img, img_similarity = meta_loader.get_dataset()

    # Train Dialogue
    train_diag_loader = DialogueTrainLoader(path=args.train_diag_path)
    train_raw_dataset = train_diag_loader.get_dataset()

    # Preprocess
    preprocessor = Preprocessor(num_rank=3, num_coordi=4, threshold=args.threshold)
    train_dataset = preprocessor(train_raw_dataset, img2id, id2img, img_similarity)

    # Augmentation
    augmentation = Augmentation(num_aug=args.num_aug, num_rank=3, num_coordi=4, threshold=args.threshold)
    train_dataset = augmentation(train_dataset, img2id, id2img, img_similarity)

    # Encoder
    encoder = Encoder(swer=swer, img2id=img2id, num_coordi=4, mem_size=args.mem_size, meta_size=4)
    encoded_train_dataset = encoder(train_dataset)
    
    # Dataset & DataLoader
    train_dataset = FH2024Dataset(dataset=encoded_train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

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

    # Loss Function & Optimizer
    criterion = CrossEntropyLoss()
    optimizer_type = get_optimizer()
    optimizer = optimizer_type(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize SI parameter
    W = {}
    p_old = {}
    for n, p in net.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            W[n] = p.data.clone().zero_()
            p_old[n] = p.data.clone()
    
    # train
    train_losses, ce_losses, si_losses = [], [], []
    best_loss = float('inf')
    net.to(device)

    for epoch in range(args.epoch):
        net.train()
        train_loss, ce_loss_total, si_loss_total = 0, 0, 0

        for batch in tqdm(train_loader):
            desc = batch['description'].to(device)
            coordi = batch['coordi'].to(device)
            rank = batch['rank'].to(device)

            optimizer.zero_grad()

            logits = net(desc, coordi)
            loss_ce = criterion(logits, rank)

            if args.use_cl:
                loss_si = surrogate_loss(net)
                loss = loss_ce + si_c * loss_si
            else:
                loss_si = 0.0
                loss = loss_ce

            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            ce_loss_total += loss_ce.item()
            si_loss_total += loss_si

            # SI 관련 업데이트
            for n, p in net.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad * (p.detach() - p_old[n]))
                    p_old[n] = p.detach().clone()

        # 평균 loss 저장    
        train_loss /= len(train_loader)
        ce_loss_avg = ce_loss_total / len(train_loader)
        si_loss_avg = si_loss_total / len(train_loader) if args.use_cl else 0

        train_losses.append(train_loss)
        ce_losses.append(ce_loss_avg)
        si_losses.append(si_loss_avg)

        # 출력
        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"    Train Loss: {train_loss:.4f}, CE Loss: {ce_loss_avg:.4f}, SI Loss: {si_loss_avg:.4f}")

        # loss가 가장 적은 모델 저장
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(net.state_dict(), f'./pth/model_{args.seed}_final.pt')

        # ckpt에 따라 저장
        if epoch % args.save_freq == 0:
            torch.save(net.state_dict(), f'./pth/model_{args.seed}_epoch_{epoch}.pt')

    # SI 관련 최종 업데이트
    update_omega(net, device, W, epsilon)

    # 최종 모델 저장
    torch.save(net.state_dict(), f'./pth/model_final.pt')

    # Loss 그래프
    plot_loss(train_losses, ce_losses, si_losses)

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    os.makedirs("pth", exist_ok=True)
    os.makedirs("loss", exist_ok=True)
    train()