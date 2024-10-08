'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
from dataset import *
from networks import *

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.utils.data.distributed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """ The main function of the test process for performance measurement. """
    net = Baseline_MNet_color().to(DEVICE)
    trained_weights = torch.load('../model/Baseline_MNet_color/08-12/model_30.pt',map_location=DEVICE)
    net.load_state_dict(trained_weights)

    # 아래 경로는 포함된 샘플(validation set)의 경로로, 실제 추론환경에서의 경로는 task.ipynb를 참고 바랍니다. 
    df = pd.read_csv('../Dataset/Fashion-How24_sub2_val.csv')
    val_dataset = ETRIDataset_color_test(df, base_path='../Dataset/val/')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    gt_list = np.array([])
    pred_list = np.array([])

    for j, sample in enumerate(val_dataloader):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
        out = net(sample)

        gt = np.array(sample['color_label'].cpu())
        gt_list = np.concatenate([gt_list, gt], axis=0)

        _, indx = out.max(1)
        pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)

    top_1, acsa, class_accuracies = get_class_metrics(gt_list, pred_list)
    print("------------------------------------------------------")
    print("Color: Top-1=%.5f, ACSA=%.5f" % (top_1, acsa))
    print("Class-wise Accuracies:")
    for i, acc in enumerate(class_accuracies):
        print(f"Class {i}: {acc:.5f}")
    print("------------------------------------------------------")
    return top_1


def get_test_metrics(y_true, y_pred, verbose=True):
    """
    :return: asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    return top_1, cs_accuracy.mean()

def get_class_metrics(y_true, y_pred, verbose=True):
    """
    :return: top_1, cs_accuracy.mean(), class_accuracies
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    class_accuracies = cs_accuracy.tolist()  # 클래스별 정확도 리스트로 변환

    return top_1, cs_accuracy.mean(), class_accuracies

if __name__ == '__main__':
    main()

