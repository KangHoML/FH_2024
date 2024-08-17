def etri_task1_submit():
    
    from dataset import ETRIDataset_emo_test
    from networks import ResExtractor, Baseline_ResNet_emo, Baseline_MNet_emo

    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix

    import torch
    import torch.utils.data
    import torch.utils.data.distributed

    from tqdm import tqdm

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Baseline_MNet_emo().to(DEVICE)
    trained_weights = torch.load('model/model_20.pt', map_location=DEVICE) # 자기 모델 경로를 지정합니다
    net.load_state_dict(trained_weights)
    df = pd.read_csv('/aif/Dataset/Fashion-How24_sub1_test.csv') # 제출 시 데이터 경로 준수. /aif/ 아래에 있습니다.
    val_dataset = ETRIDataset_emo_test(df, base_path='/aif/Dataset/test/') # 제출 시 데이터 경로 준수. /aif/ 아래에 있습니다.
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0) # 반드시 shuffle=False

    daily_pred_list = np.array([])
    gender_pred_list = np.array([])
    embel_pred_list = np.array([])

    for j, sample in tqdm(enumerate(val_dataloader)):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
        out_daily, out_gender, out_embel = net(sample)

        daily_pred = out_daily
        _, daily_indx = daily_pred.max(1)
        daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)

        gender_pred = out_gender
        _, gender_indx = gender_pred.max(1)
        gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)

        embel_pred = out_embel
        _, embel_indx = embel_pred.max(1)
        embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

    # 예측 결과를 dataframe으로 변환한 다음 함수의 결과로 return합니다.
    # 'image_name', 'daily', 'gender', 'embel'의 컬럼명과 image_name의 샘플 순서를 지켜주시기 바랍니다.
    # Baseline이 아닌 다른 모델을 사용하는 경우에도 같은 형식의 dataframe으로 return할 수 있도록 합니다.
    out = pd.DataFrame({'image_name':df['image_name'], 'daily':daily_pred_list, 'gender':gender_pred_list, 'embel':embel_pred_list})
   
    return out # 반드시 추론결과를 return

def submit():
    return etri_task1_submit

import aifactory.score as aif
import time
t = time.time()
if __name__ == "__main__":
    #-----------------------------------------------------#
    aif.submit(model_name="augument_sharpness",             # 본인의 모델명 입력(버전 관리에 용이하게끔 편의에 맞게 지정합니다)
               key="5db975da-7cc5-4b68-88b0-b2ad90576d42",                                   # 본인의 task key 입력
               func=submit                                 # 3.에서 wrapping한 submit function
               )
    #-----------------------------------------------------#
    print(time.time() - t)