def etri_task2_submit():

    from dataset import ETRIDataset_color_test
    from networks import Baseline_MNet_color    #ResExtractor,

    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix

    import torch
    import torch.utils.data
    import torch.utils.data.distributed

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ The main function of the test process for performance measurement. """
    net = Baseline_MNet_color().to(DEVICE)
    trained_weights = torch.load('./model/model_30.pt',map_location=DEVICE) # 자기 모델 경로를 지정합니다
    net.load_state_dict(trained_weights)

    df = pd.read_csv('/aif/Dataset/Fashion-How24_sub2_test.csv') # 제출 시 데이터 경로 준수. /aif/ 아래에 있습니다.
    val_dataset = ETRIDataset_color_test(df, base_path='/aif/Dataset/test/') # 제출 시 데이터 경로 준수. /aif/ 아래에 있습니다.
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0) # 반드시 shuffle=False

    pred_list = np.array([])

    for j, sample in enumerate(val_dataloader):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
        out = net(sample)

        _, indx = out.max(1)
        pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)
        
    # 예측 결과를 dataframe으로 변환한 다음 함수의 결과로 return합니다.
    # 'image_name', 'color'의 컬럼명과 image_name의 샘플 순서를 지켜주시기 바랍니다.
    # Baseline이 아닌 다른 모델을 사용하는 경우에도 같은 형식의 dataframe으로 return할 수 있도록 합니다.
    out = pd.DataFrame({'image_name':df['image_name'],'color':pred_list})
   
    return out # 반드시 추론결과를 return


def submit():
    return etri_task2_submit


import aifactory.score as aif
import time
t = time.time()
if __name__ == "__main__":
    #-----------------------------------------------------#
    aif.submit(model_name="etri-task2_test",               # 본인의 모델명 입력(버전 관리에 용이하게끔 편의에 맞게 지정합니다)
               key="ad6201d4-165b-45c5-9e19-3d33526d01ba",                                  # 본인의 task key 입력
               func=submit                            # 3.에서 wrapping한 submit function
               )
    #-----------------------------------------------------#
    print(time.time() - t)