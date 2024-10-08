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
import torch.utils.data
import numpy as np
from torchvision import transforms
from skimage import io, transform, color
import random

class BackGround(object):
    """Operator that resizes to the desired size while maintaining the ratio
            fills the remaining part with a black background

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size.
    """
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, landmarks, sub_landmarks=None):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]

            new_image = np.zeros((self.output_size, self.output_size, 3))

            if h > w:
                new_image[:,(112 - new_w//2):(112 - new_w//2 + new_w),:] = img
                landmarks = landmarks + [112 - new_w//2, 0]
            else:
                new_image[(112 - new_h//2):(112 - new_h//2 + new_h), :, :] = img
                landmarks = landmarks + [0, 112 - new_h//2]

            if sub_landmarks is not None:
                sub_landmarks = sub_landmarks * [new_w / w, new_h / h]
                if h > w:
                    sub_landmarks = sub_landmarks + [112 - new_w // 2, 0]
                else:
                    sub_landmarks = sub_landmarks + [0, 112 - new_h // 2]
                return new_image, landmarks, sub_landmarks
            else:
                return new_image, landmarks
        else:
            new_image = np.zeros((self.output_size, self.output_size, 3))
            if h > w:
                new_image[:,(112 - new_w//2):(112 - new_w//2 + new_w),:] = img
            else:
                new_image[(112 - new_h//2):(112 - new_h//2 + new_h), :, :] = img

            return new_image


# class BBoxCrop(object):
#     """ Operator that crops according to the given bounding box coordinates. """
    
#     def __call__(self, image, x_1, y_1, x_2, y_2):
#         h, w = image.shape[:2]

#         top = y_1
#         left = x_1
#         new_h = y_2 - y_1
#         new_w = x_2 - x_1

#         image = image[top: top + new_h,
#                       left: left + new_w]

#         return image


class ETRIDataset_color(torch.utils.data.Dataset):
    """ Dataset containing color category. """
    
    def __init__(self, df, base_path, target_per_class = 800):
        self.df = df
        self.base_path = base_path
        # self.bbox_crop = BBoxCrop()
        self.background = BackGround(227)
        # self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # for vis
        self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.to_pil = transforms.ToPILImage()

        self.label_cnt = 19
        self.target_per_class = target_per_class
        self.expanded_image_paths = []
        self.expanded_labels = []
        self.transforms = transforms.Compose([
            transforms.ToTensor() ,
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor() 
        ])
        self.expand_dataset()

    def expand_dataset(self):
        class_dict = {}
        self.label_cnt = 0
        
        # 클래스별로 이미지 분류
        for i, row in self.df.iterrows():
            label = row['Color']  # CSV에서 라벨이 있는 열 이름을 지정하세요
            image_path = row['image_name']  # CSV에서 이미지 경로가 있는 열 이름을 지정하세요

            if label not in class_dict:
                class_dict[label] = []
                self.label_cnt += 1
            class_dict[label].append(image_path)
                
        # 각 클래스에 대해 균등하게 50개로 확장
        for label, image in class_dict.items():
            num_samples = len(image)
            if num_samples >= self.target_per_class:
                # 이미 충분한 데이터가 있는 경우 랜덤하게 샘플링
                sampled_paths = random.sample(image, self.target_per_class)
                self.expanded_image_paths.extend(sampled_paths)
                self.expanded_labels.extend([label] * self.target_per_class)
            else:
                path = []
                repeat = self.target_per_class // num_samples
                path.extend(image * repeat)
                path.extend(random.sample(image, self.target_per_class % num_samples))  # 순환하면서 이미지 선택
                self.expanded_image_paths.extend(path)
                self.expanded_labels.extend([label] * self.target_per_class)

        assert len(self.expanded_image_paths) == len(self.expanded_labels), "Mismatch between image paths and labels!"


    def __getitem__(self, i):
        image_path = self.expanded_image_paths[i]
        color_label = self.expanded_labels[i]

        image = io.imread(self.base_path + image_path)
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)  

        image = self.background(image, None)
        image_ = image.copy()

        image_ = self.transforms(image_)
        # image_ = self.to_tensor(image_)
        image_ = self.normalize(image_)
        image_ = image_.float()

        ret = {}
        ret['ori_image'] = image
        ret['image'] = image_
        ret['color_label'] = color_label

        return ret

    def __len__(self):
        return self.label_cnt * self.target_per_class



class ETRIDataset_color_test(torch.utils.data.Dataset):
    """ Dataset containing color category. """
    
    def __init__(self, df, base_path):
        self.df = df
        self.base_path = base_path
        # self.bbox_crop = BBoxCrop()
        self.background = BackGround(224)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # for vis
        self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.to_pil = transforms.ToPILImage()


    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(self.base_path + sample['image_name'])
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)
        color_label = sample['Color']
        # # crop only if bbox info is available
        # try:
        #     bbox_xmin = sample['BBox_xmin']
        #     bbox_ymin = sample['BBox_ymin']
        #     bbox_xmax = sample['BBox_xmax']
        #     bbox_ymax = sample['BBox_ymax']
    
        #     image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        # except:
        #     pass
        image = self.background(image, None)

        image_ = image.copy()

        image_ = self.to_tensor(image_)
        image_ = self.normalize(image_)
        image_ = image_.float()

        ret = {}
        ret['ori_image'] = image
        ret['image'] = image_
        ret['color_label'] = color_label

        return ret

    def __len__(self):
        return len(self.df)
