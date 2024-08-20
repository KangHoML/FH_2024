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

Update: 2024.04.20.
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
            Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """
    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':                               #subtask2는 외부    Pretrained 모델 허용X
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class MnExtractor(nn.Module):
    """Feature extractor based on MobileNetv2 structure
    Args:
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, pretrained=True):
        super(MnExtractor, self).__init__()

        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.modules_front = list(self.net.children())[:-1]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class Baseline_ResNet_color(nn.Module):
    """ Classification network of color category based on ResNet18 structure. """
    def __init__(self):
        super(Baseline_ResNet_color, self).__init__()

        self.encoder = ResExtractor('50')
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.color_linear = nn.Linear(512, 19)

    def forward(self, x):
        """ Forward propagation with input 'x'. """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out = self.color_linear(flatten)

        return out


class Baseline_MNet_color(nn.Module):
    """ Classification network of emotion categories based on MobileNetv2 structure. """
    
    def __init__(self):
        super(Baseline_MNet_color, self).__init__()

        self.encoder = MnExtractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.color_linear = nn.Linear(1280, 19)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out = self.color_linear(flatten)

        return out


if __name__ == '__main__':
    pass




class ColorCNN(nn.Module):
    def __init__(self):
        super(ColorCNN, self).__init__()
        # First base network
        self.conv1_1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.norm1_1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.pool1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2_1 = nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2)
        self.norm2_1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        
        self.conv4_1 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        
        self.conv5_1 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.pool5_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second base network (same architecture)
        self.conv1_2 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.norm1_2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.pool1_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2_2 = nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2)
        self.norm2_2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3_2 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        
        self.conv4_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        
        self.conv5_2 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.pool5_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 19)  # 8 output classes for color recognition
        
    def forward(self, x):
        # Forward pass for the first base network
        x1 = self.conv1_1(x['image'])
        x1 = self.pool1_1(self.norm1_1(F.relu(x1)))
        x1 = self.pool2_1(self.norm2_1(F.relu(self.conv2_1(x1))))
        x1 = F.relu(self.conv3_1(x1))
        x1 = F.relu(self.conv4_1(x1))
        x1 = self.pool5_1(F.relu(self.conv5_1(x1)))
        
        # Forward pass for the second base network
        x2 = self.conv1_2(x['image'])
        x2 = self.pool1_2(self.norm1_2(F.relu(x2)))
        x2 = self.pool2_2(self.norm2_2(F.relu(self.conv2_2(x2))))
        x2 = F.relu(self.conv3_2(x2))
        x2 = F.relu(self.conv4_2(x2))
        x2 = self.pool5_2(F.relu(self.conv5_2(x2)))
        
        # Concatenate the outputs of the two base networks
        x_concat = torch.cat((x1, x2), dim=1)
        x_concat = x_concat.view(x_concat.size(0), -1)  # Flatten
        
        # Fully connected layers
        x_concat = F.relu(self.fc1(x_concat))
        x_concat = self.dropout1(x_concat)
        x_concat = F.relu(self.fc2(x_concat))
        x_concat = self.dropout2(x_concat)
        x_concat = self.fc3(x_concat)
        
        return x_concat

class ColorLinear(nn.Module):
    def __init__(self):
        super(ColorCNN, self).__init__()
        self.pointconv = nn.Conv2d(3,3, kernel_size=3, stride=1, padding=1)




    def forward(self,x):
        return x


if __name__ == "__main__":
    # Example usage
    model = ColorCNN()
    print(model)

    # Assuming an input image of size 227x227x3
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output)
