import torch
import torchvision.models as MODEL
import torch.nn as nn
import aConfigration



def main():
    build_model()


def build_model():
    # alexnet = MODEL.alexnet(pretrained=True)
    resnet50 = MODEL.resnet50(pretrained=True)
    # alexnet.classifier = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(9216, 4096),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(4096, 2048),
    #     nn.ReLU(),
    #     nn.Linear(2048, aConfigration.LABEL_NUMS),
    # )
    # print(alexnet)

    resnet50.avgpool = nn.AdaptiveAvgPool2d(1)
    resnet50.fc = nn.Linear(2048,aConfigration.LABEL_NUMS)

    return resnet50


if __name__ == '__main__':
    main()



    # alexnet如下：
    #AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Dropout(p=0.5)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace)
#     (3): Dropout(p=0.5)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )