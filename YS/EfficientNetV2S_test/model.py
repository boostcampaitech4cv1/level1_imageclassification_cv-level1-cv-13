import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Noise studey efficientnet_b4 custom module
class efficientnet_v2_s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = efficientnet.efficientnet_v2_s(weights=True)
        self.fc = nn.Linear(in_features=1000, out_features=18)
        
    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)

        return x


# Noise studey efficientnet_b5 custom module
class NsEfnB5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = timm.create_model("tf_efficientnet_b5_ns", pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
                                            nn.Linear(in_features=2048, out_features=1024, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=1024, out_features=512, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=256, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=256, out_features=num_classes),
                                            )
    def forward(self, x):
        x = self.efficientnet(x)

        return x


# Noise studey efficientnet_b7 custom module
class NsEfnB7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = timm.create_model("tf_efficientnet_b7_ns", pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
                                            nn.Linear(in_features=2560, out_features=1024, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=1024, out_features=512, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=256, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=256, out_features=num_classes),
                                            )
    def forward(self, x):
        x = self.efficientnet(x)

        return x




# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. ?????? ?????? ???????????? parameter ??? num_claases ??? ??????????????????.
        2. ????????? ?????? ??????????????? ????????? ????????????.
        3. ????????? output_dimension ??? num_classes ??? ??????????????????.
        """

    def forward(self, x):
        """
        1. ????????? ????????? ?????? ??????????????? forward propagation ??? ??????????????????
        2. ????????? ?????? output ??? return ????????????
        """
        return x

