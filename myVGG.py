import torch
import torch.nn as nn
import vgg


class MyVGG(nn.Module):

    def __init__(self, num_classes=40, feature_path=None):
        super(MyVGG, self).__init__()
        net = vgg.vgg16(pretrained=True, model_path=feature_path)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
                       
        )

    def forward(self, x):

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.sigmoid(x)      
        return x
