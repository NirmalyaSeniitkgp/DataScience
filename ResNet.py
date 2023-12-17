import torch
import torch.nn as nn

class BasicBlockWithIdentityInputAddition(nn.Module):
    def __init__(self,in_channels, out_channels, stride, expansion=1):
        super(BasicBlockWithIdentityInputAddition, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out +=identity
        out = self.relu(out)
        return out


class BasicBlockWithDownsampledInputAddition(nn.Module):
    def __init__(self,in_channels, out_channels, stride, expansion=1):
        super(BasicBlockWithDownsampledInputAddition, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels*self.expansion,kernel_size=1,
                      stride=stride, bias=False),
                      nn.BatchNorm2d(self.out_channels*self.expansion))


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out +=identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, expansion=1, num_classes=1000):
        super(ResNet18,self).__init__()
        self.expansion=expansion
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,
                             stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(num_features=64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer10=BasicBlockWithIdentityInputAddition(in_channels=64,out_channels=64,stride=1)
        self.layer11=BasicBlockWithIdentityInputAddition(in_channels=64,out_channels=64,stride=1)
        self.layer20=BasicBlockWithDownsampledInputAddition(in_channels=64,out_channels=128,stride=2)
        self.layer21=BasicBlockWithIdentityInputAddition(in_channels=128,out_channels=128,stride=1)
        self.layer30=BasicBlockWithDownsampledInputAddition(in_channels=128,out_channels=256,stride=2)
        self.layer31=BasicBlockWithIdentityInputAddition(in_channels=256,out_channels=256,stride=1)
        self.layer40=BasicBlockWithDownsampledInputAddition(in_channels=256,out_channels=512,stride=2)
        self.layer41=BasicBlockWithIdentityInputAddition(in_channels=512,out_channels=512,stride=1)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer30(x)
        x = self.layer31(x)
        x = self.layer40(x)
        x = self.layer41(x)
        print('Dimension of the last convolutional feature map:', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


if __name__=='__main__':
    input = torch.rand(1,3,224,224)
    model = ResNet18()
    print(model)
    output = model(input)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(total_parameters)
    
