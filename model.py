import torch.nn as nn
import torch
import torchvision.models as models


class TransferModel(nn.Module):

    def __init__(self, model, freeze_backbone=True, mode='cbr'):
        assert mode == 'cbr', "Invalid Mode"

        super(TransferModel, self).__init__()
        self.freeze_backbone = freeze_backbone

        old_modules = [x for x in model.modules()]
        backbone_out_features = old_modules[-1].in_features
        self.backbone = nn.Sequential(*old_modules[1:-1])
        self.decoder = nn.Linear(backbone_out_features, 1)



    def forward(self, X):
        with torch.set_grad_enabled(not self.freeze_backbone):
            encoding = self.backbone(X)
        output = self.decoder(encoding)
        return output




def make_model(name):
    valid_names = ["cbr_large_tall", "cbr_large_wide", "cbr_small", "cbr_tiny", "resnet18", "resnet34", "resnet50"]
    assert name in valid_names, f"Invalid Model: {name}"

    if name == "cbr_large_tall":
        return make_cbr_large_tall()
    elif name == "cbr_large_wide":
        return make_cbr_large_wide()
    elif name == "cbr_small":
        return make_cbr_small()
    elif name == "cbr_tiny":
        return make_cbr_tiny()
    elif name == "resnet18":
        return make_resnet18()
    elif name == "resnet34":
        return make_resnet34()
    elif name == "resnet50":
        return make_resnet50()
    else:
        assert False, f"Invalid Model: {name}"


def make_resnet18():
    model = models.resnet18(pretrained=False)
    resnet_imagenet_to_resnet_chexpert(model)
    return model

def make_resnet34():
    model = models.resnet34(pretrained=False)
    resnet_imagenet_to_resnet_chexpert(model)
    return model

def make_resnet50():
    model = models.resnet50(pretrained=False)
    resnet_imagenet_to_resnet_chexpert(model)
    return model

def make_cbr_large_tall():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=7, padding=3), nn.BatchNorm2d(32) , nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(32, 64, kernel_size=7, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(64, 128, kernel_size=7, padding=3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(128, 256, kernel_size=7, padding=3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(256, 512, kernel_size=7, padding=3),nn.BatchNorm2d(512), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
        nn.Linear(512, 1)
    )

def make_cbr_large_wide():
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(64, 128, kernel_size=7, padding=3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(128, 256, kernel_size=7, padding=3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(256, 512, kernel_size=7, padding=3), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
        nn.Linear(512, 1)
    )

def make_cbr_small():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=7, padding=3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(32, 64, kernel_size=7, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(64, 128, kernel_size=7, padding=3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(128, 256, kernel_size=7, padding=3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
        nn.Linear(256, 1)
    )

def make_cbr_tiny():
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(128, 256, kernel_size=5, padding=2), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(256, 512, kernel_size=5, padding=2), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
        nn.Linear(512, 1)
    )

@torch.no_grad()
def resnet_imagenet_to_resnet_chexpert(model):
    old_first_conv = model.conv1
    new_first_conv = nn.Conv2d(1, old_first_conv.out_channels, kernel_size=old_first_conv.kernel_size, padding=old_first_conv.padding)
    new_first_conv.weight = nn.Parameter((old_first_conv.weight*torch.tensor([0.3, 0.59, 0.11]).view(1, 3, 1, 1)).sum(axis=1).unsqueeze(1))
    model.conv1 = new_first_conv

    old_fc = model.fc
    model.fc = nn.Linear(old_fc.in_features, 1)


if __name__ == '__main__':
    from torchsummary import summary

    '''
    input_shape = (1, 256, 256)
    for model_name in ["resnet18", "resnet34", "resnet50"]:
        print(model_name)
        model = make_model(model_name)
        summary(model, input_shape) # Params: 8,534,977 | 8,436,097 | 2,110,657 | 4,305,793 | 11,170,817 | 21,278,977 | 23,503,873
    '''
    x = make_model('cbr_large_wide')
    #breakpoint()
    data = torch.rand((8, 1, 256, 256))
    y = TransferModel(x)
    breakpoint()

