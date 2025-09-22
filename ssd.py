import torch
import torch.nn as nn
import torchvision.models as models
import math

def generate_default_boxes():
    fmap_sizes = [38, 19, 10, 5, 3, 1]
    scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
    aspect_ratios = [[2, .5], [2, 3, .5, 1./3], [2, 3, .5, 1./3], [2, 3, .5, 1./3], [2, .5], [2, .5]]

    boxes = []
    for k, fmap in enumerate(fmap_sizes):
        for i in range(fmap):
            for j in range(fmap):
                cx = (j + 0.5) / fmap
                cy = (i + 0.5) / fmap

                s = scales[k]
                boxes.append([cx, cy, s, s])

                s_next = math.sqrt(s * scales[min(k+1, len(scales)-1)])
                boxes.append([cx, cy, s_next, s_next])

                for ar in aspect_ratios[k]:
                    boxes.append([cx, cy, s*math.sqrt(ar), s/math.sqrt(ar)])
    return torch.tensor(boxes, dtype=torch.float32)

class SSD300(nn.Module):
    def __init__(self, num_classes=3):
        super(SSD300, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.base = nn.Sequential(*list(vgg.children())[:-1])

        self.extra_layers = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(1024, num_classes * 4, kernel_size=3, padding=1)
        self.loc_head = nn.Conv2d(1024, 4 * 4, kernel_size=3, padding=1)

        self.num_classes = num_classes
        self.default_boxes = generate_default_boxes()

    def forward(self, x):
        x = self.base(x)
        x = self.extra_layers(x)

        cls_out = self.cls_head(x)
        loc_out = self.loc_head(x)

        cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
        loc_out = loc_out.permute(0, 2, 3, 1).contiguous()

        return loc_out.view(loc_out.size(0), -1, 4), cls_out.view(cls_out.size(0), -1, self.num_classes)
