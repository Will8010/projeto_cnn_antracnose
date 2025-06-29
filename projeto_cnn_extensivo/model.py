import torch.nn as nn
import torch.nn.functional as F

class ConvNeuNet(nn.Module):
    def __init__(self, size_kernel, num_stride, act_fu, size_denseLayer,
                 data_augmentation, batch_normalisation, padding, dropout_rate,
                 num_filters, classes=10, input_channels=3):
        super(ConvNeuNet, self).__init__()

        self.batch_norm = batch_normalisation
        width = height = 256
        af = lambda: nn.ReLU() if act_fu == 'relu' else nn.SELU() if act_fu == 'selu' else nn.Mish()

        def conv_block(in_c, out_c, k, i):
            nonlocal width, height
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=num_stride, padding=padding)
            width = (width - k[0] + 2 * padding) // num_stride + 1
            height = (height - k[1] + 2 * padding) // num_stride + 1
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            width = (width - 2) // 2 + 1
            height = (height - 2) // 2 + 1
            return nn.Sequential(
                conv,
                nn.BatchNorm2d(out_c) if batch_normalisation else nn.Identity(),
                af(),
                nn.Dropout(p=dropout_rate),
                pool
            )

        self.features = nn.Sequential(
            conv_block(input_channels, num_filters[0], size_kernel[0], 0),
            conv_block(num_filters[0], num_filters[1], size_kernel[1], 1),
            conv_block(num_filters[1], num_filters[2], size_kernel[2], 2),
            conv_block(num_filters[2], num_filters[3], size_kernel[3], 3),
            conv_block(num_filters[3], num_filters[4], size_kernel[4], 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[4] * width * height, size_denseLayer),
            af(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(size_denseLayer, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
