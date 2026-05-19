import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # Ingeniería inversa exacta de los índices de DeepLIIF
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect', bias=False),  # Índice 0
            nn.BatchNorm2d(in_features),  # Índice 1
            nn.ReLU(inplace=True),  # Índice 2
            nn.Dropout(0.5),  # Índice 3 (Capa oculta sin pesos que desplaza el índice)
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect', bias=False),  # Índice 4
            nn.BatchNorm2d(in_features)  # Índice 5
        )

    def forward(self, x):
        return x + self.conv_block(x)


class DeepLIIFResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(DeepLIIFResNetGenerator, self).__init__()

        # Capa Inicial
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Bloques Residuales (Ahora con la estructura exacta)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Capa de Salida
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)