import torch.nn.functional as F
from torch import nn


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=6625,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        use_guide=False,
        **kwargs,
    ):
        super(CTCHead, self).__init__()
        self.use_guide = use_guide
        if use_guide:
            # Depthwise-separable Conv1d block:
            # DW Conv1d(k=5, groups=in_channels) → BN → ReLU → PW Conv1d(k=1) → BN
            self.guide_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    5,
                    padding=2,
                    groups=in_channels,
                    bias=True,
                ),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, in_channels, 1, bias=True),
                nn.BatchNorm1d(in_channels),
            )
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, labels=None):
        if self.use_guide:
            # x is [B, W, C], guide_layer is Conv1d on [B, C, W]
            x = x.permute(0, 2, 1)  # [B, C, W]
            x = self.guide_layer(x)
            x = x.permute(0, 2, 1)  # [B, W, C]

        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
