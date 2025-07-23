from torch import nn

from ..necks.rnn import Im2Seq, SequenceEncoder
from .rec_ctc_head import CTCHead


class FCTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        if self.only_transpose:
            return x.permute([0, 2, 1])
        return self.fc(x.permute([0, 2, 1]))


class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop("head_list")

        self.gtc_head = "sar"
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "SARHead":
                pass
            elif name == "NRTRHead":
                pass
            elif name == "CTCHead":
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]["Neck"]
                encoder_type = neck_args.pop("name")
                self.ctc_encoder = SequenceEncoder(
                    in_channels=in_channels, encoder_type=encoder_type, **neck_args
                )
                # ctc head
                head_args = self.head_list[idx][name].get("Head", {})
                if head_args is None:
                    head_args = {}

                self.ctc_head = CTCHead(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args,
                )
            else:
                raise NotImplementedError(f"{name} is not supported in MultiHead yet")

    def forward(self, x, data=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)

        head_out = {}
        head_out["ctc"] = ctc_out
        head_out["res"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder

        # eval mode
        if not self.training:
            return ctc_out
        if self.gtc_head == "sar":
            sar_out = self.sar_head(x, data[1:])["res"]
            head_out["sar"] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), data[1:])["res"]
            head_out["nrtr"] = gtc_out
        return head_out
