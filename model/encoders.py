"""
img -> resnet18 -> 128 tensor
bb -> transformer -> 128 tensor
clinical -> fc -> 128 tensor
chexpert -> fc -> 128 tensor
report -> transformer -> 128 tensor

640 tensor -> fc -> 5 outputs <-MSE-> (clinical features)
"""
from collections import OrderedDict
from typing import Dict, List
import torch
import math

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import BioGptTokenizer
import dataset.constants as r_constants


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, hidden_dim)

    def forward(self, x):
        return self.model(x)


class BBEncoder(nn.Module):
    def __init__(
        self, hidden_dim, nhead, nlayer, emb_dim, label_cols, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(layer, num_layers=nlayer)
        self.bb_emb = nn.Embedding(len(label_cols) + 1, emb_dim)
        self.sos = nn.Parameter(data=torch.randn((1, hidden_dim)), requires_grad=True)
        self.pad = nn.Parameter(data=torch.randn((1, hidden_dim)), requires_grad=True)
        self.bb_fc = nn.Linear(4 + emb_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=0)

    def forward(self, x: List[List[Dict]]):
        bb_inputs = []
        for b in x:
            if len(b["boxes"]) > 0:
                emb_l = self.bb_emb(b["labels"])
                bbs = self.bb_fc(torch.concat([b["boxes"], emb_l], dim=1))  # L, D
                bb_inputs.append(bbs)
            else:
                bb_inputs.append(None)

        padded_inputs = []  # N, L, D
        padding_masks = []
        lengths = []

        max_len = max([len(i) if not i is None else 0 for i in bb_inputs]) + 1

        for b in bb_inputs:
            if b is None:
                padded_inputs.append(
                    torch.concat(
                        [self.sos, self.pad.repeat(max_len - 1, 1)],
                        dim=0,
                    )
                )
                padding_masks.append([False] + [True] * (max_len - 1))
                lengths.append(1)
            elif b.shape[0] < max_len:
                padded_inputs.append(
                    torch.concat(
                        [self.sos, b, self.pad.repeat(max_len - (b.shape[0] + 1), 1)],
                        dim=0,
                    )
                )
                padding_masks.append(
                    [False] * b.shape[0] + [True] * (max_len - b.shape[0])
                )
                lengths.append(b.shape[0] + 1)
            else:
                padded_inputs.append(
                    torch.concat(
                        [self.sos, b],
                        dim=0,
                    )
                )
                padding_masks.append([False] * max_len)
                lengths.append(max_len)
        self.padded_inputs = padded_inputs
        self.lengths = lengths
        out = self.model(
            self.pos_enc(torch.stack(padded_inputs)),
            src_key_padding_mask=torch.tensor(
                padding_masks, device=next(self.parameters()).device
            ),
        )

        return out[torch.arange(out.shape[0]), torch.tensor(lengths).long() - 1, :]


class FCCatEncoder(nn.Module):
    def __init__(
        self,
        num_dim,
        hidden_dim,
        n_layers,
        cat_map={
            "gender": 2,
        },
        cat_dim=32,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cat_emb = nn.ModuleDict(
            {c: nn.Embedding(v, cat_dim) for c, v in cat_map.items()}
        )

        self.in_layer = nn.Linear(num_dim + (len(cat_map) * cat_dim), hidden_dim)
        self.model = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x):
        x_map = {
            "cat": {},
        }
        for c in x:
            for k in c["cat"].keys():
                if not k in x_map["cat"]:
                    x_map["cat"][k] = []
                x_map["cat"][k].append(c["cat"][k])

        x_map["num"] = [c["num"] for c in x]

        cat = OrderedDict(
            {
                c: c_emb(torch.stack(x_map["cat"][c]))
                for c, c_emb in self.cat_emb.items()
            }
        )
        cat = torch.concat([c for c in cat.values()], dim=1)
        # cat = self.cat_emb["gender"](torch.stack(x_map["cat"]["gender"]))
        num = torch.stack(x_map["num"])

        out = self.model(self.in_layer(torch.concat([cat, num], dim=1)))

        return out


class FCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            *[
                nn.Linear(input_dim, hidden_dim)
                if i == 0
                else nn.Linear(hidden_dim, hidden_dim)
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        return self.model(x)


class ReportEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.token_emb = TokenEmbedding(self.tokenizer.vocab_size, hidden_dim)
        self.pad_id = self.tokenizer.pad_token_id
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=0)

    def forward(self, x: List[str]):
        n = len(x)
        # lengths = [len(i) for i in x]
        x_tokens = self.tokenizer(
            x, return_tensors="pt", padding=True, return_length=True
        )
        self.x_tokens = x_tokens
        inputs = x_tokens["input_ids"].to(next(self.parameters()).device)
        lengths = x_tokens["length"]

        padding_masks = inputs == self.tokenizer.pad_token_id
        x_emb = self.token_emb(inputs)
        out = self.model(self.pos_enc(x_emb), src_key_padding_mask=padding_masks)
        self.out = out

        return out[torch.arange(n), torch.tensor(lengths).long() - 1, :]


class OneDCoreModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim,
        n_core_layers=5,
        bb_n_heads=4,
        bb_n_layers=4,
        bb_emb_dim=32,
        bb_label_cols=r_constants.TOP5_LABEL_COLS,
        clinical_n_layer=5,
        chespert_n_layers=5,
        report_n_heads=4,
        report_n_layers=4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.modalities = [
            "xray",
            "bb_label",
            "clinical_data",
            "chexpert_label",
            "report",
        ]
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.bb_enc = BBEncoder(
            hidden_dim=hidden_dim,
            nhead=bb_n_heads,
            nlayer=bb_n_layers,
            emb_dim=bb_emb_dim,
            label_cols=bb_label_cols,
        )
        self.clinical_enc = FCCatEncoder(
            num_dim=1,
            hidden_dim=hidden_dim,
            n_layers=clinical_n_layer,
        )
        self.chexpert_enc = FCEncoder(
            hidden_dim=hidden_dim,
            input_dim=14,
            n_layers=chespert_n_layers,
        )
        self.report_enc = ReportEncoder(
            hidden_dim=hidden_dim,
            n_heads=report_n_heads,
            n_layers=report_n_layers,
        )

        self.pre_core = nn.Linear(hidden_dim * 5, hidden_dim)

        self.core = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(n_core_layers)]
        )

        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _get_batch_map(self, x):
        batch_map = {m: [d[m] for d in x] for m in self.modalities}
        return batch_map

    def forward(self, x):
        batch_map = self._get_batch_map(x)

        img_out = self.img_enc(torch.stack(batch_map["xray"]))
        bb_out = self.bb_enc(batch_map["bb_label"])
        clinical_out = self.clinical_enc(batch_map["clinical_data"])
        chexpert_out = self.chexpert_enc(
            torch.stack(batch_map["chexpert_label"]).float()
        )
        report_out = self.report_enc(
            batch_map["report"],
        )

        cat_input = torch.concat(
            [img_out, bb_out, clinical_out, chexpert_out, report_out], dim=1
        )

        out = self.core(self.pre_core(cat_input))
        out = self.out_proj(out)

        return out
