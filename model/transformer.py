import torch
import math

from torch import nn
from dataset.constants import TOP5_LABEL_COLS
from transformers import BioGptTokenizer


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


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


class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dims,
        img_patch_size=8,
        bb_label_emb_dim=32,
        reflacx_label_cols=TOP5_LABEL_COLS,
        clinical_cat_dim=32,
        tf_head=4,
        tf_layer=4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dims = out_dims
        self.img_patch_size = img_patch_size

        self.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=img_patch_size,
            stride=img_patch_size,
        )
        self.bb_emb = nn.Embedding(
            num_embeddings=len(reflacx_label_cols) + 1,  # with background.
            embedding_dim=bb_label_emb_dim,
            padding_idx=0,
        )
        self.bb_fc = nn.Linear(4 + bb_label_emb_dim, hidden_dim)

        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.token_emb = TokenEmbedding(self.tokenizer.vocab_size, hidden_dim)

        self.bos_tensor = nn.Parameter(
            data=self.token_emb(torch.tensor([self.tokenizer.bos_token_id])).reshape(
                -1, 1
            ),
            requires_grad=False,
        )

        self.eos_tensor = nn.Parameter(
            self.token_emb(torch.tensor([self.tokenizer.eos_token_id])).reshape(-1, 1),
            requires_grad=False,
        )

        self.pad_tensor = nn.Parameter(
            self.token_emb(torch.tensor([self.tokenizer.pad_token_id])).reshape(-1, 1),
            requires_grad=False,
        )

        self.clinical_cat_emb = nn.ModuleDict(
            {
                "gender": nn.Embedding(2, clinical_cat_dim),
            }
        )

        self.chexpert_fc = nn.Linear(14, hidden_dim)
        self.clinical_fc = nn.Linear(33, hidden_dim)

        self.pos_enc = PositionalEncoding(hidden_dim, dropout=0)

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=tf_head,
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(layer, num_layers=tf_layer)
        self.modalities = [
            "xray",
            "bb_label",
            "clinical_data",
            "chexpert_label",
            "report",
        ]

        self.output_proj = nn.Linear(hidden_dim, out_dims)

    def _get_bb_inputs(self, bb_label):
        bb_inputs = []
        for d in bb_label:
            if len(d["boxes"]) > 0:
                emb_l = self.bb_emb(d["labels"])
                bbs = self.bb_fc(torch.concat([d["boxes"], emb_l], dim=1))
                bb_inputs.append(bbs)
            else:
                bb_inputs.append(None)
        return bb_inputs

    def _get_clinical_inputs(self, clinical):
        clinical_map = {
            "cat": {},
        }
        for c in clinical:
            for k in c["cat"].keys():
                if not k in clinical_map["cat"]:
                    clinical_map["cat"][k] = []
                clinical_map["cat"][k].append(c["cat"][k])

        clinical_map["num"] = [c["num"] for c in clinical]

        clinical_cat = self.clinical_cat_emb["gender"](
            torch.stack(clinical_map["cat"]["gender"])
        )
        clinical_num = torch.stack(clinical_map["num"])

        clinical_inputs = self.clinical_fc(
            torch.concat([clinical_cat, clinical_num], dim=1)
        ).unsqueeze(1)

        return clinical_inputs

    def _get_img_inputs(self, imgs):
        imgs = torch.stack(imgs)
        n, c, h, w = imgs.shape
        n_h = h // self.img_patch_size
        n_w = w // self.img_patch_size

        img_inputs = self.conv_proj(imgs)
        img_inputs = img_inputs.reshape(n, self.hidden_dim, n_h * n_w)

        return img_inputs

    def _get_chexpert_inputs(self, chexpert_label):
        return self.chexpert_fc(torch.stack(chexpert_label).float()).unsqueeze(1)

    def _get_report_inputs(self, report):
        # device = list(self.parameters())[0].device
        report_tks = self.tokenizer(report)
        return [
            self.token_emb(torch.tensor(n, device="cuda"))
            for n in report_tks["input_ids"]
        ]

    def _concat_inputs(
        self, img_inputs, bb_inputs, clinical_inputs, chexpert_inputs, report_inputs
    ):
        input_list = []
        for img, bb, c, chexpert, report in zip(
            img_inputs, bb_inputs, clinical_inputs, chexpert_inputs, report_inputs
        ):
            cat_list = [
                self.bos_tensor,
                img,
                self.eos_tensor,
                self.bos_tensor,
                c.reshape(self.hidden_dim, -1),
                self.eos_tensor,
                self.bos_tensor,
                chexpert.reshape(self.hidden_dim, -1),
                self.eos_tensor,
                self.bos_tensor,
                report.permute(1, 0),
                self.eos_tensor,
            ]

            if not bb is None and len(bb) > 0:
                cat_list += [
                    self.bos_tensor,
                    bb.permute(1, 0),
                    self.eos_tensor,
                ]

            cat_t = torch.concat(cat_list, dim=1)
            input_list.append(cat_t)

        padded_inputs = []
        padding_masks = []
        lengths = []
        max_len = max([i.shape[-1] for i in input_list])
        for i in input_list:
            lengths.append(i.shape[-1])
            if i.shape[-1] < max_len:
                padded_inputs.append(
                    torch.concat(
                        [i, self.pad_tensor.repeat(1, max_len - i.shape[-1])], dim=1
                    )
                )
                padding_masks.append(
                    [False] * i.shape[-1] + [True] * (max_len - i.shape[-1])
                )
            else:
                padded_inputs.append(i)
                padding_masks.append([False] * max_len)

        # need to get masks here.

        return (
            torch.stack(padded_inputs).permute(0, 2, 1),
            torch.tensor(padding_masks, device="cuda"),
            lengths,
        )

    def _get_batch_map(self, x):
        batch_map = {m: [d[m] for d in x] for m in self.modalities}
        return batch_map

    def forward(self, x):
        batch_map = self._get_batch_map(x)

        img_inputs = self._get_img_inputs(batch_map["xray"])
        bb_inputs = self._get_bb_inputs(batch_map["bb_label"])
        clinical_inputs = self._get_clinical_inputs(batch_map["clinical_data"])
        chexpert_inputs = self._get_chexpert_inputs(batch_map["chexpert_label"])
        report_inputs = self._get_report_inputs(batch_map["report"])

        cat_inputs, padding_masks, lengths = self._concat_inputs(
            img_inputs,
            bb_inputs,
            clinical_inputs,
            chexpert_inputs,
            report_inputs,
        )

        outputs = self.model(
            self.pos_enc(cat_inputs), src_key_padding_mask=padding_masks
        )
        # torch.arange(outputs.shape[-1])

        self.outputs = outputs
        self.lengths = lengths

        # raise StopIteration()

        # outputs = self.output_proj(outputs[:, torch.tensor(lengths).long(), :])
        outputs = self.output_proj(
            outputs[torch.arange(outputs.shape[0]), torch.tensor(lengths).long() - 1, :]
        )

        self.outputs = outputs
        return outputs
