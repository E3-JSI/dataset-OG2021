import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer


class MultilingualLM(nn.Module):
    def __init__(
        self, model_type: str = "sbert", pooling_type: str = "mean", use_gpu=False
    ):
        """The Multilingual BERT model adapted to support multiple pooling options.
        Args:
            model_type (str): The language model used to generate the embeddings. Options:
                "mbert" - The multilingual BERT model.
                "distilbert" - The multilingual DistilBERT model.
                "xlmroberta" - The XLM-RoBERTa model.
                "sbert" - The multilingual Sentence-BERT model.
                Default to "sbert".
            pooling_type (str): The embedding pooling type. Options:
                "cls" - Representation with the [CLS] embedding.
                "max" - Representation with the positional maximum of the text
                        token embedding.
                "mean" - Representation with the average of the text token
                         embeddings.
                "sent" - Representation with the use of the sentence
                         transformer encoder.
                Default to "mean".
        """
        super(MultilingualLM, self).__init__()
        self.model_type = model_type
        self.pooling_type = pooling_type
        # set the maximum sequence length
        self.max_seq_length = 128 if self.model_type == "sbert" else 512

        if self.model_type == "mbert":
            model_name = "bert-base-multilingual-cased"
        elif self.model_type == "distilbert":
            model_name = "distilbert-base-multilingual-cased"
        elif self.model_type == "xlmroberta":
            model_name = "xlm-roberta-base"
        elif self.model_type == "sbert":
            model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        else:
            raise Exception(f"Unsupported model type: {self.model_type}")

        if self.pooling_type not in ["cls", "max", "mean"]:
            raise Exception(f"Unsupported pooling type: {self.pooling_type}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Generates the document embedding.
        Args:
            text (str): The text to be embedded.
        """

        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # get query embeddings
        embeds = self.model(**encodings)

        if self.pooling_type == "cls":
            pooling_func = cls_pooling
        elif self.pooling_type == "max":
            pooling_func = max_pooling
        elif self.pooling_type == "mean":
            pooling_func = mean_pooling

        embeds = pooling_func(embeds, encodings["attention_mask"])
        # normalize the vector before calculating
        embeds = f.normalize(embeds, p=2, dim=1)
        return embeds.cpu()


# ===============================================
# Helper Functions
# ===============================================


def cls_pooling(model_output, attn_mask):
    return model_output[0][:, 0]


def max_pooling(model_output, attn_mask):
    token_embeds = model_output[0]
    input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    token_embeds[input_mask_expanded == 0] = -1e9  # padding tokens to large negatives
    return torch.max(token_embeds, 1)[0]


def mean_pooling(model_output, attn_mask):
    token_embeds = model_output[0]
    input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    sum_embeddings = torch.sum(token_embeds * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
