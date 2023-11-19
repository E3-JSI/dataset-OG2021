import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline


class MultilingualNER(nn.Module):
    def __init__(self, use_gpu: bool = False):
        """The Multilingual NER model"""
        super(MultilingualNER, self).__init__()
        self.model_name = "Babelscape/wikineural-multilingual-ner"

        device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )

    @torch.no_grad()
    def forward(self, text: str):
        ner_results = self.ner_pipeline(text)
        return ner_results
