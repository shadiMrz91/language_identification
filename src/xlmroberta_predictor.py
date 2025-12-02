import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
class XLMRobertaPredictor:
    def __init__(self, label_encoder, model_ckpt="papluca/xlm-roberta-base-language-detection", device=None):
        self.label_encoder = label_encoder
        self.model_ckpt = model_ckpt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(self.device)
        self.model.eval()

        self.our_languages = label_encoder.classes_
        self.xlmr_languages = list(self.model.config.id2label.values())
        self.lang_mapping = self._create_lang_mapping()

    def _create_lang_mapping(self):
        mapping = {}
        for our_lang in self.our_languages:
            if our_lang in self.xlmr_languages:
                mapping[our_lang] = our_lang
            else:
                our_lang_lower = our_lang.lower()
                found = False
                for xlmr_lang in self.xlmr_languages:
                    if xlmr_lang.lower() == our_lang_lower:
                        mapping[our_lang] = xlmr_lang
                        found = True
                        break
                if not found:
                    similar = [xlmr_lang for xlmr_lang in self.xlmr_languages
                               if our_lang_lower in xlmr_lang.lower() or xlmr_lang.lower() in our_lang_lower]
                    mapping[our_lang] = similar[0] if similar else self.our_languages[0]
        return mapping

    def predict(self, texts, batch_size=32):
        preds_all = []
        for i in tqdm(range(0, len(texts), batch_size), desc="XLM-R prediction"):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            for pred_idx in batch_preds:
                pred_lang_xlmr = self.model.config.id2label[pred_idx]
                mapped = next((our for our, xlmr in self.lang_mapping.items() if xlmr == pred_lang_xlmr), self.our_languages[0])
                preds_all.append(mapped)
        return self.label_encoder.transform(preds_all)