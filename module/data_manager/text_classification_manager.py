import os

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from module.data_manager.manager import DataManager


class IMDB(DataManager):
    def __init__(self, checkpoint="distilbert-base-uncased", max_length=256, encode_batch_size=32):
        super().__init__(name="IMDB")
        self.task = "TextClassification"
        self.checkpoint = checkpoint
        self.max_length = int(max_length)
        self.encode_batch_size = int(encode_batch_size)
        self.tokenizer = None
        self.encoder = None
        self.encoder_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        cache_dir = os.path.join(project_root, "data", "raw", "hf_datasets")

        raw = load_dataset("imdb", cache_dir=cache_dir)
        if "unsupervised" in raw:
            del raw["unsupervised"]

        pool = concatenate_datasets([raw["train"], raw["test"]]).shuffle(seed=shuffle_seed)
        if nrows is not None:
            pool = pool.select(range(min(int(nrows), len(pool))))

        texts = np.array(pool["text"], dtype=object)
        labels = np.array(pool["label"], dtype=np.int64)

        n = len(texts)
        rng = np.random.RandomState(shuffle_seed)
        indices = rng.permutation(n)

        n_test = int(n * float(test_ratio))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_texts = texts[train_idx].tolist()
        test_texts = texts[test_idx].tolist()
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.encoder = AutoModel.from_pretrained(self.checkpoint).to(self.encoder_device)
        self.encoder.eval()

        self.X_train = self._encode_texts(train_texts)
        self.X_test = self._encode_texts(test_texts)
        self.y_train = torch.tensor(train_labels, dtype=torch.long)
        self.y_test = torch.tensor(test_labels, dtype=torch.long)

        self.X = torch.cat([self.X_train, self.X_test], dim=0)
        self.y = torch.cat([self.y_train, self.y_test], dim=0)

    @torch.no_grad()
    def _encode_texts(self, texts):
        features = []
        loader = DataLoader(texts, batch_size=self.encode_batch_size, shuffle=False)
        for batch_texts in loader:
            encoded = self.tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.encoder_device) for k, v in encoded.items()}
            out = self.encoder(**encoded)
            # DistilBERT: use [CLS]-position representation as fixed embedding.
            cls_emb = out.last_hidden_state[:, 0, :].detach().cpu()
            features.append(cls_emb)
        return torch.cat(features, dim=0).float()
