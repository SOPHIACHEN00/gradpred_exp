import torch
from transformers import AutoModelForSequenceClassification, AdamW
from evaluate import load as load_metric

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_text_model(checkpoint, num_labels=2, device=DEVICE):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )
    return model.to(device)


def train_text_model(model, trainloader, epochs=1, lr=5e-5, device=DEVICE):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
