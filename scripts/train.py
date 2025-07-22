from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datasets import Dataset
# from torch.utils.data import DataLoader, Dataset
# import accelerator

CHECKPOINT_DIR = Path("./EmbeddingModel/checkpoints_v2")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# class AudioDataset(Dataset):
#     def __init__(self, dataset, size:int = 5000):
#         self.dataset = [data for data in tqdm(dataset.take(size), desc="Loading Dataset", total=size)]
#         self.size = size

#     def __len__(self):
#         return self.size

#     def __getitem__(oself, idx):
#         data = self.dataset[idx]
#         data = {"audio": data["audio"]["array"], "text": data["text"]}
#         return data


class AudioEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        
        model_name = "openai/whisper-large"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).get_encoder()
        self.embedding_head = torch.nn.Linear(self.model.config.hidden_size, embed_dim)
        self.freeze_backbone()

    def forward(self, features):
        
        # features = self.feature_extractor(
           
        # ).to(self.model.device)
        
        features = self.model(**features).last_hidden_state
        embeddings = self.embedding_head(features)
        # Average pooling over sequence dimension
        embeddings = embeddings.mean(dim=1)
        return embeddings

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False


class TextEncoder(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        self.text_model = text_model

    def forward(self, model_inputs: dict):
        attention_mask = model_inputs["attention_mask"]
        token_embeddings = self.text_model(**model_inputs).last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum pooling with attention mask weighting
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class AudioTextModel(torch.nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder()
        self.temperature = torch.nn.Parameter(torch.tensor(0.07))
        
    def forward(self, audio, text):
        audio_embeddings = self.audio_encoder(audio)
        text_embeddings = self.text_encoder(text)
        
        
        audio_embeddings_norm = F.normalize(audio_embeddings, p=2, dim=1)
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=1)
        
        
        similarity_matrix = torch.matmul(audio_embeddings_norm, text_embeddings_norm.T) / self.temperature
        
        
        batch_size = audio_embeddings.shape[0]
        labels = torch.arange(batch_size, device=audio_embeddings.device)
        
        
        loss_a2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2a = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_a2t + loss_t2a) / 2
        
        return {"loss": loss, "logits": similarity_matrix}


@dataclass
class AudioTextDataCollator:
    tokenizer: AutoTokenizer
    feature_extractor: WhisperFeatureExtractor
    device: Optional[Union[str, torch.device]] = None
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_arrays = []
        texts = []
        
        for feature in features:
            # if not feature:
            #     continue
            # print(len(features), feature.keys(), feature["audio"].shape)
            audio_arrays.append(feature["audio"])
            texts.append(feature["text"].replace(" <PERIOD>", ".").replace(" <COMMA>", ", "))
            
        # Tokenize texts
        text_inputs = self.tokenizer(
            texts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        )

        audio_inputs = self.feature_extractor(
             raw_speech=audio_arrays, 
            return_tensors="pt", 
            sampling_rate=16000
        )
        # print("input size", audio_inputs["input_features"].shape, text_inputs["input_ids"].shape)
       
        return {
            "audio": audio_inputs,
            "text": text_inputs,
        }


def compute_metrics(eval_preds):
    logits, _ = eval_preds
    batch_size = logits.shape[0]
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    accuracy = np.mean([probs[i, i] for i in range(batch_size)])
    
    # Calculate recall@k metrics
    recalls = {}
    for k in [1, 5, 10]:
        if k <= batch_size:
            recall_k = 0
            for i in range(batch_size):
                sorted_indices = np.argsort(-logits[i])
                if i in sorted_indices[:k]:
                    recall_k += 1
            recalls[f"recall@{k}"] = recall_k / batch_size
    
    metrics = {
        "accuracy": accuracy,
        **recalls
    }
    return metrics

def group_batch(batch):
    return {k: [i["array"] if k == "audio" else i for i in v] for k, v in batch.items() if k in ["audio", "text"]}



def main():
    
    
    dataset = load_dataset(
            "/home/jeswanth/projects/datasets/gigaspeech", 
            "xl", 
            data_dir="/home/jeswanth/projects/datasets/gigaspeech", 
            cache_dir="/home/jeswanth/projects/datasets/gigaspeech", 
            trust_remote_code=True, 
            keep_in_memory=True, 
            streaming=True
    )
    dataset = dataset.map(group_batch, batched=True, batch_size=32)
    
    # train_ds = AudioDataset(dataset["train"], size=500)
    # eval_ds = AudioDataset(dataset["validation"], size=100)
    train_ds = dataset["train"].take(72000)
    eval_ds = dataset["validation"].take(1000)

    def gen_data():
        for data in train_ds:
            yield data

    def gen_val_data():
        for data in eval_ds:
            yield data
    train_ds = Dataset.from_generator(gen_data, cache_dir="/home/jeswanth/projects/datasets/gigaspeech", num_proc=12)
    eval_ds = Dataset.from_generator(gen_val_data, cache_dir="/home/jeswanth/projects/datasets/gigaspeech", num_proc=12)
    # dataloader = accelerator.prepare(train_ds)

    
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = AudioTextModel(embed_dim=768)
    
    
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        eval_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        learning_rate=2e-4,
        # max_steps=196,
        per_device_train_batch_size=120,
        per_device_eval_batch_size=120,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1,
        fp16=True,  
        gradient_accumulation_steps=1,
        warmup_steps=500,
        report_to="tensorboard",
        dataloader_num_workers=4,
    )
    
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
    data_collator = AudioTextDataCollator(tokenizer=tokenizer, feature_extractor=feature_extractor)
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    trainer.save_model(os.path.join(str(CHECKPOINT_DIR), "final_model"))
    
    final_model_dir = Path(os.path.join(str(CHECKPOINT_DIR), "final_encoders"))
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.audio_encoder.state_dict(), final_model_dir / "audio_encoder.pth")
    torch.save(model.text_encoder.state_dict(), final_model_dir / "text_encoder.pth")
    
    print("Training complete. Models saved.")


if __name__ == "__main__":
    main()