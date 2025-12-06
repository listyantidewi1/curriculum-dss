# ============================================================
# train_jobbert_research.py — The "Q1 Quality" Trainer
# ============================================================

import os
import numpy as np
import torch
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import f1_score, classification_report
from config import (
    JOBBERT_MODEL_NAME, 
    PROJECT_ROOT, 
    SKILL_LABEL2ID, KNOWLEDGE_LABEL2ID,
    SKILL_ID2LABEL, KNOWLEDGE_ID2LABEL
)
from multitask_model import ResearchJobBERT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = PROJECT_ROOT / "models" / "jobbert_research_v1"

# ============================================
# 1. Robust Data Loading & Weight Calculation
# ============================================

def get_class_weights(dataset, label2id, label_col):
    """
    Calculates Inverse Class Frequency weights.
    Formula: Total_Samples / (Num_Classes * Frequency_of_Class)
    """
    print(f"⚖️ Calculating weights for {label_col}...")
    
    # Flatten all tags in the dataset
    all_tags = []
    for row in dataset:
        all_tags.extend(row[label_col])
    
    counter = Counter(all_tags)
    total_count = len(all_tags)
    num_classes = len(label2id)
    
    weights = []
    # Iterate in order of IDs (0, 1, 2...)
    for label_name, label_id in sorted(label2id.items(), key=lambda x: x[1]):
        count = counter.get(label_name, 0)
        if count == 0:
            # Handle rare classes that might not appear in sample
            w = 1.0 
        else:
            w = total_count / (num_classes * count)
        
        # Dampen weights slightly to prevent explosion (e.g. max weight 20)
        # w = min(w, 20.0) 
        weights.append(w)
        print(f"   Class '{label_name}' (ID {label_id}): Count={count}, Weight={w:.4f}")
        
    return torch.tensor(weights).to(DEVICE)

def load_skillspan():
    print("📥 Loading SkillSpan...")
    ds = load_dataset("jjzha/skillspan")
    
    # Use standard split
    train_data = ds["train"]
    val_data = ds["validation"] if "validation" in ds else ds["dev"] if "dev" in ds else ds["train"].train_test_split(0.1)["test"]

    return train_data, val_data

# ============================================
# 2. Dataset Processing
# ============================================

class ResearchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        s_tags = item['tags_skill']
        k_tags = item['tags_knowledge']

        # Tokenize
        encoding = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True, 
            padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        word_ids = encoding.word_ids()

        # Align Labels
        labels_s = np.full(self.max_len, -100)
        labels_k = np.full(self.max_len, -100)
        
        prev_wid = None
        for i, wid in enumerate(word_ids):
            if wid is None: continue
            if wid != prev_wid and wid < len(s_tags):
                # Standardize Labels (Handle B-SKILL vs B)
                s_lbl = s_tags[wid].replace("-SKILL", "") if "SKILL" in s_tags[wid] else s_tags[wid]
                k_lbl = k_tags[wid].replace("-KNOWLEDGE", "") if "KNOWLEDGE" in k_tags[wid] else k_tags[wid]
                
                labels_s[i] = SKILL_LABEL2ID.get(s_lbl, 0) # Default to O if mismatch
                labels_k[i] = KNOWLEDGE_LABEL2ID.get(k_lbl, 0)
            prev_wid = wid

        batch = {k: v.squeeze(0) for k, v in encoding.items()}
        batch["labels_skill"] = torch.tensor(labels_s, dtype=torch.long)
        batch["labels_knowledge"] = torch.tensor(labels_k, dtype=torch.long)
        return batch

# ============================================
# 3. Custom Trainer
# ============================================

class CustomTrainer(Trainer):
    """Custom Trainer to pass the loss weights to the model."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # The model architecture itself handles the weighted loss calculation 
        # inside its forward() method, so we just call it standardly.
        outputs = model(**inputs)
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    predictions, labels = p
    preds_s_logits, preds_k_logits = predictions
    labels_s, labels_k = labels

    preds_s = np.argmax(preds_s_logits, axis=2)
    preds_k = np.argmax(preds_k_logits, axis=2)

    # Filter -100
    true_p_s = [[SKILL_ID2LABEL[p] for (p, l) in zip(pr, lb) if l != -100] for pr, lb in zip(preds_s, labels_s)]
    true_l_s = [[SKILL_ID2LABEL[l] for (p, l) in zip(pr, lb) if l != -100] for pr, lb in zip(preds_s, labels_s)]
    
    true_p_k = [[KNOWLEDGE_ID2LABEL[p] for (p, l) in zip(pr, lb) if l != -100] for pr, lb in zip(preds_k, labels_k)]
    true_l_k = [[KNOWLEDGE_ID2LABEL[l] for (p, l) in zip(pr, lb) if l != -100] for pr, lb in zip(preds_k, labels_k)]

    return {
        "skill_f1": f1_score(true_l_s, true_p_s),
        "know_f1": f1_score(true_l_k, true_p_k),
    }

# ============================================
# 4. Main Execution
# ============================================

def main():
    # A. Setup
    tokenizer = AutoTokenizer.from_pretrained(JOBBERT_MODEL_NAME)
    train_raw, val_raw = load_skillspan()
    
    # B. Calculate Weights
    s_weights = get_class_weights(train_raw, SKILL_LABEL2ID, "tags_skill")
    k_weights = get_class_weights(train_raw, KNOWLEDGE_LABEL2ID, "tags_knowledge")
    
    # C. Prepare Datasets
    train_ds = ResearchDataset(train_raw, tokenizer)
    val_ds = ResearchDataset(val_raw, tokenizer)
    
    # D. Init Model
    def model_init():
        model = ResearchJobBERT(JOBBERT_MODEL_NAME, len(SKILL_LABEL2ID), len(KNOWLEDGE_LABEL2ID))
        # INJECT WEIGHTS HERE
        model.set_class_weights(s_weights, k_weights)
        model.to(DEVICE)
        return model

    # E. Training Config
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5, # Optimal for BERT NER
        per_device_train_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="skill_f1", # Optimize for the HARD task
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available()
    )

    trainer = CustomTrainer(
        model_init=model_init,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🚀 Starting Research-Grade Training...")
    trainer.train()
    
    # F. Save Final
    print(f"💾 Saving to {OUTPUT_DIR}")
    # We must save the custom model carefully
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Also save the classification heads separately just in case
    torch.save({
        'classifier_skill_state': trainer.model.classifier_skill.state_dict(),
        'classifier_knowledge_state': trainer.model.classifier_knowledge.state_dict()
    }, os.path.join(OUTPUT_DIR, "classifier_states.pt"))
    print("✅ Done.")

if __name__ == "__main__":
    main()