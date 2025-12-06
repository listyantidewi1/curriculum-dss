import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from TorchCRF import CRF  # Ensure pytorch-crf is installed

class ResearchJobBERT(nn.Module):
    def __init__(self, model_name, skill_num_labels, knowledge_num_labels, dropout=0.1):
        super(ResearchJobBERT, self).__init__()
        
        # 1. Load Backbone
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        
        # 2. Emission Layers
        self.classifier_skill = nn.Linear(self.config.hidden_size, skill_num_labels)
        self.classifier_knowledge = nn.Linear(self.config.hidden_size, knowledge_num_labels)
        
        # 3. CRF Layers (These were missing!)
        self.crf_skill = CRF(skill_num_labels, batch_first=True)
        self.crf_knowledge = CRF(knowledge_num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels_skill=None, labels_knowledge=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits_skill = self.classifier_skill(sequence_output)
        logits_knowledge = self.classifier_knowledge(sequence_output)
        
        if labels_skill is not None and labels_knowledge is not None:
            # Training Mode: Return Loss
            mask = attention_mask.byte()
            loss_s = -self.crf_skill(logits_skill, labels_skill, mask=mask, reduction='mean')
            loss_k = -self.crf_knowledge(logits_knowledge, labels_knowledge, mask=mask, reduction='mean')
            return loss_s + loss_k
        else:
            # Inference Mode: Return Decoded Tags (List of Lists)
            mask = attention_mask.byte()
            tags_skill = self.crf_skill.decode(logits_skill, mask=mask)
            tags_knowledge = self.crf_knowledge.decode(logits_knowledge, mask=mask)
            
            # The pipeline expects dictionary output for logits.
            # Since CRF returns hard predictions (integers), we wrap them.
            return {'logits_skill': tags_skill, 'logits_knowledge': tags_knowledge}