# ============================================================
# pipeline_advanced_complete.py — Complete Advanced Pipeline
# ============================================================

import os
import re
import json
import time
import torch
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Any, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher
warnings.filterwarnings('ignore')


import sys
import io

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the openai library: pip install openai")

# Import your project modules
try:
    from multitask_model import ResearchJobBERT 
    from config import *
    # Ensure OUTPUT_DIR is defined
    if 'OUTPUT_DIR' not in locals() and 'OUTPUT_DIR' not in globals():
        OUTPUT_DIR = "output"
except ImportError:
    print("Warning: Could not import project modules. Some functionality may be limited.")
    # Create dummy constants for testing
    class DummyConfig:
        MULTITASK_MODEL_DIR = "models"
        JOBBERT_MODEL_NAME = "bert-base-uncased"
        OUTPUT_DIR = "output"
        SKILL_ID2LABEL = {}
        KNOWLEDGE_ID2LABEL = {}
        CURRICULUM_COMPONENTS = {}
    
    config = DummyConfig()
    MULTITASK_MODEL_DIR = config.MULTITASK_MODEL_DIR
    JOBBERT_MODEL_NAME = config.JOBBERT_MODEL_NAME
    OUTPUT_DIR = config.OUTPUT_DIR
    SKILL_ID2LABEL = config.SKILL_ID2LABEL
    KNOWLEDGE_ID2LABEL = config.KNOWLEDGE_ID2LABEL
    CURRICULUM_COMPONENTS = config.CURRICULUM_COMPONENTS

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_float(value):
    """Convert any value to float, handling PyTorch tensors."""
    if torch.is_tensor(value):
        # 如果是PyTorch张量
        return float(value.item())
    elif hasattr(value, '__iter__') and not isinstance(value, str):
        # 如果是可迭代对象（列表、元组等），但不是字符串
        try:
            return float(value[0]) if len(value) > 0 else 0.0
        except:
            return 0.0
    else:
        # 尝试直接转换为浮点数
        try:
            return float(value)
        except:
            return 0.0

def get_avg_confidence(skills):
    """Calculate average confidence from a list of skills."""
    if not skills:
        return 0.0
    return np.mean([ensure_float(s.confidence_score) for s in skills])

# ============================================================
# 1. ENUMS & DATA CLASSES
# ============================================================

class SkillType(Enum):
    HARD = "Hard"
    SOFT = "Soft"

class BloomLevel(Enum):
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"
    NA = "N/A"

class ConfidenceTier(Enum):
    VERY_HIGH = "Very High"  # 0.9-1.0
    HIGH = "High"            # 0.8-0.9
    MEDIUM_HIGH = "Medium High"  # 0.7-0.8
    MEDIUM = "Medium"        # 0.6-0.7
    MEDIUM_LOW = "Medium Low"    # 0.5-0.6
    LOW = "Low"              # 0.4-0.5
    VERY_LOW = "Very Low"    # <0.4

@dataclass
class SkillItem:
    """Structured representation of a skill."""
    text: str
    type: SkillType
    bloom: BloomLevel
    confidence_score: float  # Continuous 0.0-1.0
    confidence_tier: ConfidenceTier
    source: str  # "BERT", "GPT", "BERT+GPT", "Hybrid"
    semantic_density: float = 1.0  # How information-dense the skill is
    context_agreement: float = 1.0  # Agreement with context
    
    def to_dict(self):
        # Ensure all numeric values are Python floats before rounding
        confidence = ensure_float(self.confidence_score)
        density = ensure_float(self.semantic_density)
        agreement = ensure_float(self.context_agreement)
        
        return {
            "skill": self.text,
            "type": self.type.value,
            "bloom": self.bloom.value,
            "confidence_score": round(confidence, 3),
            "confidence_tier": self.confidence_tier.value,
            "source": self.source,
            "semantic_density": round(density, 2),
            "context_agreement": round(agreement, 2)
        }

@dataclass
class KnowledgeItem:
    """Structured representation of knowledge."""
    text: str
    confidence_score: float
    confidence_tier: ConfidenceTier
    source: str
    
    def to_dict(self):
        # Ensure confidence_score is a Python float, not a tensor
        confidence = ensure_float(self.confidence_score)
        return {
            "knowledge": self.text,
            "confidence_score": round(confidence, 3),
            "confidence_tier": self.confidence_tier.value,
            "source": self.source
        }

# ============================================================
# 2. ADVANCED CONFIGURATION
# ============================================================

class AdvancedPipelineConfig:
    """Advanced configuration with dynamic thresholds."""
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # GPT model
    GPT_MODEL = "deepseek/deepseek-chat"
    
    # OpenAI API (replace with your actual key)
    OPENAI_API_KEY = None
    OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    
    # Base Thresholds (will be adjusted dynamically)
    BASE_SIMILARITY_THRESHOLD = 0.55
    BASE_SKILL_MATCH_THRESHOLD = 0.75
    BASE_KNOWLEDGE_MATCH_THRESHOLD = 0.85
    
    # Confidence tier thresholds
    CONFIDENCE_THRESHOLDS = {
        ConfidenceTier.VERY_HIGH: 0.9,
        ConfidenceTier.HIGH: 0.8,
        ConfidenceTier.MEDIUM_HIGH: 0.7,
        ConfidenceTier.MEDIUM: 0.6,
        ConfidenceTier.MEDIUM_LOW: 0.5,
        ConfidenceTier.LOW: 0.4,
        ConfidenceTier.VERY_LOW: 0.0
    }
    
    # Dynamic adjustment factors
    DYNAMIC_FACTORS = {
        'skill_type': {
            'hard': {'similarity': 0.05, 'match': 0.05},  # Hard skills get stricter thresholds
            'soft': {'similarity': -0.05, 'match': -0.05}  # Soft skills get more lenient
        },
        'bloom_complexity': {
            'Remember': -0.03,
            'Understand': -0.02,
            'Apply': 0.0,
            'Analyze': 0.02,
            'Evaluate': 0.03,
            'Create': 0.05
        },
        'semantic_density': {
            'low': -0.02,     # Simple skills
            'medium': 0.0,    # Average
            'high': 0.03      # Complex skills
        }
    }
    
    # Coverage weights based on confidence tiers
    COVERAGE_WEIGHTS = {
        ConfidenceTier.VERY_HIGH: 1.0,
        ConfidenceTier.HIGH: 0.9,
        ConfidenceTier.MEDIUM_HIGH: 0.8,
        ConfidenceTier.MEDIUM: 0.7,
        ConfidenceTier.MEDIUM_LOW: 0.6,
        ConfidenceTier.LOW: 0.5,
        ConfidenceTier.VERY_LOW: 0.3
    }
    
    # Partial match handling
    PARTIAL_MATCH_CONFIG = {
        'substring_weight': 0.7,  # Weight for substring matches
        'levenshtein_threshold': 0.8,  # Similarity threshold for fuzzy matching
        'min_overlap_ratio': 0.6  # Minimum overlap for partial matches
    }
    
    # File paths
    INPUT_CSV = 'DATA\\preprocessing\\data_prepared\\jobs_sentences.csv'
    SAMPLE_SIZE = 20000
    
    # Output files
    OUTPUT_FILES = {
        'advanced_skills': "advanced_skills.csv",
        'advanced_knowledge': "advanced_knowledge.csv",
        'detailed_analysis': "detailed_analysis.json",
        'coverage_report': "coverage_report.csv",
        'summary': "pipeline_summary.txt"
    }

# ============================================================
# 3. MODEL MANAGER
# ============================================================

class ModelManager:
    """Manages loading and using all models."""
    
    def __init__(self):
        self.bert_tokenizer = None
        self.bert_model = None
        self.skill_label_map = None
        self.knowledge_label_map = None
        self.openai_client = None
        self.embedder = None
        
    def load_bert_model(self) -> None:
        """Load the JobBERT model and tokenizer."""
        logger.info(f"Loading Research JobBERT from {MULTITASK_MODEL_DIR}...")
        
        weights_path = os.path.join(MULTITASK_MODEL_DIR, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights: {weights_path}")
        
        # Load state dict to get label dimensions
        state_dict = torch.load(weights_path, map_location=DEVICE)
        skill_labels = state_dict['classifier_skill.weight'].shape[0]
        knowledge_labels = state_dict['classifier_knowledge.weight'].shape[0]
        
        # Initialize model
        model = ResearchJobBERT(JOBBERT_MODEL_NAME, skill_labels, knowledge_labels)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MULTITASK_MODEL_DIR)
        
        # Create label maps
        def create_label_map(n_labels):
            return {i: ('B' if i < n_labels-1 else 'O') for i in range(n_labels)}
        
        skill_map = SKILL_ID2LABEL if len(SKILL_ID2LABEL) == skill_labels else create_label_map(skill_labels)
        knowledge_map = KNOWLEDGE_ID2LABEL if len(KNOWLEDGE_ID2LABEL) == knowledge_labels else create_label_map(knowledge_labels)
        
        self.bert_tokenizer = tokenizer
        self.bert_model = model
        self.skill_label_map = skill_map
        self.knowledge_label_map = knowledge_map
        
        logger.info("[✓] JobBERT model loaded successfully")
    
    def load_openai_client(self) -> None:
        """Load OpenAI/OpenRouter client, reading key from file if needed."""
        base_url = AdvancedPipelineConfig.OPENAI_BASE_URL

        # 1) Try environment variable first (optional)
        api_key = os.getenv("OPENROUTER_API_KEY")

        # 2) If not set, read from api_keys/OpenRouter.txt
        if not api_key:
            key_path = os.path.join("api_keys", "OpenRouter.txt")  # works on Win & Linux
            try:
                with open(key_path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"⚠️ API key file not found at {key_path}.")
                self.openai_client = None
                return

        if not api_key:
            logger.warning("⚠️ OpenRouter API key is empty.")
            self.openai_client = None
            return

        # 3) Initialize client
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("[✓] OpenAI client loaded successfully")

    
    def load_embedder(self) -> None:
        """Load sentence transformer embedder."""
        logger.info(f"Loading embedding model: {AdvancedPipelineConfig.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(AdvancedPipelineConfig.EMBEDDING_MODEL)
        logger.info("[✓] Embedding model loaded successfully")
    
    def extract_with_bert(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract skills and knowledge using BERT."""
        if not self.bert_model:
            raise ValueError("BERT model not loaded")
        
        # Tokenize
        words = text.replace("/", " / ").split()
        inputs = self.bert_tokenizer(
            words, 
            is_split_into_words=True, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = self.bert_model(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
        
        # Decode predictions
        skill_preds = outputs['logits_skill'][0]
        knowledge_preds = outputs['logits_knowledge'][0]
        
        skills = self._decode_crf_predictions(
            skill_preds, inputs, words, self.skill_label_map
        )
        knowledge = self._decode_crf_predictions(
            knowledge_preds, inputs, words, self.knowledge_label_map
        )
        
        # Refine skills (split compounds)
        refined_skills = self._refine_bert_skills(skills)
        
        return refined_skills, knowledge
    
    def _refine_bert_skills(self, raw_skills: List[Dict]) -> List[Dict]:
        """Split compound JobBERT skills into atomic skills."""
        refined = []
        for item in raw_skills:
            text = item['text']
            conf = item['confidence']
            parts = re.split(r',|;|\s+and\s+|\s+&\s+', text)
            for p in parts:
                p = p.strip()
                if len(p) > 2:
                    refined.append({'text': p, 'confidence': conf})
        return refined
    
    def _decode_crf_predictions(self, predictions, inputs, words, label_map):
        """Decode CRF predictions to extract entities."""
        extracted = []
        current_entity = []
        current_confidences = []
        
        for i, word_id in enumerate(inputs.word_ids()):
            if word_id is None:
                continue
            if i > 0 and inputs.word_ids()[i] == inputs.word_ids()[i-1]:
                continue
            if i >= len(predictions):
                break
            
            label = label_map.get(predictions[i], 'O')
            confidence = float(0.95)  # Placeholder for CRF
            
            if 'B' in label:
                if current_entity:
                    extracted.append({
                        "text": " ".join(current_entity),
                        "confidence": float(np.mean(current_confidences)) 
                    })
                current_entity = [words[word_id]]
                current_confidences = [confidence]
            
            elif 'I' in label and current_entity:
                current_entity.append(words[word_id])
                current_confidences.append(confidence)
            
            else:
                if current_entity:
                    extracted.append({
                        "text": " ".join(current_entity),
                        "confidence": float(np.mean(current_confidences))
                    })
                current_entity = []
                current_confidences = []
        
        if current_entity:
            extracted.append({
                "text": " ".join(current_entity),
                "confidence": np.mean(current_confidences)
            })
        
        return extracted

# ============================================================
# 4. ADVANCED TAXONOMY MANAGER
# ============================================================

class AdvancedTaxonomyManager:
    """Enhanced taxonomy manager with dynamic adjustments."""
    
    # Comprehensive taxonomies
    BLOOMS_TAXONOMY = {
        'Remember': [
            'identify', 'define', 'know', 'label', 'list', 'match', 'name', 'recall',
            'recognize', 'reproduce', 'state', 'memorize', 'quote', 'repeat',
            'retrieve', 'duplicate', 'acquire', 'recite', 'enumerate', 'record',
            'find', 'locate', 'cite', 'recollect', 'bookmark', 'search', 'google',
            'access', 'browse', 'capture', 'note', 'mark', 'highlight', 'point out'
        ],
        'Understand': [
            'describe', 'comprehend', 'explain', 'summarize', 'summarise',
            'interpret', 'paraphrase', 'classify', 'discuss', 'express',
            'indicate', 'report', 'restate', 'translate', 'exemplify',
            'infer', 'clarify', 'extend', 'generalize', 'map',
            'articulate', 'elaborate', 'characterize', 'detail', 'outline',
            'review', 'present', 'illustrate', 'rephrase', 'give examples',
            'summarize key points', 'describe in own words', 'relate ideas',
            'simplify', 'comment', 'annotate', 'associate', 'categorize',
            'tag', 'convert', 'decode', 'visualize'
        ],
        'Apply': [
            'apply', 'use', 'execute', 'implement', 'solve', 'demonstrate',
            'operate', 'calculate', 'complete', 'show', 'employ', 'practice',
            'utilize', 'run', 'administer', 'carry out', 'make use of',
            'select method', 'test implementation', 'use in a new situation',
            'manage', 'operate', 'adapt', 'change', 'experiment', 'explore',
            'perform', 'simulate', 'compute', 'discover', 'manipulate',
            'modify', 'prepare', 'relate', 'deploy', 'upload', 'share',
            'edit', 'install', 'configure', 'backup', 'code', 'compile',
            'connect', 'decrypt', 'document', 'encrypt', 'iterate',
            'provision', 'recover', 'restore', 'script', 'troubleshoot',
            'maintain', 'execute script', 'schedule', 'sketch',
            'organize steps', 'implement procedure', 'follow instructions'
        ],
        'Analyze': [
            'analyze', 'compare', 'contrast', 'deconstruct', 'attribute',
            'differentiate', 'discriminate', 'distinguish', 'separate',
            'organize', 'structure', 'outline', 'break down', 'examine',
            'inspect', 'investigate', 'question', 'test', 'diagram',
            'dissect', 'detect', 'survey', 'trace', 'correlate',
            'interpret data', 'discover patterns', 'identify relationships',
            'identify causes', 'diagnose', 'root cause', 'extrapolate',
            'infer', 'identify', 'debug', 'monitor', 'audit', 'benchmark',
            'profile', 'decompile', 'disassemble', 'reverse-engineer',
            'log', 'query', 'analyze logs', 'analyze performance',
            'ensure', 'maintain'
        ],
        'Evaluate': [
            'evaluate', 'assess', 'judge', 'critique', 'criticize', 'appraise',
            'argue', 'defend', 'justify', 'conclude', 'decide', 'persuade',
            'prioritize', 'rank', 'rate', 'recommend', 'choose', 'support',
            'validate', 'verify', 'value', 'weigh', 'consider', 'debate',
            'check', 'measure', 'score', 'grade', 'estimate impact',
            'determine effectiveness', 'review', 'approve', 'reject',
            'review code', 'comment on code', 'inspect', 'predict',
            'test', 'optimize', 'tune', 'balance', 'benchmark alternatives',
            'compare alternatives', 'analyze trade-offs', 'risk assess',
            'sanity check', 'playtest'
        ],
        'Create': [
            'create', 'design', 'develop', 'construct', 'produce', 'plan',
            'invent', 'formulate', 'compose', 'generate', 'devise', 'hypothesize',
            'originate', 'assemble', 'compile', 'integrate', 'rearrange',
            'reconstruct', 'reorganize', 'revise', 'rewrite', 'synthesize',
            'write', 'brainstorm', 'ideate', 'conceive', 'build', 'program',
            'architect', 'engineer', 'prototype', 'model', 'orchestrate',
            'containerize', 'virtualize', 'automate', 'initialize',
            'instantiate', 'launch', 'animate', 'blog', 'storyboard',
            'draft', 'author', 'publish', 'refactor', 'redesign',
            'reengineer', 'extend functionality', 'innovate', 'improve',
            'enhance', 'customize', 'configure solution', 'design pattern',
            'develop strategy', 'propose approach', 'test concept',
            'theorize', 'solutioning'
        ]
    }
    
    # Enhanced keyword sets with weights
    STRICT_SOFT_KEYWORDS = {
        "communication": 1.0, "team": 0.9, "interpersonal": 1.0, "empathy": 1.0,
        "negotiation": 0.9, "presentation": 0.8, "verbal": 0.7, "written": 0.7,
        "work ethic": 1.0, "self-motivated": 0.9, "detail oriented": 0.8,
        "passionate": 0.7, "enthusiastic": 0.7, "proactive": 0.8, "flexible": 0.8,
        "integrity": 1.0, "conflict": 0.9, "listen": 0.8, "relationship": 0.9,
        "coordinate": 0.7, "coordination": 0.7, "schedule": 0.6, "scheduling": 0.6,
        "track": 0.5, "tracking": 0.5, "follow-up": 0.7, "liaise": 0.8,
        "liaison": 0.8, "travel": 0.5, "handover": 0.6, "meeting": 0.6,
        "administrative": 0.7, "organize": 0.6, "organization": 0.6
    }
    
    # Ambiguous keywords with context hints
    AMBIGUOUS_KEYWORDS = {
        "manage": {'hard_hints': ['database', 'system', 'server', 'code', 'infrastructure'],
                   'soft_hints': ['team', 'people', 'stakeholders', 'relationships']},
        "management": {'hard_hints': ['project', 'data', 'system', 'database'],
                       'soft_hints': ['people', 'team', 'change', 'conflict']},
        "oversee": {'hard_hints': ['process', 'system', 'implementation'],
                    'soft_hints': ['team', 'staff', 'department']},
        "plan": {'hard_hints': ['project', 'implementation', 'architecture'],
                 'soft_hints': ['strategy', 'career', 'development']}
    }
    
    @classmethod
    def get_confidence_tier(cls, score: float) -> ConfidenceTier:
        """Get confidence tier from continuous score."""
        # Ensure score is a Python float
        score = ensure_float(score)
        
        for tier, threshold in AdvancedPipelineConfig.CONFIDENCE_THRESHOLDS.items():
            if score >= threshold:
                return tier
        return ConfidenceTier.VERY_LOW
    
    @classmethod
    def calculate_semantic_density(cls, text: str) -> float:
        """Calculate how information-dense a skill description is."""
        words = text.lower().split()
        
        # Simple metrics
        word_count = len(words)
        if word_count == 0:
            return 0.5
        
        unique_words = len(set(words))
        avg_word_length = np.mean([len(w) for w in words])
        
        # Technical indicator words
        technical_indicators = {'api', 'database', 'algorithm', 'framework', 
                               'protocol', 'interface', 'architecture', 'deployment'}
        technical_count = sum(1 for w in words if w in technical_indicators)
        
        # Calculate density score (0-1)
        uniqueness = unique_words / word_count
        technical_ratio = technical_count / word_count
        complexity = min(avg_word_length / 10, 1.0)  # Normalize
        
        density = (uniqueness * 0.4 + technical_ratio * 0.4 + complexity * 0.2)
        return min(density * 1.5, 1.0)  # Scale up slightly
    
    @classmethod
    def get_dynamic_threshold(cls, skill_type: SkillType, bloom_level: BloomLevel, 
                             semantic_density: float) -> Dict[str, float]:
        """Calculate dynamic thresholds based on skill characteristics."""
        
        # Get base adjustments
        type_factor = AdvancedPipelineConfig.DYNAMIC_FACTORS['skill_type'][
            'hard' if skill_type == SkillType.HARD else 'soft'
        ]
        bloom_factor = AdvancedPipelineConfig.DYNAMIC_FACTORS['bloom_complexity'].get(
            bloom_level.value, 0.0
        )
        
        # Determine density category
        if semantic_density < 0.4:
            density_cat = 'low'
        elif semantic_density < 0.7:
            density_cat = 'medium'
        else:
            density_cat = 'high'
        
        density_factor = AdvancedPipelineConfig.DYNAMIC_FACTORS['semantic_density'][density_cat]
        
        # Calculate adjusted thresholds
        similarity_threshold = (
            AdvancedPipelineConfig.BASE_SIMILARITY_THRESHOLD + 
            type_factor['similarity'] + 
            bloom_factor + 
            density_factor
        )
        
        skill_match_threshold = (
            AdvancedPipelineConfig.BASE_SKILL_MATCH_THRESHOLD + 
            type_factor['match'] + 
            bloom_factor * 0.5 + 
            density_factor * 0.5
        )
        
        return {
            'similarity': max(0.3, min(0.9, similarity_threshold)),
            'skill_match': max(0.5, min(0.95, skill_match_threshold)),
            'knowledge_match': AdvancedPipelineConfig.BASE_KNOWLEDGE_MATCH_THRESHOLD
        }
    
    @classmethod
    def classify_skill_with_context(cls, text: str, gpt_type: Optional[str] = None, 
                                   context: List[str] = None) -> Tuple[SkillType, float]:
        """
        Advanced skill classification with context awareness.
        Returns (skill_type, confidence_score)
        """
        text_lower = text.lower()
        context_lower = [c.lower() for c in context] if context else []
        
        # 1. Check for strict soft keywords with confidence
        soft_confidence = 0.0
        for keyword, weight in cls.STRICT_SOFT_KEYWORDS.items():
            if keyword in text_lower:
                soft_confidence = max(soft_confidence, weight)
        
        if soft_confidence > 0.7:
            return SkillType.SOFT, soft_confidence
        
        # 2. Check ambiguous keywords with context
        for keyword, hints in cls.AMBIGUOUS_KEYWORDS.items():
            if keyword in text_lower:
                # Check context for hints
                hard_context_hits = sum(1 for hint in hints['hard_hints'] 
                                      if any(hint in ctx for ctx in context_lower))
                soft_context_hits = sum(1 for hint in hints['soft_hints'] 
                                      if any(hint in ctx for ctx in context_lower))
                
                if hard_context_hits > soft_context_hits:
                    return SkillType.HARD, 0.7
                elif soft_context_hits > hard_context_hits:
                    return SkillType.SOFT, 0.7
        
        # 3. Use GPT classification if available with context adjustment
        if gpt_type:
            gpt_confidence = 0.8  # Base confidence for GPT
            # Adjust based on context consistency
            context_words = ' '.join(context_lower)
            if any(word in context_words for word in ['technical', 'technology', 'software', 'code']):
                if gpt_type == 'Hard':
                    gpt_confidence += 0.1
                else:
                    gpt_confidence -= 0.1
            
            return SkillType.HARD if gpt_type == 'Hard' else SkillType.SOFT, gpt_confidence
        
        # 4. Default to Hard with moderate confidence
        return SkillType.HARD, 0.6
    
    @classmethod
    def get_bloom_with_confidence(cls, text: str, skill_type: SkillType) -> Tuple[BloomLevel, float]:
        """Get Bloom level with confidence score."""
        if skill_type == SkillType.SOFT:
            return BloomLevel.NA, 1.0
        
        text_lower = text.lower()
        matches = []
        
        for bloom_level, keywords in cls.BLOOMS_TAXONOMY.items():
            for keyword in keywords:
                # Check for exact word match (higher confidence)
                if f' {keyword} ' in f' {text_lower} ':
                    matches.append((bloom_level, 0.9))
                # Check for substring match (lower confidence)
                elif keyword in text_lower:
                    matches.append((bloom_level, 0.6))
        
        if matches:
            # Sort by Bloom level (higher is better) and confidence
            bloom_order = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
            matches.sort(key=lambda x: (bloom_order.index(x[0]), x[1]), reverse=True)
            bloom_str, confidence = matches[0]
            return BloomLevel(bloom_str), confidence
        
        # Default to Apply with moderate confidence
        return BloomLevel.APPLY, 0.5
    
    @classmethod
    def calculate_partial_match_confidence(cls, text1: str, text2: str, 
                                         match_type: str = 'substring') -> float:
        """Calculate confidence for partial matches."""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        if match_type == 'substring':
            # Check if one is substring of the other
            if text1_lower in text2_lower or text2_lower in text1_lower:
                overlap = min(len(text1_lower), len(text2_lower)) / max(len(text1_lower), len(text2_lower), 1)
                base_confidence = AdvancedPipelineConfig.PARTIAL_MATCH_CONFIG['substring_weight']
                return base_confidence * overlap
        
        elif match_type == 'fuzzy':
            # Simple Levenshtein-like similarity
            similarity = SequenceMatcher(None, text1_lower, text2_lower).ratio()
            if similarity > AdvancedPipelineConfig.PARTIAL_MATCH_CONFIG['levenshtein_threshold']:
                return similarity * 0.8
        
        elif match_type == 'word_overlap':
            # Word overlap ratio
            words1 = set(text1_lower.split())
            words2 = set(text2_lower.split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            if overlap > AdvancedPipelineConfig.PARTIAL_MATCH_CONFIG['min_overlap_ratio']:
                return overlap * 0.7
        
        return 0.0

# ============================================================
# 5. GPT EXTRACTOR
# ============================================================

class GPTExtractor:
    """Handles GPT-based extraction."""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def extract_skills_and_knowledge(self, text: str, bert_knowledge: List[Dict]) -> Dict:
        """Extract skills and knowledge using GPT."""
        if not self.client:
            return {"skills": [], "knowledge": []}
        
        # Prepare knowledge context
        knowledge_context = ", ".join([k['text'] for k in bert_knowledge])
        
        system_prompt = """You are an expert in job market analysis, skill extraction, and educational taxonomy mapping.

        Your task is to extract:
        1. **Skills** → actions, behaviors, abilities, competencies.
        2. **Knowledge** → tools, technologies, concepts, frameworks, domains.

        Follow ALL rules strictly:

        ========================
        ### GENERAL RULES
        ========================
        1. Output **ONLY** a valid JSON object with keys: `"skills"` and `"knowledge"`.
        2. If the input text is **NOT in English**, ONLY extract knowledge

        ========================
        ### EXTRACTION RULES
        ========================
        3. **Skills extraction**:
        - Must be action-oriented (verbs + context).
        - Structure:  
            { "skill": "verb + object/context",  
            "type": "Hard" | "Soft",  
            "bloom": "Remember" | "Apply" | "Understand" | "Analyze" | "Evaluate" | "Create" | "N/A" }

        4. **Hard vs Soft skills**:
            - HARD SKILL:
                * technical, measurable, domain-specific
                * requires tools, technologies, frameworks, or specialized knowledge
                * examples: "analyze data", "manage CI/CD pipeline", "write C++", "use Python", "operate CNC machine", "implement APIs", "design database schema"
                * must receive an appropriate Bloom level
            - SOFT SKILL:
                * ANY non-technical skill involving cognitive, interpersonal, behavioral,
                    personal management, communication, or strategic thinking abilities
                * does NOT require a specific technical tool or domain expertise
                * examples: "communicate ideas", "collaborate with team", "solve problems",
                    "manage time", "adapt to change", "think critically", "lead a team"
                * bloom = "N/A"

        5. **Knowledge extraction**:
        - Knowledge is **NOT a skill**.
        - Knowledge is NOT educational degree (degree, bachelor, master, PhD, doctoral, diploma).
        - Knowledge items MUST be nouns representing tools, technologies, platforms, or theoretical concepts.
        - Output format: list of strings. Example:  
            ["Python", "Docker", "Agile methodology"]

        6. **Bloom Taxonomy Mapping**:
        - Only HARD skills get Bloom levels.
        - Must be inferable from the verb/action:  
            • Apply → implement, build, use  
            • Understand → explain, describe, interpret 
            - Remember → identify, list, recall 
            • Analyze → troubleshoot, break down, examine  
            • Evaluate → assess, validate, prioritize  
            • Create → design, develop, architect

        ========================
        ### CONTEXT INTEGRATION
        ========================
        7. Use "Context (Tools detected by JobBERT)" ONLY as a hint to infer skills or knowledge.  
        Do NOT fabricate new skills, knowledges, or tools not present in job text or context.

        ========================
        ### OUTPUT VALIDATION
        ========================
        8. No duplicates.  
        9. No hallucinations.  
        10. No commentary, only JSON.
        """
        
        user_prompt = f"""
            Job Description: "{text}"
            Context (Tools detected by BERT): [{knowledge_context}]

            Return JSON: {{
                "skills": [...],
                "knowledge": [...]
            }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=AdvancedPipelineConfig.GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            return {
                "skills": data.get("skills", []),
                "knowledge": data.get("knowledge", [])
            }
            
        except Exception as e:
            logger.error(f"GPT extraction error: {e}")
            return {"skills": [], "knowledge": []}

# ============================================================
# 6. CONTEXT-AWARE EXTRACTOR
# ============================================================

class ContextAwareExtractor:
    """Extraction with context awareness and model agreement."""
    
    def __init__(self, model_manager, gpt_extractor):
        self.model_manager = model_manager
        self.gpt_extractor = gpt_extractor
        self.embedder = model_manager.embedder
        
    def extract_with_context_awareness(self, text: str) -> Dict[str, List[SkillItem]]:
        """Extract skills with context awareness and model agreement analysis."""
        
        # Extract knowledge with BERT first
        bert_skills_raw, bert_knowledge = self.model_manager.extract_with_bert(text)
        
        # Extract from GPT
        gpt_output = self.gpt_extractor.extract_skills_and_knowledge(text, bert_knowledge)
        
        # Convert to structured format
        bert_skills = self._process_bert_skills(bert_skills_raw, text)
        gpt_skills = self._process_gpt_skills(gpt_output.get('skills', []), text)
        
        # Analyze model agreement
        agreement_scores = self._analyze_model_agreement(bert_skills, gpt_skills)
        
        # Adjust confidences based on agreement
        bert_skills = self._adjust_confidence_with_agreement(bert_skills, agreement_scores, 'bert')
        gpt_skills = self._adjust_confidence_with_agreement(gpt_skills, agreement_scores, 'gpt')
        
        # Check Bloom consistency
        bert_skills = self._check_bloom_consistency(bert_skills, text)
        gpt_skills = self._check_bloom_consistency(gpt_skills, text)
        
        return {
            'bert': bert_skills,
            'gpt': gpt_skills,
            'bert_knowledge': bert_knowledge,
            'gpt_knowledge': gpt_output.get('knowledge', []),
            'agreement_scores': agreement_scores
        }
    
    def _process_bert_skills(self, raw_skills: List[Dict], context: str) -> List[SkillItem]:
        """Process BERT skills with advanced features."""
        processed = []
        
        for skill in raw_skills:
            text = skill['text']
            raw_confidence = skill['confidence']
            
            # Calculate semantic density
            semantic_density = AdvancedTaxonomyManager.calculate_semantic_density(text)
            
            # Classify skill with context
            skill_type, type_confidence = AdvancedTaxonomyManager.classify_skill_with_context(
                text, context=[context]
            )
            
            # Get Bloom level with confidence
            bloom_level, bloom_confidence = AdvancedTaxonomyManager.get_bloom_with_confidence(
                text, skill_type
            )
            
            # Calculate overall confidence
            base_confidence = raw_confidence * 0.7 + type_confidence * 0.3
            adjusted_confidence = base_confidence * (0.5 + bloom_confidence * 0.5)
            final_confidence = adjusted_confidence * (0.8 + semantic_density * 0.2)
            
            skill_item = SkillItem(
                text=text,
                type=skill_type,
                bloom=bloom_level,
                confidence_score=final_confidence,
                confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(final_confidence),
                source="BERT",
                semantic_density=semantic_density,
                context_agreement=1.0  # Will be adjusted later
            )
            processed.append(skill_item)
        
        return processed
    
    def _process_gpt_skills(self, gpt_skills: List[Dict], context: str) -> List[SkillItem]:
        """Process GPT skills with advanced features."""
        processed = []
        
        for skill in gpt_skills:
            text = skill.get('skill', '').strip()
            if not text:
                continue
            
            gpt_type = skill.get('type', 'Hard')
            gpt_bloom = skill.get('bloom', 'Apply')
            
            # Calculate semantic density
            semantic_density = AdvancedTaxonomyManager.calculate_semantic_density(text)
            
            # Classify skill with context (use GPT's classification as hint)
            skill_type, type_confidence = AdvancedTaxonomyManager.classify_skill_with_context(
                text, gpt_type, context=[context]
            )
            
            # Convert GPT bloom to enum
            try:
                bloom_level = BloomLevel(gpt_bloom)
                bloom_confidence = 0.8  # Base confidence for GPT's bloom assignment
            except:
                bloom_level, bloom_confidence = AdvancedTaxonomyManager.get_bloom_with_confidence(
                    text, skill_type
                )
            
            # Calculate overall confidence
            base_confidence = 0.8  # Base for GPT
            adjusted_confidence = base_confidence * (0.6 + type_confidence * 0.4)
            final_confidence = adjusted_confidence * (0.7 + bloom_confidence * 0.3)
            final_confidence *= (0.8 + semantic_density * 0.2)
            
            skill_item = SkillItem(
                text=text,
                type=skill_type,
                bloom=bloom_level,
                confidence_score=final_confidence,
                confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(final_confidence),
                source="GPT",
                semantic_density=semantic_density,
                context_agreement=1.0  # Will be adjusted later
            )
            processed.append(skill_item)
        
        return processed
    
    def _analyze_model_agreement(self, bert_skills: List[SkillItem], 
                                gpt_skills: List[SkillItem]) -> Dict[str, float]:
        """Analyze agreement between BERT and GPT models."""
        if not bert_skills or not gpt_skills:
            return {'overall': 0.0, 'matches': [], 'bert_count': 0, 'gpt_count': 0, 'match_count': 0}
        
        bert_texts = [s.text.lower() for s in bert_skills]
        gpt_texts = [s.text.lower() for s in gpt_skills]
        
        # Calculate embeddings
        bert_embeddings = self.embedder.encode(bert_texts, convert_to_tensor=True)
        gpt_embeddings = self.embedder.encode(gpt_texts, convert_to_tensor=True)
        
        # Calculate similarity matrix
        similarity_matrix = util.cos_sim(bert_embeddings, gpt_embeddings)
        
        # Find matches
        matches = []
        for i, bert_skill in enumerate(bert_skills):
            for j, gpt_skill in enumerate(gpt_skills):
                similarity = similarity_matrix[i][j].item()
                
                # Check for various match types
                if similarity > 0.85:
                    matches.append({
                        'bert_idx': i,
                        'gpt_idx': j,
                        'similarity': similarity,
                        'type': 'exact'
                    })
                else:
                    # Check for partial matches
                    partial_confidence = AdvancedTaxonomyManager.calculate_partial_match_confidence(
                        bert_skill.text, gpt_skill.text, 'substring'
                    )
                    if partial_confidence > 0.5:
                        matches.append({
                            'bert_idx': i,
                            'gpt_idx': j,
                            'similarity': partial_confidence,
                            'type': 'partial'
                        })
        
        # Calculate agreement scores
        if matches:
            avg_similarity = np.mean([m['similarity'] for m in matches])
            coverage = len(matches) / max(len(bert_skills), len(gpt_skills))
            overall_agreement = avg_similarity * coverage
        else:
            overall_agreement = 0.0
        
        return {
            'overall': overall_agreement,
            'matches': matches,
            'bert_count': len(bert_skills),
            'gpt_count': len(gpt_skills),
            'match_count': len(matches)
        }
    
    def _adjust_confidence_with_agreement(self, skills: List[SkillItem], 
                                        agreement_scores: Dict, 
                                        model: str) -> List[SkillItem]:
        """Adjust confidence scores based on model agreement."""
        adjusted = []
        
        for skill in skills:
            # Base adjustment from overall agreement
            agreement_factor = 1.0 + (agreement_scores['overall'] * 0.2 - 0.1)
            
            # Additional adjustment if this skill has a match
            matched = False
            for match in agreement_scores.get('matches', []):
                idx = match['bert_idx'] if model == 'bert' else match['gpt_idx']
                if skills.index(skill) == idx:
                    matched = True
                    # Higher confidence for matched skills
                    match_factor = 1.0 + (match['similarity'] * 0.3)
                    agreement_factor *= match_factor
                    break
            
            # Apply adjustment
            new_confidence = min(1.0, ensure_float(skill.confidence_score) * agreement_factor)
            
            adjusted_skill = SkillItem(
                text=skill.text,
                type=skill.type,
                bloom=skill.bloom,
                confidence_score=new_confidence,
                confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(new_confidence),
                source=skill.source,
                semantic_density=skill.semantic_density,
                context_agreement=agreement_factor
            )
            adjusted.append(adjusted_skill)
        
        return adjusted
    
    def _check_bloom_consistency(self, skills: List[SkillItem], context: str) -> List[SkillItem]:
        """Check if Bloom levels are consistent with skill context."""
        adjusted = []
        
        for skill in skills:
            if skill.bloom == BloomLevel.NA:
                adjusted.append(skill)
                continue
            
            # Check if bloom level makes sense for the skill text
            expected_bloom, confidence = AdvancedTaxonomyManager.get_bloom_with_confidence(
                skill.text, skill.type
            )
            
            if expected_bloom == skill.bloom:
                # Consistent - boost confidence
                new_confidence = min(1.0, ensure_float(skill.confidence_score) * (1.0 + confidence * 0.1))
            else:
                # Inconsistent - reduce confidence
                bloom_order = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
                expected_idx = bloom_order.index(expected_bloom.value) if expected_bloom.value in bloom_order else 2
                actual_idx = bloom_order.index(skill.bloom.value) if skill.bloom.value in bloom_order else 2
                bloom_distance = abs(expected_idx - actual_idx)
                penalty = 1.0 - (bloom_distance * 0.1)
                new_confidence = ensure_float(skill.confidence_score) * penalty
            
            adjusted_skill = SkillItem(
                text=skill.text,
                type=skill.type,
                bloom=skill.bloom,  # Keep original for now
                confidence_score=new_confidence,
                confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(new_confidence),
                source=skill.source,
                semantic_density=skill.semantic_density,
                context_agreement=skill.context_agreement
            )
            adjusted.append(adjusted_skill)
        
        return adjusted

# ============================================================
# 7. ADVANCED FUSION ENGINE
# ============================================================

class AdvancedFusionEngine:
    """Advanced fusion with continuous confidence and partial matching."""
    
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
    
    def fuse_skills_advanced(self, bert_skills: List[SkillItem], 
                           gpt_skills: List[SkillItem]) -> List[SkillItem]:
        """
        Advanced fusion with continuous confidence scores and partial matching.
        """
        # Start with all GPT skills
        fused = gpt_skills.copy()
        
        # Track which BERT skills have been matched
        bert_matched = [False] * len(bert_skills)
        
        if bert_skills and gpt_skills:
            # Calculate all pairwise similarities
            bert_texts = [s.text for s in bert_skills]
            gpt_texts = [s.text for s in gpt_skills]
            
            bert_embeddings = self.embedder.encode(bert_texts, convert_to_tensor=True)
            gpt_embeddings = self.embedder.encode(gpt_texts, convert_to_tensor=True)
            
            similarity_matrix = util.cos_sim(bert_embeddings, gpt_embeddings)
            
            # Process each BERT skill
            for i, bert_skill in enumerate(bert_skills):
                best_match_idx = -1
                best_match_score = 0.0
                best_match_type = None
                
                # Check for exact matches
                for j, gpt_skill in enumerate(gpt_skills):
                    similarity = similarity_matrix[i][j].item()
                    
                    # Get dynamic threshold for this skill
                    thresholds = AdvancedTaxonomyManager.get_dynamic_threshold(
                        bert_skill.type, bert_skill.bloom, bert_skill.semantic_density
                    )
                    
                    if similarity > thresholds['skill_match']:
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_idx = j
                            best_match_type = 'exact'
                
                # Check for partial matches if no exact match found
                if best_match_idx == -1:
                    for j, gpt_skill in enumerate(gpt_skills):
                        # Check substring match
                        if bert_skill.text.lower() in gpt_skill.text.lower() or \
                           gpt_skill.text.lower() in bert_skill.text.lower():
                            
                            overlap = min(len(bert_skill.text), len(gpt_skill.text)) / \
                                     max(len(bert_skill.text), len(gpt_skill.text), 1)
                            
                            if overlap > 0.6:  # Significant overlap
                                partial_score = overlap * 0.8
                                if partial_score > best_match_score:
                                    best_match_score = partial_score
                                    best_match_idx = j
                                    best_match_type = 'substring'
                
                # Handle the match
                if best_match_idx != -1:
                    bert_matched[i] = True
                    gpt_skill = gpt_skills[best_match_idx]
                    
                    # Fuse the matched skills
                    fused_skill = self._fuse_matched_skills(
                        bert_skill, gpt_skill, best_match_score, best_match_type
                    )
                    fused[best_match_idx] = fused_skill
        
        # Add unmatched BERT skills
        for i, matched in enumerate(bert_matched):
            if not matched:
                bert_skill = bert_skills[i]
                # Adjust confidence for standalone BERT skill
                adjusted_confidence = ensure_float(bert_skill.confidence_score) * 0.8  # Penalty for no GPT match
                
                fused_skill = SkillItem(
                    text=bert_skill.text,
                    type=bert_skill.type,
                    bloom=bert_skill.bloom,
                    confidence_score=adjusted_confidence,
                    confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(adjusted_confidence),
                    source="BERT (standalone)",
                    semantic_density=bert_skill.semantic_density,
                    context_agreement=bert_skill.context_agreement * 0.9
                )
                fused.append(fused_skill)
        
        return fused
    
    def _fuse_matched_skills(self, bert_skill: SkillItem, gpt_skill: SkillItem,
                           match_score: float, match_type: str) -> SkillItem:
        """Fuse two matched skills into one."""
        
        # Decide which text to use (prefer GPT for clarity, BERT for completeness)
        if match_type == 'exact' or match_score > 0.9:
            # High confidence match - use GPT text (usually better formatted)
            fused_text = gpt_skill.text
        else:
            # Partial match - use the longer/more complete text
            fused_text = bert_skill.text if len(bert_skill.text) > len(gpt_skill.text) else gpt_skill.text
        
        # Decide skill type (prioritize strict classification)
        if bert_skill.type == SkillType.SOFT or gpt_skill.type == SkillType.SOFT:
            fused_type = SkillType.SOFT
        else:
            # If both are hard, check confidence
            if ensure_float(bert_skill.confidence_score) > ensure_float(gpt_skill.confidence_score):
                fused_type = bert_skill.type
            else:
                fused_type = gpt_skill.type
        
        # Decide Bloom level
        if fused_type == SkillType.SOFT:
            fused_bloom = BloomLevel.NA
        else:
            # For hard skills, prefer the higher Bloom level with confidence adjustment
            bloom_order = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
            bert_bloom_value = bloom_order.index(bert_skill.bloom.value) if bert_skill.bloom.value in bloom_order else -1
            gpt_bloom_value = bloom_order.index(gpt_skill.bloom.value) if gpt_skill.bloom.value in bloom_order else -1
            
            if bert_bloom_value > gpt_bloom_value:
                fused_bloom = bert_skill.bloom
                bloom_confidence = ensure_float(bert_skill.confidence_score) * 0.7
            else:
                fused_bloom = gpt_skill.bloom
                bloom_confidence = ensure_float(gpt_skill.confidence_score) * 0.7
        
        # Calculate fused confidence
        base_confidence = (ensure_float(bert_skill.confidence_score) + ensure_float(gpt_skill.confidence_score)) / 2
        match_bonus = match_score * 0.2
        type_agreement = 1.0 if bert_skill.type == gpt_skill.type else 0.8
        
        fused_confidence = min(1.0, base_confidence * (1.0 + match_bonus) * type_agreement)
        
        # Calculate semantic density (average)
        fused_density = (ensure_float(bert_skill.semantic_density) + ensure_float(gpt_skill.semantic_density)) / 2
        
        # Context agreement (weighted by confidence)
        fused_context_agreement = (
            ensure_float(bert_skill.context_agreement) * ensure_float(bert_skill.confidence_score) +
            ensure_float(gpt_skill.context_agreement) * ensure_float(gpt_skill.confidence_score)
        ) / (ensure_float(bert_skill.confidence_score) + ensure_float(gpt_skill.confidence_score))
        
        return SkillItem(
            text=fused_text,
            type=fused_type,
            bloom=fused_bloom,
            confidence_score=fused_confidence,
            confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(fused_confidence),
            source="BERT+GPT",
            semantic_density=fused_density,
            context_agreement=fused_context_agreement
        )
    
    def fuse_knowledge_advanced(self, bert_knowledge: List[Dict], 
                              gpt_knowledge: List[str]) -> List[KnowledgeItem]:
        """Fuse knowledge items from BERT and GPT."""
        fused_knowledge = []
        
        # Convert BERT knowledge to KnowledgeItems
        bert_items = []
        for item in bert_knowledge:
            confidence = item.get('confidence', 0.5)
            bert_items.append(KnowledgeItem(
                text=item['text'],
                confidence_score=confidence * 0.8,  # Slightly penalize BERT knowledge
                confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(confidence * 0.8),
                source="BERT"
            ))
        
        # Convert GPT knowledge to KnowledgeItems
        gpt_items = []
        for item in gpt_knowledge:
            gpt_items.append(KnowledgeItem(
                text=item,
                confidence_score=0.75,  # Base confidence for GPT
                confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(0.75),
                source="GPT"
            ))
        
        # Start with GPT knowledge
        fused = gpt_items.copy()
        bert_matched = [False] * len(bert_items)
        
        if bert_items and gpt_items:
            # Calculate similarities
            bert_texts = [k.text for k in bert_items]
            gpt_texts = [k.text for k in gpt_items]
            
            bert_embeddings = self.embedder.encode(bert_texts, convert_to_tensor=True)
            gpt_embeddings = self.embedder.encode(gpt_texts, convert_to_tensor=True)
            
            similarity_matrix = util.cos_sim(bert_embeddings, gpt_embeddings)
            
            # Match BERT to GPT
            for i, bert_item in enumerate(bert_items):
                best_match_idx = -1
                best_match_score = 0.0
                
                for j, gpt_item in enumerate(gpt_items):
                    similarity = similarity_matrix[i][j].item()
                    if similarity > AdvancedPipelineConfig.BASE_KNOWLEDGE_MATCH_THRESHOLD:
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_idx = j
                
                if best_match_idx != -1:
                    bert_matched[i] = True
                    gpt_item = gpt_items[best_match_idx]
                    
                    # Fuse matched knowledge
                    fused_confidence = (ensure_float(bert_item.confidence_score) + ensure_float(gpt_item.confidence_score)) / 2
                    fused_confidence *= (1.0 + best_match_score * 0.1)  # Bonus for match
                    
                    fused_knowledge_item = KnowledgeItem(
                        text=gpt_item.text,  # Use GPT text (usually cleaner)
                        confidence_score=fused_confidence,
                        confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(fused_confidence),
                        source="BERT+GPT"
                    )
                    fused[best_match_idx] = fused_knowledge_item
        
        # Add unmatched BERT knowledge
        for i, matched in enumerate(bert_matched):
            if not matched:
                bert_item = bert_items[i]
                # Lower confidence for standalone BERT knowledge
                adjusted_confidence = ensure_float(bert_item.confidence_score) * 0.7
                fused.append(KnowledgeItem(
                    text=bert_item.text,
                    confidence_score=adjusted_confidence,
                    confidence_tier=AdvancedTaxonomyManager.get_confidence_tier(adjusted_confidence),
                    source="BERT (standalone)"
                ))
        
        return fused

# ============================================================
# 8. ADVANCED COVERAGE CALCULATOR
# ============================================================

class AdvancedCoverageCalculator:
    """Advanced coverage calculation with continuous confidence."""
    
    def __init__(self, embedder: SentenceTransformer, curriculum_tensors: Dict):
        self.embedder = embedder
        self.curriculum_tensors = curriculum_tensors
    
    def calculate_advanced_coverage(self, skills: List[SkillItem], 
                                  knowledge: List[KnowledgeItem]) -> Dict:
        """
        Calculate comprehensive coverage metrics with advanced features.
        """
        # Separate hard and soft skills
        hard_skills = [s for s in skills if s.type == SkillType.HARD]
        soft_skills = [s for s in skills if s.type == SkillType.SOFT]
        
        # Calculate coverage for each component
        skill_coverage = self._calculate_component_coverage(hard_skills, 'skill')
        knowledge_coverage = self._calculate_component_coverage(knowledge, 'knowledge')
        
        # Combine coverage
        combined_coverage = {}
        all_components = set(self.curriculum_tensors.keys())
        
        for component in all_components:
            skill_weight = skill_coverage.get(component, 0.0)
            knowledge_weight = knowledge_coverage.get(component, 0.0)
            combined_coverage[component] = max(skill_weight, knowledge_weight)
        
        # Calculate metrics
        total_components = len(all_components)
        covered_components = len([c for c in combined_coverage.values() if c > 0])
        
        # Weighted coverage scores
        skill_coverage_score = sum(skill_coverage.values())
        knowledge_coverage_score = sum(knowledge_coverage.values())
        total_coverage_score = sum(combined_coverage.values())
        
        # Normalize
        skill_coverage_pct = skill_coverage_score / total_components if total_components > 0 else 0
        knowledge_coverage_pct = knowledge_coverage_score / total_components if total_components > 0 else 0
        total_coverage_pct = total_coverage_score / total_components if total_components > 0 else 0
        
        # Find missing components
        missing_components = [
            comp for comp in all_components 
            if combined_coverage.get(comp, 0.0) == 0
        ]
        
        # Quality metrics
        avg_skill_confidence = np.mean([ensure_float(s.confidence_score) for s in skills]) if skills else 0.0
        avg_knowledge_confidence = np.mean([ensure_float(k.confidence_score) for k in knowledge]) if knowledge else 0.0
        
        return {
            'skill_coverage': skill_coverage,
            'knowledge_coverage': knowledge_coverage,
            'combined_coverage': combined_coverage,
            'metrics': {
                'total_components': total_components,
                'covered_components': covered_components,
                'coverage_percentage': total_coverage_pct,
                'skill_coverage_pct': skill_coverage_pct,
                'knowledge_coverage_pct': knowledge_coverage_pct,
                'avg_skill_confidence': avg_skill_confidence,
                'avg_knowledge_confidence': avg_knowledge_confidence,
                'hard_skill_count': len(hard_skills),
                'soft_skill_count': len(soft_skills),
                'knowledge_count': len(knowledge)
            },
            'missing_components': missing_components,
            'coverage_gaps': self._identify_coverage_gaps(combined_coverage)
        }
    
    def _calculate_component_coverage(self, items: List, item_type: str) -> Dict[str, float]:
        """Calculate coverage for specific items."""
        if not items:
            return {}
        
        # Extract texts and confidence scores
        if item_type == 'skill':
            texts = [item.text for item in items]
            confidences = [ensure_float(item.confidence_score) for item in items]
            # Get weights from confidence tiers
            weights = [AdvancedPipelineConfig.COVERAGE_WEIGHTS[item.confidence_tier] 
                      for item in items]
        else:  # knowledge
            texts = [item.text for item in items]
            confidences = [ensure_float(item.confidence_score) for item in items]
            weights = [AdvancedPipelineConfig.COVERAGE_WEIGHTS[item.confidence_tier] 
                      for item in items]
        
        # Calculate embeddings
        embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        
        coverage = {}
        
        for component_id, component_embeddings in self.curriculum_tensors.items():
            # Calculate similarities
            similarities = util.cos_sim(embeddings, component_embeddings).max(dim=1)[0]
            
            # Find best match for each item
            best_matches = []
            for idx, similarity in enumerate(similarities):
                similarity_float = similarity.item() if torch.is_tensor(similarity) else float(similarity)
                if similarity_float > AdvancedPipelineConfig.BASE_SIMILARITY_THRESHOLD:
                    # Calculate adjusted weight
                    base_weight = weights[idx]
                    confidence_factor = confidences[idx]
                    similarity_factor = similarity_float
                    
                    # Dynamic adjustment based on item characteristics
                    if item_type == 'skill':
                        skill = items[idx]
                        thresholds = AdvancedTaxonomyManager.get_dynamic_threshold(
                            skill.type, skill.bloom, skill.semantic_density
                        )
                        threshold_factor = similarity_float / thresholds['similarity'] if thresholds['similarity'] > 0 else 1.0
                    else:
                        threshold_factor = 1.0
                    
                    adjusted_weight = base_weight * confidence_factor * similarity_factor * threshold_factor
                    best_matches.append(adjusted_weight)
            
            if best_matches:
                coverage[component_id] = max(best_matches)
        
        return coverage
    
    def _identify_coverage_gaps(self, coverage: Dict[str, float]) -> List[Dict]:
        """Identify specific coverage gaps with severity levels."""
        gaps = []
        
        for component_id, weight in coverage.items():
            if weight < 0.3:  # Very poor coverage
                severity = 'high'
            elif weight < 0.5:  # Poor coverage
                severity = 'medium'
            elif weight < 0.7:  # Moderate coverage
                severity = 'low'
            else:
                continue  # Good coverage
            
            gaps.append({
                'component': component_id,
                'coverage_weight': round(weight, 3),
                'severity': severity,
                'suggestion': self._get_coverage_suggestion(component_id, weight)
            })
        
        return gaps
    
    def _get_coverage_suggestion(self, component_id: str, weight: float) -> str:
        """Get suggestion for improving coverage."""
        suggestions = [
            "Consider adding more specific skills in this area",
            "Include more technical knowledge items",
            "Add both theoretical and practical aspects",
            "Cover multiple Bloom taxonomy levels",
            "Include both hard and soft skills"
        ]
        
        # Simple heuristic based on weight
        if weight < 0.3:
            return suggestions[0]  # Most critical
        elif weight < 0.5:
            return suggestions[1]
        else:
            return suggestions[2]

# ============================================================
# 9. DATA MANAGER
# ============================================================

class AdvancedDataManager:
    """Handles data I/O for advanced pipeline."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define file paths
        self.files = {
            'skills': os.path.join(output_dir, AdvancedPipelineConfig.OUTPUT_FILES['advanced_skills']),
            'knowledge': os.path.join(output_dir, AdvancedPipelineConfig.OUTPUT_FILES['advanced_knowledge']),
            'coverage': os.path.join(output_dir, AdvancedPipelineConfig.OUTPUT_FILES['coverage_report']),
            'summary': os.path.join(output_dir, AdvancedPipelineConfig.OUTPUT_FILES['summary']),
            'comprehensive': os.path.join(output_dir, "comprehensive_analysis.csv"),
            'comparison': os.path.join(output_dir, "model_comparison.csv")
        }
        
        # Initialize files
        self._initialize_files()
    
    def _initialize_files(self):
        """Create empty files with headers."""
        # Skills CSV
        skills_columns = ['job_id', 'date_posted', 'skill', 'type', 'bloom',
                        'confidence_score', 'confidence_tier', 'source',
                        'semantic_density', 'context_agreement']
        
        pd.DataFrame(columns=skills_columns).to_csv(self.files['skills'], index=False)
        
        # Knowledge CSV
        knowledge_columns = ['job_id', 'date_posted', 'knowledge', 'confidence_score', 'confidence_tier', 'source']
        pd.DataFrame(columns=knowledge_columns).to_csv(self.files['knowledge'], index=False)
        
        # Coverage CSV
        coverage_columns = ['job_id', 'date_posted', 'total_components', 'covered_components', 'coverage_percentage',
                           'skill_coverage_pct', 'knowledge_coverage_pct', 'avg_skill_confidence',
                           'avg_knowledge_confidence', 'missing_components_count']
        
        pd.DataFrame(columns=coverage_columns).to_csv(self.files['coverage'], index=False)
        
        # Comprehensive Analysis CSV
        comprehensive_columns = [
            'job_id',
            'raw_text',
            # JobBERT outputs
            'jobbert_skills_raw', 'jobbert_skills_count', 'jobbert_knowledge_raw', 'jobbert_knowledge_count',
            # GPT outputs  
            'gpt_skills_raw', 'gpt_skills_count', 'gpt_knowledge_raw', 'gpt_knowledge_count',
            # Hybrid outputs
            'final_skills', 'final_skills_count', 'final_knowledge', 'final_knowledge_count',
            # Taxonomy and classification
            'jobbert_hard_skills', 'jobbert_soft_skills', 'jobbert_bloom_distribution',
            'gpt_hard_skills', 'gpt_soft_skills', 'gpt_bloom_distribution',
            'final_hard_skills', 'final_soft_skills', 'final_bloom_distribution',
            # Confidence scores
            'jobbert_avg_confidence', 'gpt_avg_confidence', 'final_avg_confidence',
            'jobbert_confidence_distribution', 'gpt_confidence_distribution', 'final_confidence_distribution',
            # Coverage metrics
            'coverage_total_components', 'covered_components', 'coverage_percentage',
            'skill_coverage_pct', 'knowledge_coverage_pct', 'missing_components_count',
            # Model agreement
            'model_agreement_score', 'exact_matches', 'partial_matches', 'no_matches',
            # Fusion statistics
            'bert_only_skills', 'gpt_only_skills', 'fused_skills',
            'bert_only_knowledge', 'gpt_only_knowledge', 'fused_knowledge',
            # Quality metrics
            'avg_semantic_density', 'avg_context_agreement', 'bloom_consistency_score',
            # Extraction time (if available)
            'extraction_time_seconds'
        ]
        pd.DataFrame(columns=comprehensive_columns).to_csv(self.files['comprehensive'], index=False)
        
        # Model Comparison CSV
        comparison_columns = [
            'job_id', 'model', 'skill_count', 'hard_skill_count', 'soft_skill_count',
            'knowledge_count', 'avg_confidence', 'bloom_remember', 'bloom_understand', 
            'bloom_apply', 'bloom_analyze', 'bloom_evaluate', 'bloom_create', 'bloom_na',
            'coverage_pct', 'confidence_vh', 'confidence_h', 'confidence_mh', 'confidence_m',
            'confidence_ml', 'confidence_l', 'confidence_vl'
        ]
        pd.DataFrame(columns=comparison_columns).to_csv(self.files['comparison'], index=False)
    
    # def load_job_data(self, sample_size: int = None) -> List[str]:
    #     """Load job descriptions from CSV."""
    #     if sample_size is None:
    #         sample_size = AdvancedPipelineConfig.SAMPLE_SIZE
        
    #     logger.info(f"Loading data from {AdvancedPipelineConfig.INPUT_CSV}")
        
    #     try:
    #         df = pd.read_csv(AdvancedPipelineConfig.INPUT_CSV)
            
    #         if len(df) > sample_size:
    #             df_sample = df.sample(n=sample_size, random_state=42)
    #         else:
    #             df_sample = df
            
    #         # Fixed: Changed .ast(str) to .astype(str)
    #         jobs = df_sample['sentence'].astype(str).tolist()
    #         logger.info(f"Loaded {len(jobs)} job descriptions")
    #         return jobs
            
    #     except Exception as e:
    #         logger.error(f"Failed to load CSV: {e}")
    #         # Return some sample data for testing
    #         return [
    #             "Develop and maintain Python applications using Django framework",
    #             "Manage team of developers and coordinate with stakeholders",
    #             "Analyze data using SQL and create reports in Tableau",
    #             "Communicate effectively with cross-functional teams"
    #         ]
    
    def load_job_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load job descriptions + metadata from CSV."""
        if sample_size is None:
            sample_size = AdvancedPipelineConfig.SAMPLE_SIZE

        logger.info(f"Loading data from {AdvancedPipelineConfig.INPUT_CSV}")

        try:
            df = pd.read_csv(AdvancedPipelineConfig.INPUT_CSV)

            # Expect columns: job_id, sentence, date_posted
            required = {'job_id', 'sentence_text'}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"INPUT_CSV must contain columns {missing}")

            if len(df) > sample_size:
                df_sample = df.sample(n=sample_size, random_state=None)
            else:
                df_sample = df

            logger.info(f"Loaded {len(df_sample)} job sentences")
            # Return the whole sampled frame so we keep id + date
            return df_sample

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise


    
    def save_results(self, job_id: int, results: Dict, extraction_time: float = None) -> None:
        date_posted = results.get('date_posted')

        """Save results for a job."""
        try:
            # Save skills
            if results.get('skills'):
                skills_df = pd.DataFrame([s.to_dict() for s in results['skills']])
                skills_df.insert(0, 'job_id', job_id)
                skills_df.insert(1, 'date_posted', date_posted)
                skills_df.to_csv(self.files['skills'], mode='a', header=False, index=False)

            
            # Save knowledge
            if results.get('knowledge'):
                knowledge_df = pd.DataFrame([k.to_dict() for k in results['knowledge']])
                knowledge_df.insert(0, 'job_id', job_id)
                knowledge_df.insert(1, 'date_posted', date_posted)
                knowledge_df.to_csv(self.files['knowledge'], mode='a', header=False, index=False)

            
            # Save coverage metrics
            coverage_metrics = results.get('coverage_analysis', {}).get('metrics', {})
            if coverage_metrics:
                coverage_row = {
                    'job_id': job_id,
                    'date_posted': date_posted,
                    'total_components': coverage_metrics.get('total_components', 0),
                    'covered_components': coverage_metrics.get('covered_components', 0),
                    'coverage_percentage': coverage_metrics.get('coverage_percentage', 0),
                    'skill_coverage_pct': coverage_metrics.get('skill_coverage_pct', 0),
                    'knowledge_coverage_pct': coverage_metrics.get('knowledge_coverage_pct', 0),
                    'avg_skill_confidence': coverage_metrics.get('avg_skill_confidence', 0),
                    'avg_knowledge_confidence': coverage_metrics.get('avg_knowledge_confidence', 0),
                    'missing_components_count': len(results.get('coverage_analysis', {}).get('missing_components', []))
                }
                pd.DataFrame([coverage_row]).to_csv(self.files['coverage'], mode='a', header=False, index=False)
            
            # Save comprehensive analysis
            self._save_comprehensive_analysis(job_id, results, extraction_time)
            
            # Save model comparison
            self._save_model_comparison(job_id, results)
            
            # Save detailed analysis as JSON
            analysis_file = os.path.join(self.output_dir, f'job_{job_id}_analysis.json')
            with open(analysis_file, 'w') as f:
                json.dump({
                    'job_id': job_id,
                    'text': results.get('text', ''),
                    'skills': [s.to_dict() for s in results.get('skills', [])],
                    'knowledge': [k.to_dict() for k in results.get('knowledge', [])],
                    'coverage_analysis': results.get('coverage_analysis', {}),
                    'extraction_metrics': results.get('extraction_metrics', {})
                }, f, indent=2, default=str)
            
            logger.debug(f"Saved results for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error saving results for job {job_id}: {e}")
    
    def _save_comprehensive_analysis(self, job_id: int, results: Dict, extraction_time: float = None):
        """Save comprehensive analysis for a job."""
        try:
            # Extract data from results
            raw_text = results.get('text', '')
            extraction_results = results.get('extraction_results', {})
            bert_skills = extraction_results.get('bert', [])
            gpt_skills = extraction_results.get('gpt', [])
            bert_knowledge = extraction_results.get('bert_knowledge', [])
            gpt_knowledge = extraction_results.get('gpt_knowledge', [])
            final_skills = results.get('skills', [])
            final_knowledge = results.get('knowledge', [])
            
            # Helper functions
            def count_skills_by_type(skills, skill_type):
                return len([s for s in skills if s.type.value == skill_type])
            
            def get_bloom_distribution(skills):
                distribution = {
                    'Remember': 0, 'Understand': 0, 'Apply': 0, 
                    'Analyze': 0, 'Evaluate': 0, 'Create': 0, 'N/A': 0
                }
                for skill in skills:
                    bloom = skill.bloom.value
                    distribution[bloom] = distribution.get(bloom, 0) + 1
                return json.dumps(distribution)
            
            def get_confidence_distribution(skills):
                distribution = {
                    'Very High': 0, 'High': 0, 'Medium High': 0,
                    'Medium': 0, 'Medium Low': 0, 'Low': 0, 'Very Low': 0
                }
                for skill in skills:
                    tier = skill.confidence_tier.value
                    distribution[tier] = distribution.get(tier, 0) + 1
                return json.dumps(distribution)
            
            # Calculate statistics
            bert_hard_count = count_skills_by_type(bert_skills, 'Hard')
            bert_soft_count = count_skills_by_type(bert_skills, 'Soft')
            gpt_hard_count = count_skills_by_type(gpt_skills, 'Hard')
            gpt_soft_count = count_skills_by_type(gpt_skills, 'Soft')
            final_hard_count = count_skills_by_type(final_skills, 'Hard')
            final_soft_count = count_skills_by_type(final_skills, 'Soft')
            
            # Model agreement analysis
            agreement_scores = extraction_results.get('agreement_scores', {})
            exact_matches = len([m for m in agreement_scores.get('matches', []) if m.get('type') == 'exact'])
            partial_matches = len([m for m in agreement_scores.get('matches', []) if m.get('type') == 'partial'])
            no_matches = len(bert_skills) + len(gpt_skills) - (exact_matches + partial_matches)
            
            # Fusion statistics
            bert_only = len([s for s in final_skills if 'BERT (standalone)' in s.source])
            gpt_only = len([s for s in final_skills if s.source == 'GPT'])
            fused = len([s for s in final_skills if 'BERT+GPT' in s.source])
            
            bert_knowledge_only = len([k for k in final_knowledge if 'BERT (standalone)' in k.source])
            gpt_knowledge_only = len([k for k in final_knowledge if k.source == 'GPT'])
            fused_knowledge = len([k for k in final_knowledge if 'BERT+GPT' in k.source])
            
            # Quality metrics
            avg_semantic_density = np.mean([ensure_float(s.semantic_density) for s in final_skills]) if final_skills else 0.0
            avg_context_agreement = np.mean([ensure_float(s.context_agreement) for s in final_skills]) if final_skills else 0.0
            
            # Bloom consistency (check if Bloom levels make sense)
            bloom_consistency_score = 0.0
            if final_skills:
                consistent_count = 0
                for skill in final_skills:
                    if skill.type.value == 'Soft' and skill.bloom.value == 'N/A':
                        consistent_count += 1
                    elif skill.type.value == 'Hard' and skill.bloom.value != 'N/A':
                        consistent_count += 1
                bloom_consistency_score = consistent_count / len(final_skills)
            
            # Coverage data
            coverage_analysis = results.get('coverage_analysis', {})
            coverage_metrics = coverage_analysis.get('metrics', {})
            
            comprehensive_row = {
                'job_id': job_id,
                'raw_text': raw_text[:500],  # Truncate for CSV readability
                
                # JobBERT outputs
                'jobbert_skills_raw': json.dumps([s.text for s in bert_skills]),
                'jobbert_skills_count': len(bert_skills),
                'jobbert_knowledge_raw': json.dumps([k['text'] for k in bert_knowledge] if isinstance(bert_knowledge, list) else []),
                'jobbert_knowledge_count': len(bert_knowledge),
                
                # GPT outputs
                'gpt_skills_raw': json.dumps([s.text for s in gpt_skills]),
                'gpt_skills_count': len(gpt_skills),
                'gpt_knowledge_raw': json.dumps(gpt_knowledge),
                'gpt_knowledge_count': len(gpt_knowledge),
                
                # Hybrid outputs
                'final_skills': json.dumps([s.text for s in final_skills]),
                'final_skills_count': len(final_skills),
                'final_knowledge': json.dumps([k.text for k in final_knowledge]),
                'final_knowledge_count': len(final_knowledge),
                
                # Taxonomy and classification
                'jobbert_hard_skills': bert_hard_count,
                'jobbert_soft_skills': bert_soft_count,
                'jobbert_bloom_distribution': get_bloom_distribution(bert_skills),
                
                'gpt_hard_skills': gpt_hard_count,
                'gpt_soft_skills': gpt_soft_count,
                'gpt_bloom_distribution': get_bloom_distribution(gpt_skills),
                
                'final_hard_skills': final_hard_count,
                'final_soft_skills': final_soft_count,
                'final_bloom_distribution': get_bloom_distribution(final_skills),
                
                # Confidence scores
                'jobbert_avg_confidence': get_avg_confidence(bert_skills),
                'gpt_avg_confidence': get_avg_confidence(gpt_skills),
                'final_avg_confidence': get_avg_confidence(final_skills),
                
                'jobbert_confidence_distribution': get_confidence_distribution(bert_skills),
                'gpt_confidence_distribution': get_confidence_distribution(gpt_skills),
                'final_confidence_distribution': get_confidence_distribution(final_skills),
                
                # Coverage metrics
                'coverage_total_components': coverage_metrics.get('total_components', 0),
                'covered_components': coverage_metrics.get('covered_components', 0),
                'coverage_percentage': coverage_metrics.get('coverage_percentage', 0),
                'skill_coverage_pct': coverage_metrics.get('skill_coverage_pct', 0),
                'knowledge_coverage_pct': coverage_metrics.get('knowledge_coverage_pct', 0),
                'missing_components_count': len(coverage_analysis.get('missing_components', [])),
                
                # Model agreement
                'model_agreement_score': agreement_scores.get('overall', 0.0),
                'exact_matches': exact_matches,
                'partial_matches': partial_matches,
                'no_matches': no_matches,
                
                # Fusion statistics
                'bert_only_skills': bert_only,
                'gpt_only_skills': gpt_only,
                'fused_skills': fused,
                
                'bert_only_knowledge': bert_knowledge_only,
                'gpt_only_knowledge': gpt_knowledge_only,
                'fused_knowledge': fused_knowledge,
                
                # Quality metrics
                'avg_semantic_density': avg_semantic_density,
                'avg_context_agreement': avg_context_agreement,
                'bloom_consistency_score': bloom_consistency_score,
                
                # Extraction time
                'extraction_time_seconds': extraction_time or 0.0
            }
            
            pd.DataFrame([comprehensive_row]).to_csv(self.files['comprehensive'], mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"Error saving comprehensive analysis for job {job_id}: {e}")
    
    def _save_model_comparison(self, job_id: int, results: Dict):
        """Save model comparison data."""
        try:
            extraction_results = results.get('extraction_results', {})
            bert_skills = extraction_results.get('bert', [])
            gpt_skills = extraction_results.get('gpt', [])
            final_skills = results.get('skills', [])
            
            # Helper function to count Bloom levels
            def count_bloom_level(skills, bloom_level):
                return len([s for s in skills if s.bloom.value == bloom_level])
            
            # Helper function to count confidence tiers
            def count_confidence_tier(skills, tier):
                return len([s for s in skills if s.confidence_tier.value == tier])
            
            # Coverage data
            coverage_analysis = results.get('coverage_analysis', {})
            coverage_metrics = coverage_analysis.get('metrics', {})
            
            # Create comparison rows for each model
            models = [
                ('JobBERT', bert_skills),
                ('GPT', gpt_skills),
                ('Hybrid', final_skills)
            ]
            
            for model_name, skills in models:
                if not skills:
                    continue
                
                hard_count = len([s for s in skills if s.type.value == 'Hard'])
                soft_count = len([s for s in skills if s.type.value == 'Soft'])
                avg_conf = get_avg_confidence(skills) if skills else 0.0
                
                # For knowledge, we need to handle differently
                knowledge_count = 0
                if model_name == 'JobBERT':
                    knowledge_count = len(extraction_results.get('bert_knowledge', []))
                elif model_name == 'GPT':
                    knowledge_count = len(extraction_results.get('gpt_knowledge', []))
                elif model_name == 'Hybrid':
                    knowledge_count = len(results.get('knowledge', []))
                
                comparison_row = {
                    'job_id': job_id,
                    'model': model_name,
                    'skill_count': len(skills),
                    'hard_skill_count': hard_count,
                    'soft_skill_count': soft_count,
                    'knowledge_count': knowledge_count,
                    'avg_confidence': avg_conf,
                    
                    # Bloom distribution
                    'bloom_remember': count_bloom_level(skills, 'Remember'),
                    'bloom_understand': count_bloom_level(skills, 'Understand'),
                    'bloom_apply': count_bloom_level(skills, 'Apply'),
                    'bloom_analyze': count_bloom_level(skills, 'Analyze'),
                    'bloom_evaluate': count_bloom_level(skills, 'Evaluate'),
                    'bloom_create': count_bloom_level(skills, 'Create'),
                    'bloom_na': count_bloom_level(skills, 'N/A'),
                    
                    # Coverage (only for hybrid)
                    'coverage_pct': coverage_metrics.get('coverage_percentage', 0) if model_name == 'Hybrid' else 0,
                    
                    # Confidence distribution
                    'confidence_vh': count_confidence_tier(skills, 'Very High'),
                    'confidence_h': count_confidence_tier(skills, 'High'),
                    'confidence_mh': count_confidence_tier(skills, 'Medium High'),
                    'confidence_m': count_confidence_tier(skills, 'Medium'),
                    'confidence_ml': count_confidence_tier(skills, 'Medium Low'),
                    'confidence_l': count_confidence_tier(skills, 'Low'),
                    'confidence_vl': count_confidence_tier(skills, 'Very Low')
                }
                
                pd.DataFrame([comparison_row]).to_csv(self.files['comparison'], mode='a', header=False, index=False)
                
        except Exception as e:
            logger.error(f"Error saving model comparison for job {job_id}: {e}")
    
    def save_summary(self, pipeline_stats: Dict) -> None:
        """Save pipeline summary."""
        try:
            with open(self.files['summary'], 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("ADVANCED SKILL EXTRACTION PIPELINE - SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("Pipeline Statistics:\n")
                f.write(f"- Total jobs processed: {pipeline_stats.get('total_jobs', 0)}\n")
                f.write(f"- Average skills per job: {pipeline_stats.get('avg_skills_per_job', 0):.2f}\n")
                f.write(f"- Average knowledge per job: {pipeline_stats.get('avg_knowledge_per_job', 0):.2f}\n")
                f.write(f"- Average coverage: {pipeline_stats.get('avg_coverage', 0):.2%}\n")
                f.write(f"- Average model agreement: {pipeline_stats.get('avg_agreement', 0):.2%}\n\n")
                
                f.write("Advanced Features Used:\n")
                f.write("1. [✓] Dynamic Thresholds - Context-aware similarity matching\n")
                f.write("2. [✓] Continuous Confidence - 7-tier confidence system\n")
                f.write("3. [✓] Context Awareness - Model agreement & Bloom consistency\n")
                f.write("4. [✓] Partial Match Handling - Substring & fuzzy matching\n")
                f.write("5. [✓] Better Coverage - More BERT skills with adjusted confidence\n\n")
                
                f.write("Output Files:\n")
                for file_type, file_path in self.files.items():
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        f.write(f"- {file_type}: {file_path} ({file_size:.1f} KB)\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Comprehensive Analysis Files:\n")
                f.write("1. comprehensive_analysis.csv - Complete extraction data for analysis\n")
                f.write("2. model_comparison.csv - Side-by-side model comparison\n")
                f.write("3. advanced_skills.csv - Final hybrid skills with taxonomy\n")
                f.write("4. advanced_knowledge.csv - Final hybrid knowledge items\n")
                f.write("5. coverage_report.csv - Curriculum coverage metrics\n")
                f.write("6. job_*_analysis.json - Detailed JSON for each job\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Pipeline completed successfully!\n")
                f.write("=" * 60 + "\n")
            
            logger.info(f"Summary saved to {self.files['summary']}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")

# ============================================================
# 10. MAIN ADVANCED PIPELINE
# ============================================================

class AdvancedSkillExtractionPipeline:
    """Complete advanced pipeline orchestrator."""
    
    def __init__(self):
        self.model_manager = None
        self.gpt_extractor = None
        self.context_extractor = None
        self.fusion_engine = None
        self.coverage_calculator = None
        self.data_manager = None
        self.curriculum_tensors = None
    
    def initialize(self, output_dir: str = None) -> None:
        """Initialize all advanced components."""
        logger.info("🚀 Initializing Advanced Skill Extraction Pipeline...")
        
        # Use provided output_dir or default from config
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        try:
            # 1. Initialize Model Manager
            self.model_manager = ModelManager()
            self.model_manager.load_bert_model()
            self.model_manager.load_openai_client()
            self.model_manager.load_embedder()
            
            # 2. Initialize GPT Extractor
            self.gpt_extractor = GPTExtractor(self.model_manager.openai_client)
            
            # 3. Initialize Context-Aware Extractor
            self.context_extractor = ContextAwareExtractor(
                self.model_manager, 
                self.gpt_extractor
            )
            
            # 4. Initialize Fusion Engine
            self.fusion_engine = AdvancedFusionEngine(self.model_manager.embedder)
            
            # 5. Load curriculum data
            self._load_curriculum_tensors()
            
            # 6. Initialize Coverage Calculator
            self.coverage_calculator = AdvancedCoverageCalculator(
                self.model_manager.embedder,
                self.curriculum_tensors
            )
            
            # 7. Initialize Data Manager
            self.data_manager = AdvancedDataManager(output_dir)
            
            logger.info("[✓] Advanced pipeline initialized successfully")
            logger.info("\n" + "="*60)
            logger.info("Advanced Features Loaded:")
            logger.info("1. Dynamic Thresholds - Adjust based on skill characteristics")
            logger.info("2. Continuous Confidence - 7-tier confidence system")
            logger.info("3. Context Awareness - Model agreement, Bloom consistency")
            logger.info("4. Partial Match Handling - Substring and fuzzy matching")
            logger.info("5. Better Coverage - More BERT skills with adjusted confidence")
            logger.info("6. Comprehensive Analysis - Complete CSV for visualization")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _load_curriculum_tensors(self):
        """Load and prepare curriculum data."""
        if not CURRICULUM_COMPONENTS:
            logger.warning("No curriculum components found. Using empty curriculum.")
            self.curriculum_tensors = {}
            return
        
        self.curriculum_tensors = {}
        
        for component_id, data in CURRICULUM_COMPONENTS.items():
            phrases = []
            for level_phrases in data.values():
                phrases.extend(level_phrases)
            
            unique_phrases = list(set(phrases))
            embeddings = self.model_manager.embedder.encode(
                unique_phrases,
                convert_to_tensor=True
            )
            self.curriculum_tensors[component_id] = embeddings
        
        logger.info(f"Loaded {len(self.curriculum_tensors)} curriculum components")
    
    from typing import Optional  # already imported at top

    def process_job(self, job_id: int, text: str,
                    date_posted: Optional[str] = None) -> Tuple[Dict, float]:
        start_time = time.time()
        try:
            logger.debug(f"Processing job {job_id}")
            
            # 1. Extract with context awareness
            extraction_results = self.context_extractor.extract_with_context_awareness(text)
            
            bert_skills = extraction_results['bert']
            gpt_skills = extraction_results['gpt']
            bert_knowledge = extraction_results['bert_knowledge']
            gpt_knowledge = extraction_results['gpt_knowledge']
            agreement_scores = extraction_results['agreement_scores']
            
            # 2. Advanced fusion
            fused_skills = self.fusion_engine.fuse_skills_advanced(bert_skills, gpt_skills)
            fused_knowledge = self.fusion_engine.fuse_knowledge_advanced(bert_knowledge, gpt_knowledge)
            
            # 3. Calculate advanced coverage
            coverage_results = self.coverage_calculator.calculate_advanced_coverage(
                fused_skills, fused_knowledge
            )
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            # 4. Prepare results
            results = {
                'job_id': job_id,
                'text': text,
                'date_posted': date_posted,
                'skills': fused_skills,
                'knowledge': fused_knowledge,
                'extraction_results': extraction_results,  # Added for comprehensive analysis
                'extraction_metrics': {
                    'bert_skill_count': len(bert_skills),
                    'gpt_skill_count': len(gpt_skills),
                    'fused_skill_count': len(fused_skills),
                    'model_agreement': agreement_scores['overall'],
                    'match_count': agreement_scores.get('match_count', 0),
                    'extraction_time': extraction_time
                },
                'coverage_analysis': coverage_results
            }
            
            return results, extraction_time
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            extraction_time = time.time() - start_time
            # Return empty results on error
            return {
                'job_id': job_id,
                'text': text,
                'date_posted': date_posted,
                'skills': [],
                'knowledge': [],
                'extraction_results': {},
                'extraction_metrics': {
                    'bert_skill_count': 0,
                    'gpt_skill_count': 0,
                    'fused_skill_count': 0,
                    'model_agreement': 0.0,
                    'match_count': 0,
                    'extraction_time': extraction_time
                },
                'coverage_analysis': {
                    'metrics': {
                        'total_components': 0,
                        'covered_components': 0,
                        'coverage_percentage': 0.0,
                        'skill_coverage_pct': 0.0,
                        'knowledge_coverage_pct': 0.0,
                        'avg_skill_confidence': 0.0,
                        'avg_knowledge_confidence': 0.0,
                        'hard_skill_count': 0,
                        'soft_skill_count': 0,
                        'knowledge_count': 0
                    },
                    'missing_components': [],
                    'coverage_gaps': []
                }
            }, extraction_time
    
    def run(self, sample_size: int = None) -> None:
        """Run the complete advanced pipeline."""
        logger.info("🚀 Starting Advanced Skill Extraction Pipeline...")
        
        # Load job data (DataFrame)
        jobs_df = self.data_manager.load_job_data(sample_size)

        if jobs_df.empty:
            logger.warning("No jobs to process. Exiting.")
            return

        logger.info(f"Processing {len(jobs_df)} job descriptions...")

        pipeline_stats = {
            'total_jobs': len(jobs_df),
            'total_skills': 0,
            'total_knowledge': 0,
            'total_coverage': 0.0,
            'total_agreement': 0.0,
            'total_extraction_time': 0.0
        }

        count = 0
        
        for idx, row in jobs_df.iterrows():
            try:
                if idx > 0:
                    time.sleep(0.5)
                count = count + 1
                job_id = row['job_id']
                text = str(row['sentence_text'])
                date_posted = row.get('date_posted', None)

                results, extraction_time = self.process_job(job_id, text, date_posted)

                self.data_manager.save_results(job_id, results, extraction_time)
                ...
                if (count + 1) % 10 == 0:
                    logger.info(f"Processed {count + 1}/{len(jobs_df)} jobs")

            except Exception as e:
                logger.error(f"Failed to process job {row.get('job_id', idx)}: {e}")
                continue

        
        # Calculate averages
        pipeline_stats['avg_skills_per_job'] = pipeline_stats['total_skills'] / pipeline_stats['total_jobs'] if pipeline_stats['total_jobs'] > 0 else 0
        pipeline_stats['avg_knowledge_per_job'] = pipeline_stats['total_knowledge'] / pipeline_stats['total_jobs'] if pipeline_stats['total_jobs'] > 0 else 0
        pipeline_stats['avg_coverage'] = pipeline_stats['total_coverage'] / pipeline_stats['total_jobs'] if pipeline_stats['total_jobs'] > 0 else 0
        pipeline_stats['avg_agreement'] = pipeline_stats['total_agreement'] / pipeline_stats['total_jobs'] if pipeline_stats['total_jobs'] > 0 else 0
        pipeline_stats['avg_extraction_time'] = pipeline_stats['total_extraction_time'] / pipeline_stats['total_jobs'] if pipeline_stats['total_jobs'] > 0 else 0
        
        # Save summary
        self.data_manager.save_summary(pipeline_stats)
        
        logger.info("\n" + "="*60)
        logger.info("[✓] PIPELINE COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total jobs processed: {pipeline_stats['total_jobs']}")
        logger.info(f"Average skills per job: {pipeline_stats['avg_skills_per_job']:.2f}")
        logger.info(f"Average knowledge per job: {pipeline_stats['avg_knowledge_per_job']:.2f}")
        logger.info(f"Average coverage: {pipeline_stats['avg_coverage']:.2%}")
        logger.info(f"Average model agreement: {pipeline_stats['avg_agreement']:.2%}")
        logger.info(f"Average extraction time: {pipeline_stats['avg_extraction_time']:.2f}s")
        logger.info(f"\nOutput directory: {OUTPUT_DIR}")
        logger.info("\nNew Comprehensive Analysis Files:")
        logger.info("1. comprehensive_analysis.csv - Complete data for visualization")
        logger.info("2. model_comparison.csv - Side-by-side model comparison")
        logger.info("="*60)

# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():
    """Main entry point."""
    try:
        # Create pipeline instance
        pipeline = AdvancedSkillExtractionPipeline()
        
        # Initialize pipeline
        pipeline.initialize()
        
        # Run pipeline
        pipeline.run(sample_size=AdvancedPipelineConfig.SAMPLE_SIZE)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()