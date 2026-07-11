"""
ARGUS Agentic Engine - Module D: OSINT NLP Agent
Uses RoBERTa to analyze simulated digital footprints for predatory precursor language.
Outputs: digital_risk_score, risk_multiplier, flagged_phrases, category.
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("ARGUS.OSINT_NLP")

# ?????????????????????????????????????????????????????????????
# PREDATORY LANGUAGE LEXICON
# Phrase-level risk scores for zero-shot augmentation of RoBERTa
# ?????????????????????????????????????????????????????????????

PREDATORY_LEXICON: Dict[str, float] = {
    # Surveillance & following
    "follow someone without being noticed": 0.95,
    "track someone without knowing": 0.93,
    "tail a person": 0.90,
    "how to shadow someone": 0.88,
    "surveillance techniques": 0.75,
    "avoid being seen while following": 0.92,

    # Location targeting
    "areas with low police presence": 0.89,
    "places with no cameras": 0.85,
    "parks near schools at night": 0.91,
    "isolated locations": 0.72,
    "dark alleys": 0.68,
    "remote areas": 0.60,

    # Counter-surveillance
    "disable security camera": 0.87,
    "avoid facial recognition": 0.82,
    "how to not be recorded": 0.80,
    "hide from cameras": 0.78,
    "blind spots on cctv": 0.85,

    # Target selection language
    "vulnerable targets": 0.90,
    "easy victims": 0.93,
    "alone at night": 0.65,
    "approach from behind": 0.88,
    "attack from behind": 0.95,
    "catch someone off guard": 0.85,

    # Anonymity & concealment
    "tor browser dark web": 0.70,
    "anonymous location": 0.65,
    "untraceable": 0.72,
    "no witnesses": 0.88,
    "cover tracks": 0.80,

    # Predatory reconnaissance language
    "pattern of movement": 0.75,
    "routine schedule": 0.60,
    "when someone is home alone": 0.82,
    "how long does it take police to respond": 0.78,
    "self defense weapons legal carry": 0.45,
}

RISK_CATEGORIES = {
    "PREDATORY_RECONNAISSANCE": [
        "follow", "track", "tail", "shadow", "surveillance", "routine", "pattern"
    ],
    "COUNTER_SURVEILLANCE": [
        "camera", "recognition", "recorded", "blind spot", "cctv", "avoid detection"
    ],
    "LOCATION_TARGETING": [
        "police", "isolated", "dark", "remote", "alley", "park", "night"
    ],
    "TARGET_SELECTION": [
        "vulnerable", "victim", "alone", "approach", "attack", "behind", "guard"
    ],
    "CONCEALMENT": [
        "anonymous", "untraceable", "tor", "witnesses", "cover", "tracks"
    ],
}


class OSINTNLPAgent:
    """
    Module D: OSINT Digital Footprint Analyzer.

    Primary model: RoBERTa-base with a zero-shot sequence classification head.
    Augmented by: lexicon-based phrase matching for precision on predatory language.

    Pipeline:
      1. Lexicon match -> phrase-level risk scores + flagged phrases
      2. RoBERTa zero-shot on full text -> semantic risk score
      3. Combine scores -> final digital_risk_score and risk_multiplier
    """

    def __init__(self, use_gpu: bool = torch.cuda.is_available()):
        self.device = "cuda" if use_gpu else "cpu"
        self._roberta = None
        self._tokenizer = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Lazy-load RoBERTa zero-shot classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("[OSINT NLP] Loading RoBERTa zero-shot classifier...")
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-roberta-base",
                device=0 if self.device == "cuda" else -1,
                batch_size=8,
            )
            self._model_loaded = True
            logger.info("[OSINT NLP] RoBERTa loaded [OK]")
        except Exception as e:
            logger.warning(
                f"[OSINT NLP] RoBERTa load failed: {e}. "
                "Using lexicon-only mode (no transformer inference)."
            )
            self._classifier = None

    def _lexicon_analysis(self, texts: List[str]) -> Tuple[float, List[str], str]:
        """
        Scan texts for predatory lexicon phrases.
        Returns: (risk_score, flagged_phrases, risk_category)
        """
        combined_text = " ".join(texts).lower()
        flagged: List[Tuple[str, float]] = []

        for phrase, score in PREDATORY_LEXICON.items():
            if re.search(re.escape(phrase.lower()), combined_text):
                flagged.append((phrase, score))

        if not flagged:
            return 0.0, [], "NONE"

        # Aggregate risk: weighted max with a spread bonus
        scores = [s for _, s in flagged]
        max_score = max(scores)
        spread_bonus = min(len(flagged) * 0.03, 0.20)  # More flags = higher risk
        lexicon_score = float(np.clip(max_score + spread_bonus, 0.0, 1.0))

        # Determine primary risk category
        category_hits: Dict[str, int] = {cat: 0 for cat in RISK_CATEGORIES}
        for phrase, _ in flagged:
            for cat, keywords in RISK_CATEGORIES.items():
                if any(kw in phrase for kw in keywords):
                    category_hits[cat] += 1

        primary_category = max(category_hits, key=category_hits.get)
        if category_hits[primary_category] == 0:
            primary_category = "GENERAL_THREAT"

        flagged_phrases = [p for p, _ in sorted(flagged, key=lambda x: -x[1])]
        return lexicon_score, flagged_phrases, primary_category

    def _roberta_analysis(self, texts: List[str]) -> float:
        """
        RoBERTa zero-shot classification on combined text.
        Classifies against predatory vs benign intent labels.
        Returns risk score 0?1.
        """
        if not self._model_loaded or self._classifier is None:
            return 0.0

        try:
            combined = " | ".join(texts[:10])[:512]  # RoBERTa max 512 tokens
            candidate_labels = [
                "predatory criminal intent",
                "planning a violent act",
                "stalking or surveillance",
                "normal everyday activity",
                "innocent curiosity",
            ]

            result = self._classifier(
                combined,
                candidate_labels=candidate_labels,
                multi_label=False,
            )

            label_scores: Dict[str, float] = dict(
                zip(result["labels"], result["scores"])
            )

            # Risk labels vs benign labels
            risk_score = (
                label_scores.get("predatory criminal intent", 0.0) * 0.40 +
                label_scores.get("planning a violent act", 0.0) * 0.35 +
                label_scores.get("stalking or surveillance", 0.0) * 0.25
            )

            return float(np.clip(risk_score, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"[OSINT NLP] RoBERTa inference error: {e}")
            return 0.0

    def analyze(self, digital_footprint: Dict) -> Dict:
        """
        Full OSINT analysis pipeline for one subject's digital footprint.

        Args:
            digital_footprint: Dict with keys:
              - subject_id: str
              - search_history: List[str]
              - forum_posts: List[str]
              - social_media: List[str] (optional)

        Returns:
            Enrichment dict for alert payload.
        """
        subject_id = digital_footprint.get("subject_id", "UNKNOWN")
        all_texts = (
            digital_footprint.get("search_history", []) +
            digital_footprint.get("forum_posts", []) +
            digital_footprint.get("social_media", [])
        )

        if not all_texts:
            return self._empty_result(subject_id)

        logger.info(f"[OSINT NLP] Analyzing {len(all_texts)} text entries for {subject_id}")

        # ?? Lexicon Analysis ??????????????????????????????????
        lexicon_score, flagged_phrases, category = self._lexicon_analysis(all_texts)

        # ?? RoBERTa Semantic Analysis ?????????????????????????
        roberta_score = self._roberta_analysis(all_texts)

        # ?? Combined Score ????????????????????????????????????
        if self._model_loaded:
            # Weighted blend when both available
            digital_risk_score = 0.6 * lexicon_score + 0.4 * roberta_score
        else:
            digital_risk_score = lexicon_score

        # ?? Risk Multiplier ???????????????????????????????????
        # Amplifies vision score when digital footprint corroborates threat
        if digital_risk_score >= 0.80:
            risk_multiplier = 2.5
        elif digital_risk_score >= 0.60:
            risk_multiplier = 1.8
        elif digital_risk_score >= 0.40:
            risk_multiplier = 1.3
        elif digital_risk_score >= 0.20:
            risk_multiplier = 1.1
        else:
            risk_multiplier = 1.0

        result = {
            "subject_id": subject_id,
            "digital_risk_score": round(float(digital_risk_score), 4),
            "risk_multiplier": round(risk_multiplier, 2),
            "flagged_phrases": flagged_phrases[:10],  # Top 10
            "risk_category": category,
            "lexicon_score": round(float(lexicon_score), 4),
            "roberta_score": round(float(roberta_score), 4),
            "texts_analyzed": len(all_texts),
        }

        logger.info(
            f"[OSINT NLP] {subject_id}: "
            f"digital_risk={digital_risk_score:.3f} | "
            f"multiplier={risk_multiplier}x | "
            f"category={category} | "
            f"flagged={len(flagged_phrases)} phrases"
        )

        return result

    def _empty_result(self, subject_id: str) -> Dict:
        return {
            "subject_id": subject_id,
            "digital_risk_score": 0.0,
            "risk_multiplier": 1.0,
            "flagged_phrases": [],
            "risk_category": "NONE",
            "lexicon_score": 0.0,
            "roberta_score": 0.0,
            "texts_analyzed": 0,
        }

    def get_simulated_footprint(self, subject_id: str, is_high_risk: bool = False) -> Dict:
        """
        Returns a simulated digital footprint for a subject.
        Used when real OSINT data is unavailable.
        """
        osint_file = Path(__file__).parent.parent.parent / \
                     "ml_engine" / "data" / "synthetic_osint.json"

        if osint_file.exists():
            try:
                with open(osint_file) as f:
                    records = json.load(f)
                for rec in records:
                    if rec.get("subject_id") == subject_id:
                        return rec
            except Exception:
                pass

        # Generate on-the-fly if file not found - inline to avoid cross-package import
        import random
        _PREDATORY = [
            "how to follow someone without being noticed",
            "best parks near schools at night",
            "how to disable a security camera without being seen",
            "anonymous browsing methods tor vpn",
            "how to track someone location without them knowing",
            "areas with low police presence",
        ]
        _NORMAL = [
            "best pizza near me", "weather forecast tomorrow",
            "how to cook pasta carbonara", "gym workout plan",
            "python tutorial beginners", "cheap flights",
        ]

        if is_high_risk:
            queries = random.sample(_PREDATORY, k=min(5, len(_PREDATORY)))
        else:
            queries = random.sample(_NORMAL, k=min(5, len(_NORMAL)))

        return {
            "subject_id": subject_id,
            "search_history": queries,
            "forum_posts": [],
            "social_media": [],
        }


# ?????????????????????????????????????????????????????????????
# STANDALONE TEST
# ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    agent = OSINTNLPAgent()

    test_cases = [
        {
            "subject_id": "SUBJ-HIGH-RISK",
            "search_history": [
                "how to follow someone without being noticed",
                "areas with low police presence at night",
                "how to disable a security camera",
                "best approach to catch someone off guard",
                "avoid facial recognition cameras downtown",
            ],
            "forum_posts": ["Looking for isolated park areas open late at night"],
        },
        {
            "subject_id": "SUBJ-NORMAL",
            "search_history": [
                "best pizza places near me",
                "how to cook pasta carbonara",
                "weekend hiking trails",
                "python tutorial beginners",
            ],
            "forum_posts": [],
        },
    ]

    print("\n" + "="*60)
    print("  ARGUS - OSINT NLP Agent Test")
    print("="*60)

    for case in test_cases:
        result = agent.analyze(case)
        print(f"\nSubject: {result['subject_id']}")
        print(f"  Digital Risk Score: {result['digital_risk_score']:.3f}")
        print(f"  Risk Multiplier:    {result['risk_multiplier']}x")
        print(f"  Category:           {result['risk_category']}")
        print(f"  Flagged Phrases:    {result['flagged_phrases'][:3]}")
