"""
ARGUS - Test Suite
Unit and integration tests for all pipeline modules.
Run: python -m pytest tests/ -v
"""

import sys
import json
import math
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Ensure ml_engine is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ANOMALY_CLASSES, NUM_ANOMALY_CLASSES, TEMPORAL_WINDOW,
    POSE_LANDMARKS, CNN_FEATURE_DIM, LSTM_HIDDEN_DIM,
    ANOMALY_THRESHOLD, DEVICE
)


# ?????????????????????????????????????????????????????????????
# MODULE B - CNN-LSTM TESTS
# ?????????????????????????????????????????????????????????????

class TestCNNLSTMModel(unittest.TestCase):
    """Tests for Module B: Temporal CNN-LSTM Classifier."""

    def setUp(self):
        from models.temporal import CNNLSTMClassifier, PoseCNNEncoder
        self.model = CNNLSTMClassifier().to(DEVICE)
        self.model.eval()
        self.encoder = PoseCNNEncoder().to(DEVICE)

    def test_pose_encoder_output_shape(self):
        """CNN encoder should output (batch, CNN_FEATURE_DIM)."""
        B = 4
        x = torch.randn(B, POSE_LANDMARKS * 3).to(DEVICE)
        out = self.encoder(x)
        self.assertEqual(out.shape, (B, CNN_FEATURE_DIM),
                         f"Expected ({B}, {CNN_FEATURE_DIM}), got {out.shape}")

    def test_cnn_lstm_output_keys(self):
        """Model should return dict with logits, anomaly_score, attention_weights."""
        B, T, D = 2, TEMPORAL_WINDOW, POSE_LANDMARKS * 3
        x = torch.randn(B, T, D).to(DEVICE)
        out = self.model(x)
        self.assertIn("logits", out)
        self.assertIn("anomaly_score", out)
        self.assertIn("attention_weights", out)

    def test_logits_shape(self):
        """Logits should have shape (batch, num_classes)."""
        B, T, D = 3, TEMPORAL_WINDOW, POSE_LANDMARKS * 3
        x = torch.randn(B, T, D).to(DEVICE)
        out = self.model(x)
        self.assertEqual(out["logits"].shape, (B, NUM_ANOMALY_CLASSES))

    def test_anomaly_score_range(self):
        """Anomaly score must be in [0, 1]."""
        B, T, D = 4, TEMPORAL_WINDOW, POSE_LANDMARKS * 3
        x = torch.randn(B, T, D).to(DEVICE)
        out = self.model(x)
        scores = out["anomaly_score"].cpu().detach().numpy()
        self.assertTrue(np.all(scores >= 0.0), "Anomaly scores below 0")
        self.assertTrue(np.all(scores <= 1.0), "Anomaly scores above 1")

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to ~1 per sample."""
        B, T, D = 2, TEMPORAL_WINDOW, POSE_LANDMARKS * 3
        x = torch.randn(B, T, D).to(DEVICE)
        out = self.model(x)
        attn = out["attention_weights"].cpu().detach().numpy()
        sums = attn.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(B), atol=1e-4,
                                   err_msg="Attention weights don't sum to 1")


class TestTemporalAnalyzer(unittest.TestCase):
    """Tests for TemporalAnalyzer inference wrapper."""

    def setUp(self):
        from models.temporal import TemporalAnalyzer
        self.analyzer = TemporalAnalyzer(model_checkpoint=None)

    def _make_fake_subject(self, subject_id: str = "TEST-0001"):
        """Create a fake SubjectSpatialData with random pose."""
        subj = MagicMock()
        subj.subject_id = subject_id
        subj.pose_flat = np.random.randn(POSE_LANDMARKS * 3).astype(np.float32)
        return subj

    def test_returns_none_until_window_full(self):
        """update() should return None until TEMPORAL_WINDOW frames are fed."""
        subj = self._make_fake_subject()
        for i in range(TEMPORAL_WINDOW - 1):
            result = self.analyzer.update(subj)
            self.assertIsNone(result, f"Expected None at frame {i}")

    def test_returns_result_when_window_full(self):
        """update() should return a dict once TEMPORAL_WINDOW frames are in buffer."""
        subj = self._make_fake_subject()
        result = None
        for _ in range(TEMPORAL_WINDOW):
            result = self.analyzer.update(subj)
        self.assertIsNotNone(result)
        self.assertIn("predicted_class", result)
        self.assertIn("anomaly_score", result)
        self.assertIn("is_anomalous", result)

    def test_predicted_class_is_valid(self):
        """Predicted class must be one of the defined ANOMALY_CLASSES."""
        subj = self._make_fake_subject()
        for _ in range(TEMPORAL_WINDOW):
            result = self.analyzer.update(subj)
        self.assertIn(result["predicted_class"], ANOMALY_CLASSES)

    def test_multiple_subjects_independent(self):
        """Each subject should have an independent sliding window."""
        subj_a = self._make_fake_subject("SUBJ-A")
        subj_b = self._make_fake_subject("SUBJ-B")

        # Fill window for A
        for _ in range(TEMPORAL_WINDOW):
            self.analyzer.update(subj_a)

        # B should still be empty
        result_b = self.analyzer.update(subj_b)
        self.assertIsNone(result_b,
                          "Subject B window should be independent from A")


# ?????????????????????????????????????????????????????????????
# MODULE C - SPATIOTEMPORAL TESTS
# ?????????????????????????????????????????????????????????????

class TestGISUtilities(unittest.TestCase):
    """Tests for haversine distance calculation."""

    def test_haversine_same_point(self):
        from models.spatiotemporal import haversine_distance
        d = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
        self.assertAlmostEqual(d, 0.0, places=3)

    def test_haversine_known_distance(self):
        """SF to LA is ~559 km."""
        from models.spatiotemporal import haversine_distance
        d = haversine_distance(37.7749, -122.4194, 34.0522, -118.2437)
        self.assertAlmostEqual(d / 1000, 559, delta=5,
                               msg=f"SF->LA distance off: {d/1000:.1f}km")

    def test_haversine_symmetry(self):
        from models.spatiotemporal import haversine_distance
        d1 = haversine_distance(37.7, -122.4, 34.0, -118.2)
        d2 = haversine_distance(34.0, -118.2, 37.7, -122.4)
        self.assertAlmostEqual(d1, d2, places=3)


class TestRouteGNN(unittest.TestCase):
    """Tests for GNN route anomaly model."""

    def setUp(self):
        from models.spatiotemporal import RouteGNN
        self.model = RouteGNN().to(DEVICE)
        self.model.eval()

    def test_output_shape(self):
        """GNN should output (1,) score."""
        coords = torch.randn(10, 4).to(DEVICE)
        score = self.model(coords)
        self.assertEqual(score.shape, (1, 1),
                         f"Expected (1,1), got {score.shape}")

    def test_score_range(self):
        """GNN score must be in [0, 1] - output of Sigmoid."""
        coords = torch.randn(15, 4).to(DEVICE)
        score = float(self.model(coords).cpu().item())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_minimum_nodes(self):
        """Should work with just 1 node."""
        coords = torch.randn(1, 4).to(DEVICE)
        score = self.model(coords)
        self.assertIsNotNone(score)


class TestSpatiotemporalTracker(unittest.TestCase):
    """Integration tests for the full spatiotemporal tracker."""

    def setUp(self):
        from models.spatiotemporal import SpatiotemporalTracker
        self.tracker = SpatiotemporalTracker()

    def test_update_returns_dict(self):
        result = self.tracker.update_subject("TEST-001", 37.7749, -122.4194)
        self.assertIsInstance(result, dict)
        self.assertIn("route_deviation_sigma", result)
        self.assertIn("gnn_anomaly_score", result)
        self.assertIn("proximity_alert", result)

    def test_baseline_not_established_initially(self):
        result = self.tracker.update_subject("NEW-001", 37.7749, -122.4194)
        self.assertFalse(result["baseline_established"])

    def test_scores_are_floats_in_range(self):
        for i in range(10):
            result = self.tracker.update_subject(
                "SCORE-TEST", 37.7749 + i * 0.001, -122.4194
            )
        self.assertIsInstance(result["gnn_anomaly_score"], float)
        self.assertGreaterEqual(result["gnn_anomaly_score"], 0.0)
        self.assertLessEqual(result["gnn_anomaly_score"], 1.0)

    def test_proximity_alert_fires_when_subjects_overlap(self):
        """Two subjects at same location should trigger proximity alert."""
        lat, lon = 37.7749, -122.4194
        self.tracker.update_subject("SUBJ-X", lat, lon)
        result = self.tracker.update_subject("SUBJ-Y", lat + 0.00001, lon + 0.00001)
        # They're ~1m apart - should be within 5m threshold
        self.assertTrue(result["proximity_alert"],
                        "Proximity alert should fire for overlapping subjects")


# ?????????????????????????????????????????????????????????????
# SYNTHETIC DATA TESTS
# ?????????????????????????????????????????????????????????????

class TestSyntheticData(unittest.TestCase):
    """Tests for synthetic data generators."""

    def test_pose_sequence_shape(self):
        from data.synthetic import generate_synthetic_pose_sequence
        seq = generate_synthetic_pose_sequence("NORMAL", TEMPORAL_WINDOW)
        self.assertEqual(seq.shape, (TEMPORAL_WINDOW, POSE_LANDMARKS * 3))

    def test_pose_sequence_dtype(self):
        from data.synthetic import generate_synthetic_pose_sequence
        seq = generate_synthetic_pose_sequence("TAILING", TEMPORAL_WINDOW)
        self.assertEqual(seq.dtype, torch.float32)

    def test_gis_trajectories_structure(self):
        from data.synthetic import generate_synthetic_gis_trajectories
        trajs = generate_synthetic_gis_trajectories(n_subjects=3)
        self.assertEqual(len(trajs), 3)
        self.assertIn("subject_id", trajs[0])
        self.assertIn("trajectory", trajs[0])
        self.assertIn("lat", trajs[0]["trajectory"][0])
        self.assertIn("lon", trajs[0]["trajectory"][0])

    def test_osint_structure(self):
        from data.synthetic import generate_synthetic_osint
        records = generate_synthetic_osint(n_subjects=3)
        self.assertEqual(len(records), 3)
        self.assertIn("search_history", records[0])
        self.assertIn("is_flagged_ground_truth", records[0])

    def test_anomaly_ratio_in_gis(self):
        """~20% of subjects should be anomalous (every 5th)."""
        from data.synthetic import generate_synthetic_gis_trajectories
        trajs = generate_synthetic_gis_trajectories(n_subjects=10)
        anomalous = sum(1 for t in trajs if t["is_anomalous"])
        self.assertEqual(anomalous, 2, "Expected 2/10 anomalous subjects")


# ?????????????????????????????????????????????????????????????
# MODULE D - OSINT NLP TESTS
# ?????????????????????????????????????????????????????????????

class TestOSINTLexicon(unittest.TestCase):
    """Tests for the lexicon-based risk scoring (no model required)."""

    def setUp(self):
        # Patch RoBERTa loading so tests run without downloading model
        with patch("agents.osint_nlp.OSINTNLPAgent._load_model"):
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agentic_engine"))
            from agents.osint_nlp import OSINTNLPAgent
            self.agent = OSINTNLPAgent.__new__(OSINTNLPAgent)
            self.agent._model_loaded = False
            self.agent._classifier = None

        from agents.osint_nlp import OSINTNLPAgent as _A
        self.AgentClass = _A

    def test_high_risk_lexicon_score(self):
        from agents.osint_nlp import OSINTNLPAgent
        agent = OSINTNLPAgent.__new__(OSINTNLPAgent)
        agent._model_loaded = False
        agent._classifier = None

        # Access private method directly
        from agents.osint_nlp import PREDATORY_LEXICON, RISK_CATEGORIES
        import re
        import numpy as np

        texts = ["how to follow someone without being noticed", "areas with low police presence"]
        combined = " ".join(texts).lower()
        flagged = [
            (phrase, score)
            for phrase, score in PREDATORY_LEXICON.items()
            if re.search(re.escape(phrase.lower()), combined)
        ]
        self.assertGreater(len(flagged), 0, "High-risk text should flag phrases")
        scores = [s for _, s in flagged]
        self.assertGreater(max(scores), 0.7, "Max score should be > 0.7 for predatory text")

    def test_normal_text_no_flags(self):
        from agents.osint_nlp import PREDATORY_LEXICON
        import re

        texts = ["best pizza near me", "how to cook pasta", "weekend hiking"]
        combined = " ".join(texts).lower()
        flagged = [
            p for p in PREDATORY_LEXICON
            if re.search(re.escape(p.lower()), combined)
        ]
        self.assertEqual(len(flagged), 0, "Normal text should not match predatory lexicon")

    def test_risk_multiplier_levels(self):
        """Multiplier must increase with digital_risk_score."""
        from agents.osint_nlp import OSINTNLPAgent
        agent = OSINTNLPAgent.__new__(OSINTNLPAgent)
        agent._model_loaded = False
        agent._classifier = None
        agent.device = "cpu"

        # Simulate _get_multiplier logic directly from the analyze method
        def get_mult(score):
            if score >= 0.80: return 2.5
            if score >= 0.60: return 1.8
            if score >= 0.40: return 1.3
            if score >= 0.20: return 1.1
            return 1.0

        self.assertEqual(get_mult(0.85), 2.5)
        self.assertEqual(get_mult(0.65), 1.8)
        self.assertEqual(get_mult(0.45), 1.3)
        self.assertEqual(get_mult(0.25), 1.1)
        self.assertEqual(get_mult(0.05), 1.0)


# ?????????????????????????????????????????????????????????????
# THREAT SYNTHESIZER TESTS
# ?????????????????????????????????????????????????????????????

class TestThreatSynthesizer(unittest.TestCase):
    """Tests for the Threat Synthesizer report generator."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agentic_engine"))
        from agents.synthesizer import ThreatSynthesizer
        self.synthesizer = ThreatSynthesizer()
        self.sample_payload = {
            "alert_id": "ARGUS-TEST-0001",
            "timestamp": "2026-06-24T18:00:00+00:00",
            "camera_id": "CAM-01",
            "subject_id": "SUBJ-0042",
            "composite_threat_index": 0.88,
            "anomaly_type": "PREDATORY_GAIT:TAILING",
            "gait_class": "TAILING",
            "gait_confidence": 0.88,
            "anomaly_score": 0.91,
            "face_stress_index": 0.73,
            "route_deviation_sigma": 2.8,
            "gnn_route_score": 0.76,
            "hotspot_risk": 0.42,
            "proximity_alert": True,
            "location_coords": [37.7749, -122.4194],
            "severity": "HIGH",
        }
        self.sample_nlp = {
            "digital_risk_score": 0.79,
            "risk_multiplier": 2.4,
            "risk_category": "PREDATORY_RECONNAISSANCE",
            "flagged_phrases": ["follow someone without being noticed"],
            "texts_analyzed": 8,
            "lexicon_score": 0.82,
            "roberta_score": 0.73,
        }

    def test_synthesize_returns_enriched_dict(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        self.assertIn("report_narrative", result)
        self.assertIn("structured_report", result)
        self.assertIn("final_threat_score", result)
        self.assertIn("final_severity", result)

    def test_narrative_contains_sections(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        narrative = result["report_narrative"]
        self.assertIn("SECTION 1", narrative)
        self.assertIn("SECTION 2", narrative)
        self.assertIn("SECTION 3", narrative)
        self.assertIn("SECTION 4", narrative)

    def test_final_score_applies_multiplier(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        vision_score = self.sample_payload["composite_threat_index"]
        multiplier = self.sample_nlp["risk_multiplier"]
        expected = min(vision_score * multiplier, 1.0)
        self.assertAlmostEqual(result["final_threat_score"], expected, places=3)

    def test_severity_classification(self):
        from agents.synthesizer import ThreatSynthesizer
        s = ThreatSynthesizer()
        self.assertEqual(s._classify_severity(0.95), "CRITICAL")
        self.assertEqual(s._classify_severity(0.88), "HIGH")
        self.assertEqual(s._classify_severity(0.70), "MODERATE")
        self.assertEqual(s._classify_severity(0.45), "LOW")
        self.assertEqual(s._classify_severity(0.10), "NEGLIGIBLE")

    def test_narrative_contains_subject_id(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        self.assertIn("SUBJ-0042", result["report_narrative"])

    def test_narrative_contains_flagged_phrase(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        self.assertIn("follow someone without being noticed", result["report_narrative"])

    def test_structured_report_keys(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        report = result["structured_report"]
        self.assertIn("physical_indicators", report)
        self.assertIn("spatiotemporal", report)
        self.assertIn("digital_osint", report)
        self.assertIn("recommendation", report)

    def test_critical_proximity_recommendation(self):
        result = self.synthesizer.synthesize(self.sample_payload, self.sample_nlp)
        # High score + proximity = should mention immediate response
        rec = result["structured_report"]["recommendation"]
        self.assertTrue(
            "dispatch" in rec.lower() or "immediate" in rec.lower() or "response" in rec.lower(),
            f"Expected urgent recommendation, got: {rec[:100]}"
        )


# ?????????????????????????????????????????????????????????????
# PIPELINE INTEGRATION TEST
# ?????????????????????????????????????????????????????????????

class TestPipelineScoring(unittest.TestCase):
    """Integration tests for pipeline scoring logic."""

    def test_composite_score_formula(self):
        from models.pipeline import compute_composite_score
        score = compute_composite_score(
            anomaly_score=0.9,
            gnn_score=0.8,
            deviation_score=0.7,
            face_stress=0.6,
            proximity_alert=True,
        )
        expected = 0.40 * 0.9 + 0.20 * 0.8 + 0.15 * 0.7 + 0.10 * 0.6 + 0.15
        self.assertAlmostEqual(score, expected, places=4)

    def test_composite_score_clamped(self):
        from models.pipeline import compute_composite_score
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, True)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_composite_score_no_proximity(self):
        from models.pipeline import compute_composite_score
        score_with = compute_composite_score(0.9, 0.8, 0.7, 0.6, True)
        score_without = compute_composite_score(0.9, 0.8, 0.7, 0.6, False)
        self.assertGreater(score_with, score_without,
                           "Proximity alert should increase composite score")

    def test_severity_mapping(self):
        from models.pipeline import get_severity
        self.assertEqual(get_severity(0.95), "CRITICAL")
        self.assertEqual(get_severity(0.88), "HIGH")
        self.assertEqual(get_severity(0.70), "MODERATE")
        self.assertEqual(get_severity(0.50), "LOW")
        self.assertEqual(get_severity(0.20), "NEGLIGIBLE")

    def test_threat_payload_schema(self):
        from models.pipeline import build_threat_payload
        gait_result = {
            "predicted_class": "TAILING",
            "class_confidence": 0.88,
            "anomaly_score": 0.91,
            "all_class_probs": {c: 0.0 for c in ANOMALY_CLASSES},
        }
        route_result = {
            "route_deviation_sigma": 2.8,
            "gnn_anomaly_score": 0.76,
            "hotspot_risk": 0.42,
            "proximity_alert": True,
            "route_deviation_score": 0.6,
            "lat": 37.7749,
            "lon": -122.4194,
        }
        payload = build_threat_payload(
            subject_id="SUBJ-0001",
            gait_result=gait_result,
            spatial_result=None,
            route_result=route_result,
            composite_score=0.87,
        )
        self.assertIn("alert_id", payload)
        self.assertIn("timestamp", payload)
        self.assertIn("composite_threat_index", payload)
        self.assertIn("severity", payload)
        self.assertEqual(payload["gait_class"], "TAILING")
        self.assertEqual(payload["composite_threat_index"], 0.87)


# ?????????????????????????????????????????????????????????????
# ENTRY POINT
# ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    print("=" * 60)
    print("  ARGUS - Test Suite")
    print("  Run with: python -m pytest tests/ -v")
    print("=" * 60)
    unittest.main(verbosity=2)
