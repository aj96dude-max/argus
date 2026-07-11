"""
ARGUS Agentic Engine - Threat Synthesizer
Converts raw ML + NLP scores into structured Predictive Threat Assessment Reports.
Output: human-readable narrative + machine-readable JSON for law enforcement review.
"""

from __future__ import annotations
import json
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("ARGUS.Synthesizer")

# Severity color mapping (used in HTML report)
SEVERITY_COLORS = {
    "CRITICAL":  "#FF1744",
    "HIGH":      "#FF6D00",
    "MODERATE":  "#FFD600",
    "LOW":       "#00E676",
    "NEGLIGIBLE": "#78909C",
}

SEVERITY_ICONS = {
    "CRITICAL":  "?",
    "HIGH":      "?",
    "MODERATE":  "?",
    "LOW":       "?",
    "NEGLIGIBLE": "?",
}

ANOMALY_DESCRIPTIONS = {
    "PREDATORY_GAIT":     "Subject exhibits slow, deliberate ambulation with directional persistence toward a target, consistent with stalking locomotion patterns.",
    "TAILING":            "Subject has maintained persistent spatial proximity to another individual over multiple frames, consistent with deliberate following behavior.",
    "AGGRESSIVE_POSTURE": "Subject displays elevated limb extension, forward lean, and reduced bilateral symmetry in posture, consistent with pre-assault positioning.",
    "LOITERING":          "Subject has remained stationary or performed repetitive low-radius movement within a defined area for an extended period without apparent purpose.",
    "RAPID_APPROACH":     "Subject performed an accelerated, direct approach trajectory toward another individual or sensitive location.",
    "CONCEALMENT":        "Subject is exhibiting body orientation and movement designed to minimize visual profile and avoid surveillance detection.",
    "NORMAL":             "No significant anomaly detected in subject's gait or behavioral pattern.",
}

ROUTE_DEVIATION_DESCRIPTIONS = {
    (0.0, 1.0):   "Subject is operating within their established Pattern of Life corridor.",
    (1.0, 2.0):   "Subject is at the periphery of their normal movement pattern (1?2? deviation).",
    (2.0, 3.0):   "Subject has deviated significantly from their established route (2?3?).",
    (3.0, float("inf")): "Subject is operating far outside their normal movement boundaries (>3?), indicating purposeful route deviation.",
}


def _describe_deviation(sigma: float) -> str:
    for (low, high), desc in ROUTE_DEVIATION_DESCRIPTIONS.items():
        if low <= sigma < high:
            return desc
    return "Route deviation data unavailable."


def _risk_category_description(category: str) -> str:
    descs = {
        "PREDATORY_RECONNAISSANCE": "Digital footprint contains multiple search queries consistent with target surveillance and predatory planning.",
        "COUNTER_SURVEILLANCE":     "Digital footprint shows active interest in evading law enforcement and disabling surveillance infrastructure.",
        "LOCATION_TARGETING":       "Digital footprint reveals interest in specific geographic areas associated with criminal opportunity or reduced law enforcement presence.",
        "TARGET_SELECTION":         "Digital footprint contains language indicative of victim selection criteria and approach methodology research.",
        "CONCEALMENT":              "Digital footprint shows interest in anonymization and evidence elimination techniques.",
        "GENERAL_THREAT":           "Digital footprint contains language with elevated threat indicators across multiple risk categories.",
        "NONE":                     "No predatory precursor language detected in digital footprint.",
    }
    return descs.get(category, "Digital risk category could not be determined.")


def compute_final_score(payload: Dict) -> float:
    """
    Recompute final composite score incorporating digital risk multiplier.
    Final = composite_vision_score * risk_multiplier (capped at 1.0)
    """
    vision_score = payload.get("composite_threat_index", 0.0)
    multiplier = payload.get("_risk_multiplier", 1.0)
    return min(float(vision_score) * float(multiplier), 1.0)


class ThreatSynthesizer:
    """
    Synthesizes vision + NLP analysis into a complete Predictive Threat Assessment Report.
    """

    def synthesize(
        self,
        alert_payload: Dict,
        nlp_result: Dict,
    ) -> Dict:
        """
        Main synthesis entry point.

        Args:
            alert_payload: Original ThreatPayload from ML pipeline
            nlp_result:    Result from OSINTNLPAgent.analyze()

        Returns:
            Enriched payload dict with report_narrative and all derived fields.
        """
        # ?? Merge NLP results into payload ????????????????????
        enriched = {**alert_payload}
        enriched["digital_risk_score"] = nlp_result.get("digital_risk_score", 0.0)
        enriched["flagged_phrases"]    = nlp_result.get("flagged_phrases", [])
        enriched["risk_category"]      = nlp_result.get("risk_category", "NONE")
        enriched["risk_multiplier"]    = nlp_result.get("risk_multiplier", 1.0)
        enriched["_risk_multiplier"]   = nlp_result.get("risk_multiplier", 1.0)

        # ?? Final Composite Score ?????????????????????????????
        final_score = compute_final_score(enriched)
        enriched["final_threat_score"] = round(final_score, 4)

        # Upgrade severity if multiplier pushed score higher
        final_severity = self._classify_severity(final_score)
        enriched["final_severity"] = final_severity

        # ?? Generate Narrative Report ?????????????????????????
        narrative = self._generate_narrative(enriched, nlp_result)
        enriched["report_narrative"] = narrative

        # ?? Generate Structured Report ????????????????????????
        report = self._build_report(enriched, nlp_result)
        enriched["structured_report"] = report

        logger.info(
            f"[Synthesizer] Report generated for {enriched.get('alert_id')} | "
            f"Final score: {final_score:.3f} | Severity: {final_severity}"
        )

        return enriched

    def _classify_severity(self, score: float) -> str:
        if score >= 0.92:   return "CRITICAL"
        if score >= 0.85:   return "HIGH"
        if score >= 0.65:   return "MODERATE"
        if score >= 0.40:   return "LOW"
        return "NEGLIGIBLE"

    def _generate_narrative(self, payload: Dict, nlp_result: Dict) -> str:
        """
        Generate a formal law-enforcement-style threat assessment narrative.
        """
        alert_id       = payload.get("alert_id", "N/A")
        subject_id     = payload.get("subject_id", "UNKNOWN")
        timestamp      = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
        camera_id      = payload.get("camera_id", "N/A")
        anomaly_type   = payload.get("anomaly_type", "UNKNOWN")
        gait_class     = payload.get("gait_class", "UNKNOWN")
        gait_conf      = payload.get("gait_confidence", 0.0)
        anomaly_score  = payload.get("anomaly_score", 0.0)
        face_stress    = payload.get("face_stress_index", 0.0)
        dev_sigma      = payload.get("route_deviation_sigma", 0.0)
        gnn_score      = payload.get("gnn_route_score", 0.0)
        hotspot_risk   = payload.get("hotspot_risk", 0.0)
        proximity      = payload.get("proximity_alert", False)
        coords         = payload.get("location_coords", [0.0, 0.0])
        digital_risk   = payload.get("digital_risk_score", 0.0)
        risk_category  = payload.get("risk_category", "NONE")
        flagged        = payload.get("flagged_phrases", [])
        final_score    = payload.get("final_threat_score", 0.0)
        final_severity = payload.get("final_severity", "UNKNOWN")
        multiplier     = payload.get("risk_multiplier", 1.0)

        # Parse timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            ts_formatted = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            ts_formatted = timestamp

        gait_desc = ANOMALY_DESCRIPTIONS.get(gait_class, "Unclassified behavioral anomaly.")
        route_desc = _describe_deviation(dev_sigma)
        digital_desc = _risk_category_description(risk_category)

        flagged_str = ""
        if flagged:
            formatted_phrases = "\n".join(f'         ? "{p}"' for p in flagged[:6])
            flagged_str = f"""
   Flagged Queries/Posts ({len(flagged)} total):
{formatted_phrases}"""

        narrative = f"""
??????????????????????????????????????????????????????????????
   ARGUS PREDICTIVE THREAT ASSESSMENT REPORT
   Project ARGUS - Multi-Modal AI Threat Detection System
   Lead Architect: Abdullah Javed (vagabond)
??????????????????????????????????????????????????????????????

   ALERT ID:        {alert_id}
   GENERATED:       {ts_formatted}
   CAMERA:          {camera_id}
   SUBJECT ID:      {subject_id}
   COORDINATES:     {coords[0]:.6f}?N, {coords[1]:.6f}?W
   FINAL SEVERITY:  {SEVERITY_ICONS.get(final_severity, '?')} {final_severity}
   FINAL SCORE:     {final_score:.1%} confidence

??????????????????????????????????????????????????????????????
   SECTION 1: PHYSICAL BEHAVIORAL INDICATORS
??????????????????????????????????????????????????????????????

   Primary Anomaly Classification:   {anomaly_type}
   Gait Class:                       {gait_class} ({gait_conf:.1%} confidence)
   CNN-LSTM Anomaly Score:           {anomaly_score:.1%}

   BEHAVIORAL ANALYSIS:
   {gait_desc}

   Facial Stress Index:    {face_stress:.1%}
   {"HIGH - Involuntary microexpressions detected consistent with heightened arousal" if face_stress > 0.6 else "MODERATE - Elevated stress indicators present" if face_stress > 0.35 else "LOW - No significant facial stress markers"}

??????????????????????????????????????????????????????????????
   SECTION 2: SPATIOTEMPORAL ROUTE ANALYSIS
??????????????????????????????????????????????????????????????

   Route Deviation:        {dev_sigma:.2f}? from established baseline
   GNN Route Score:        {gnn_score:.1%}
   Hotspot Proximity Risk: {hotspot_risk:.1%}
   Proximity Alert:        {"YES - Subject is within 5m of another tracked individual" if proximity else "NO"}

   ROUTE ANALYSIS:
   {route_desc}

   {"[WARN] PROXIMITY ALERT: Subject has entered critical proximity range of another tracked individual. Possible predatory convergence behavior." if proximity else ""}

??????????????????????????????????????????????????????????????
   SECTION 3: DIGITAL FOOTPRINT ANALYSIS (OSINT)
??????????????????????????????????????????????????????????????

   Digital Risk Score:     {digital_risk:.1%}
   Risk Category:          {risk_category}
   Risk Multiplier:        {multiplier}x
   Texts Analyzed:         {nlp_result.get("texts_analyzed", 0)}

   OSINT ANALYSIS:
   {digital_desc}
{flagged_str}

??????????????????????????????????????????????????????????????
   SECTION 4: THREAT SYNTHESIS & RECOMMENDATION
??????????????????????????????????????????????????????????????

   Vision Threat Score:    {payload.get('composite_threat_index', 0.0):.1%}
   Digital Risk Multiplier:{multiplier}x
   ?????????????????????????????????????????????????
   FINAL COMPOSITE SCORE:  {final_score:.1%}
   FINAL SEVERITY LEVEL:   {final_severity}

   RECOMMENDATION:
   {self._get_recommendation(final_severity, final_score, proximity, digital_risk)}

??????????????????????????????????????????????????????????????
   [ARGUS] This report is generated by an AI system and is
   intended to supplement, not replace, human judgment.
   All alerts require law enforcement review before action.
??????????????????????????????????????????????????????????????
""".strip()

        return narrative

    def _get_recommendation(
        self, severity: str, score: float,
        proximity: bool, digital_risk: float
    ) -> str:
        if severity == "CRITICAL":
            rec = "IMMEDIATE RESPONSE REQUIRED. Dispatch law enforcement to subject's last known coordinates."
            if proximity:
                rec += " Another individual is at risk. Prioritize proximity location."
            if digital_risk > 0.7:
                rec += " Digital footprint corroborates high-risk physical behavior. Warrant for digital records investigation recommended."
        elif severity == "HIGH":
            rec = "ELEVATED ALERT. Assign officer surveillance to subject. Gather additional physical evidence before intervention."
            if digital_risk > 0.5:
                rec += " Cross-reference digital indicators with case file."
        elif severity == "MODERATE":
            rec = "ADVISORY. Flag subject for enhanced monitoring. Do not escalate without additional corroborating evidence."
        elif severity == "LOW":
            rec = "WATCH STATUS. Log event for pattern analysis. No immediate action warranted."
        else:
            rec = "No action required. Event logged for baseline calibration."
        return rec

    def _build_report(self, payload: Dict, nlp_result: Dict) -> Dict:
        """Build machine-readable structured report dict."""
        return {
            "alert_id":            payload.get("alert_id"),
            "generated_at":        datetime.now(timezone.utc).isoformat(),
            "subject_id":          payload.get("subject_id"),
            "camera_id":           payload.get("camera_id"),
            "location":            payload.get("location_coords"),
            "final_severity":      payload.get("final_severity"),
            "final_threat_score":  payload.get("final_threat_score"),
            "physical_indicators": {
                "anomaly_type":       payload.get("anomaly_type"),
                "gait_class":         payload.get("gait_class"),
                "gait_confidence":    payload.get("gait_confidence"),
                "anomaly_score":      payload.get("anomaly_score"),
                "face_stress_index":  payload.get("face_stress_index"),
                "description":        ANOMALY_DESCRIPTIONS.get(payload.get("gait_class", ""), ""),
            },
            "spatiotemporal": {
                "route_deviation_sigma": payload.get("route_deviation_sigma"),
                "gnn_route_score":       payload.get("gnn_route_score"),
                "hotspot_risk":          payload.get("hotspot_risk"),
                "proximity_alert":       payload.get("proximity_alert"),
            },
            "digital_osint": {
                "digital_risk_score": nlp_result.get("digital_risk_score"),
                "risk_multiplier":    nlp_result.get("risk_multiplier"),
                "risk_category":      nlp_result.get("risk_category"),
                "flagged_phrases":    nlp_result.get("flagged_phrases", []),
                "texts_analyzed":     nlp_result.get("texts_analyzed", 0),
            },
            "recommendation": self._get_recommendation(
                payload.get("final_severity", "NEGLIGIBLE"),
                payload.get("final_threat_score", 0.0),
                payload.get("proximity_alert", False),
                nlp_result.get("digital_risk_score", 0.0),
            ),
        }


if __name__ == "__main__":
    import json
    from datetime import timezone

    # Test with a synthetic alert
    test_payload = {
        "alert_id": "ARGUS-20260624-DEMO",
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    }
    test_nlp = {
        "digital_risk_score": 0.79,
        "risk_multiplier": 2.4,
        "risk_category": "PREDATORY_RECONNAISSANCE",
        "flagged_phrases": [
            "follow someone without being noticed",
            "parks near schools at night",
            "areas with low police presence",
        ],
        "texts_analyzed": 8,
        "lexicon_score": 0.82,
        "roberta_score": 0.73,
    }

    synthesizer = ThreatSynthesizer()
    report = synthesizer.synthesize(test_payload, test_nlp)

    print(report["report_narrative"])
    print("\n[Structured Report JSON]")
    print(json.dumps(report["structured_report"], indent=2))
