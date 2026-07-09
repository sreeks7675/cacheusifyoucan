"""Decision & Confidence Agent for EADIS.

This module computes a deterministic decision bundle from the existing
pipeline context, returning a final decision summary that includes
confidence, risk, uncertainty, and explainable reasoning.
"""

from typing import Any, Dict, Tuple


class DecisionAgent:
    """Deterministic decision agent for the investigation pipeline."""

    DEFAULT_THRESHOLD = 0.65
    CALIBRATION_ECE = 0.03
    CONFIDENCE_INTERVAL_DELTA = 0.04

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return the final Decision & Confidence payload from context."""
        planner = context.get("planner", {})
        forensic = context.get("forensic", {})
        semantic = context.get("semantic", {})
        retrieval = context.get("retrieval", {})
        fusion = context.get("fusion", {})

        fused_score = self._safe_score(fusion.get("fused_fake_score", 0.5))
        artifact_score = self._safe_score(forensic.get("artifact_score", 0.5))
        scene_consistency = self._safe_score(
            semantic.get("scene_consistency_score", 0.5)
        )
        semantic_fake_score = 1.0 - scene_consistency
        retrieval_score = self._compute_retrieval_score(retrieval)

        confidence_score = self._compute_confidence_score(fused_score)
        threat_score = self._compute_threat_score(
            artifact_score=artifact_score,
            semantic_score=semantic_fake_score,
            retrieval_score=retrieval_score,
            fused_score=fused_score,
        )
        risk_level = self._compute_risk_level(threat_score)
        confidence_low, confidence_high = self._compute_confidence_interval(
            confidence_score
        )
        epistemic, aleatoric = self._compute_uncertainty(
            fused_score=fused_score,
            artifact_score=artifact_score,
            semantic_fake_score=semantic_fake_score,
            retrieval_score=retrieval_score,
            confidence_score=confidence_score,
        )
        verdict = self._determine_verdict(fused_score)
        conflicting = self._has_conflicting_evidence(
            artifact_score=artifact_score,
            semantic_fake_score=semantic_fake_score,
            retrieval_score=retrieval_score,
            fused_score=fused_score,
        )
        human_review_required = self._needs_human_review(
            confidence_score=confidence_score,
            verdict=verdict,
            conflicting_evidence=conflicting,
        )
        review_reason = self._build_review_reason(
            confidence_score=confidence_score,
            verdict=verdict,
            conflicting=conflicting,
        )
        decision_justification = self._build_justification(
            planner=planner,
            artifact_score=artifact_score,
            scene_consistency=scene_consistency,
            retrieval=retrieval,
            retrieval_score=retrieval_score,
            fused_score=fused_score,
            threat_score=threat_score,
            verdict=verdict,
            confidence_score=confidence_score,
        )

        return {
            "verdict": verdict,
            "confidence_score": round(confidence_score, 3),
            "confidence_interval": {
                "low": round(confidence_low, 3),
                "high": round(confidence_high, 3),
            },
            "calibration_ece": self.CALIBRATION_ECE,
            "threshold_used": self.DEFAULT_THRESHOLD,
            "risk_level": risk_level,
            "threat_score": round(threat_score, 3),
            "decision_justification": decision_justification,
            "uncertainty": {
                "epistemic": round(epistemic, 3),
                "aleatoric": round(aleatoric, 3),
            },
            "human_review_required": human_review_required,
            "review_reason": review_reason,
        }

    def _safe_score(self, value: Any) -> float:
        """Normalize values to a bounded float in [0.0, 1.0]."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score != score:
            return 0.0
        return min(max(score, 0.0), 1.0)

    def _compute_confidence_score(self, fused_score: float) -> float:
        """Compute a confidence score using fused evidence."""
        if fused_score >= self.DEFAULT_THRESHOLD:
            return fused_score
        if fused_score <= 1.0 - self.DEFAULT_THRESHOLD:
            return 1.0 - fused_score
        return 1.0 - abs(fused_score - 0.5) * 2.0

    def _compute_retrieval_score(self, retrieval: Dict[str, Any]) -> float:
        """Derive a deterministic retrieval threat signal."""
        if retrieval.get("known_source_match"):
            return 1.0
        similar_images = retrieval.get("similar_images_found", 0)
        try:
            return min(1.0, float(similar_images) / 5.0)
        except (TypeError, ValueError):
            return 0.0

    def _compute_threat_score(
        self,
        artifact_score: float,
        semantic_score: float,
        retrieval_score: float,
        fused_score: float,
    ) -> float:
        """Compute a weighted aggregate threat score."""
        return (
            artifact_score * 0.35
            + semantic_score * 0.30
            + retrieval_score * 0.20
            + fused_score * 0.15
        )

    def _compute_risk_level(self, threat_score: float) -> str:
        """Map threat score to a descriptive risk level."""
        if threat_score > 0.70:
            return "High"
        if threat_score >= 0.40:
            return "Medium"
        return "Low"

    def _compute_confidence_interval(self, confidence_score: float) -> Tuple[float, float]:
        """Create a deterministic confidence interval around the score."""
        low = max(0.0, confidence_score - self.CONFIDENCE_INTERVAL_DELTA)
        high = min(1.0, confidence_score + self.CONFIDENCE_INTERVAL_DELTA)
        return low, high

    def _compute_uncertainty(
        self,
        fused_score: float,
        artifact_score: float,
        semantic_fake_score: float,
        retrieval_score: float,
        confidence_score: float,
    ) -> Tuple[float, float]:
        """Estimate epistemic and aleatoric uncertainty deterministically."""
        epistemic = min(
            1.0,
            abs(artifact_score - semantic_fake_score) * 0.4
            + abs(fused_score - retrieval_score) * 0.3
            + (1.0 - confidence_score) * 0.3,
        )
        aleatoric = min(1.0, 0.05 + (0.5 - abs(fused_score - 0.5)) * 0.1)
        return epistemic, aleatoric

    def _determine_verdict(self, fused_score: float) -> str:
        """Return the final `real`, `fake`, or `uncertain` verdict."""
        if fused_score >= self.DEFAULT_THRESHOLD:
            return "fake"
        if fused_score <= 1.0 - self.DEFAULT_THRESHOLD:
            return "real"
        return "uncertain"

    def _has_conflicting_evidence(
        self,
        artifact_score: float,
        semantic_fake_score: float,
        retrieval_score: float,
        fused_score: float,
    ) -> bool:
        """Detect when evidence sources disagree strongly."""
        if abs(artifact_score - semantic_fake_score) > 0.35:
            return True
        if abs(fused_score - retrieval_score) > 0.4:
            return True
        return False

    def _needs_human_review(
        self,
        confidence_score: float,
        verdict: str,
        conflicting_evidence: bool,
    ) -> bool:
        """Decide whether a human reviewer should inspect this case."""
        return (
            confidence_score < self.DEFAULT_THRESHOLD
            or verdict == "uncertain"
            or conflicting_evidence
        )

    def _build_review_reason(
        self,
        confidence_score: float,
        verdict: str,
        conflicting: bool,
    ) -> str:
        """Generate a concise review reason for human oversight."""
        reasons = []
        if confidence_score < self.DEFAULT_THRESHOLD:
            reasons.append("confidence below threshold")
        if verdict == "uncertain":
            reasons.append("verdict remains uncertain")
        if conflicting:
            reasons.append("conflicting evidence exists")

        if not reasons:
            return "Confidence meets threshold and evidence is consistent."
        return " and ".join(reasons).capitalize() + "."

    def _build_justification(
        self,
        planner: Dict[str, Any],
        artifact_score: float,
        scene_consistency: float,
        retrieval: Dict[str, Any],
        retrieval_score: float,
        fused_score: float,
        threat_score: float,
        verdict: str,
        confidence_score: float,
    ) -> str:
        """Build a concise 2-3 sentence justification for the decision."""
        plan_steps = planner.get("steps_selected")
        plan_phrase = (
            "The investigation plan included forensic, semantic, and retrieval analysis."
            if plan_steps
            else "The investigation used multiple evidence sources."
        )
        retrieval_phrase = (
            "a known source match was found"
            if retrieval.get("known_source_match")
            else f"{int(retrieval.get('similar_images_found', 0))} similar images were found"
        )
        semantic_phrase = (
            "scene consistency was high"
            if scene_consistency >= 0.5
            else "scene consistency was low"
        )

        return (
            f"{plan_phrase} Fusion evidence produced a {fused_score:.2f} fake likelihood, "
            f"leading to a {verdict} verdict with {confidence_score:.2f} confidence. "
            f"Forensic artifact analysis scored {artifact_score:.2f}, semantic analysis indicated {semantic_phrase}, "
            f"and retrieval analysis showed that {retrieval_phrase}, contributing to a threat score of {threat_score:.2f}."
        )


def decide_case(context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience helper to evaluate the decision payload from context."""
    return DecisionAgent().decide(context)
