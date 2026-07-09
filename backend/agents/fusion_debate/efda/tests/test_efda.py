"""
Test suite for the Evidence Fusion & Debate Agent.

Uses only the standard library `unittest` module — no pytest install
required. Run with:

    python -m unittest discover -s efda/tests -v

or, if you do have pytest installed, `pytest efda/tests` works too since
unittest.TestCase classes are pytest-compatible.
"""

import os
import tempfile
import unittest

from efda.agent import EvidenceFusionDebateAgent
from efda.critic import RuleBasedCritic
from efda.fusion import fuse
from efda.reliability_store import ReliabilityStore
from efda.schemas import (
    Claim,
    DisagreementType,
    EvidenceClaim,
    EvidenceItem,
    RemediationAction,
)
from efda.self_reflection import classify_disagreement, generate_dissent_report
from efda.uncertainty import compute_uncertainty


def make_claim(agent, claim, confidence, evidence=None, reliability_prior=0.6):
    return EvidenceClaim(
        agent=agent, claim=claim, confidence=confidence,
        evidence=evidence or [], reliability_prior=reliability_prior,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation(unittest.TestCase):
    def test_confidence_above_one_raises(self):
        with self.assertRaises(ValueError):
            EvidenceClaim(agent="A", claim=Claim.FAKE, confidence=1.5)

    def test_confidence_below_zero_raises(self):
        with self.assertRaises(ValueError):
            EvidenceClaim(agent="A", claim=Claim.FAKE, confidence=-0.1)

    def test_confidence_boundary_values_are_valid(self):
        EvidenceClaim(agent="A", claim=Claim.FAKE, confidence=0.0)
        EvidenceClaim(agent="A", claim=Claim.FAKE, confidence=1.0)  # should not raise

    def test_empty_agent_name_raises(self):
        with self.assertRaises(ValueError):
            EvidenceClaim(agent="", claim=Claim.FAKE, confidence=0.5)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class TestRuleBasedCritic(unittest.TestCase):
    def setUp(self):
        self.critic = RuleBasedCritic()

    def test_single_claim_survives_with_no_opposition(self):
        claims = [make_claim("A", Claim.FAKE, 0.7)]
        verdicts = self.critic.cross_examine(claims)
        self.assertTrue(verdicts[0].survived)
        self.assertEqual(verdicts[0].adjusted_confidence, 0.7)

    def test_weak_vague_claim_discounted_against_strong_specific_one(self):
        claims = [
            make_claim("Vague", Claim.FAKE, 0.5, evidence=[EvidenceItem(kind="note", description="looks off")]),
            make_claim("Specific", Claim.REAL, 0.9, evidence=[
                EvidenceItem(kind="similarity_score", description="exact db match", value=0.95)
            ]),
        ]
        verdicts = {v.agent: v for v in self.critic.cross_examine(claims)}
        self.assertFalse(verdicts["Vague"].survived)
        self.assertLess(verdicts["Vague"].adjusted_confidence, 0.5)
        self.assertTrue(verdicts["Specific"].survived)

    def test_equally_specific_and_confident_claims_both_survive(self):
        claims = [
            make_claim("A", Claim.FAKE, 0.8, evidence=[EvidenceItem(kind="bbox", description="x")]),
            make_claim("B", Claim.REAL, 0.8, evidence=[EvidenceItem(kind="bbox", description="y")]),
        ]
        verdicts = self.critic.cross_examine(claims)
        self.assertTrue(all(v.survived for v in verdicts))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class TestFusion(unittest.TestCase):
    def test_empty_claims_returns_uncertain_zero_confidence(self):
        result = fuse([], [])
        self.assertEqual(result.verdict, Claim.UNCERTAIN)
        self.assertEqual(result.confidence, 0.0)

    def test_all_agents_agree_fake_yields_fake_verdict(self):
        claims = [
            make_claim("A", Claim.FAKE, 0.8),
            make_claim("B", Claim.FAKE, 0.75),
            make_claim("C", Claim.FAKE, 0.7),
        ]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        result = fuse(claims, critic_verdicts)
        self.assertEqual(result.verdict, Claim.FAKE)
        self.assertGreater(result.confidence, 0.5)

    def test_perfectly_balanced_opposite_claims_yield_uncertain(self):
        claims = [
            make_claim("A", Claim.FAKE, 0.8, reliability_prior=0.6),
            make_claim("B", Claim.REAL, 0.8, reliability_prior=0.6),
        ]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        result = fuse(claims, critic_verdicts)
        self.assertEqual(result.verdict, Claim.UNCERTAIN)

    def test_all_uncertain_claims_yield_uncertain_verdict(self):
        claims = [make_claim("A", Claim.UNCERTAIN, 0.3), make_claim("B", Claim.UNCERTAIN, 0.4)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        result = fuse(claims, critic_verdicts)
        self.assertEqual(result.verdict, Claim.UNCERTAIN)

    def test_single_agent_claim_is_respected(self):
        claims = [make_claim("A", Claim.REAL, 0.9, reliability_prior=0.8)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        result = fuse(claims, critic_verdicts)
        self.assertEqual(result.verdict, Claim.REAL)

    def test_low_reliability_agent_contributes_less_weight(self):
        trusted = make_claim("Trusted", Claim.FAKE, 0.7, reliability_prior=0.9)
        untrusted = make_claim("Untrusted", Claim.REAL, 0.7, reliability_prior=0.1)
        critic_verdicts = RuleBasedCritic().cross_examine([trusted, untrusted])
        result = fuse([trusted, untrusted], critic_verdicts)
        self.assertEqual(result.verdict, Claim.FAKE)
        self.assertGreater(result.per_agent_weights["Trusted"], result.per_agent_weights["Untrusted"])


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

class TestUncertainty(unittest.TestCase):
    def test_single_claim_has_zero_uncertainty(self):
        claims = [make_claim("A", Claim.FAKE, 0.8)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        self.assertEqual(compute_uncertainty(claims, critic_verdicts), 0.0)

    def test_agreeing_agents_have_low_uncertainty(self):
        claims = [make_claim("A", Claim.FAKE, 0.85), make_claim("B", Claim.FAKE, 0.8)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        self.assertLess(compute_uncertainty(claims, critic_verdicts), 0.2)

    def test_opposing_agents_have_high_uncertainty(self):
        claims = [make_claim("A", Claim.FAKE, 0.85), make_claim("B", Claim.REAL, 0.8)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        self.assertGreater(compute_uncertainty(claims, critic_verdicts), 0.45)

    def test_uncertainty_always_bounded_zero_to_one(self):
        claims = [make_claim("A", Claim.FAKE, 1.0), make_claim("B", Claim.REAL, 1.0),
                  make_claim("C", Claim.UNCERTAIN, 0.9)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        u = compute_uncertainty(claims, critic_verdicts)
        self.assertTrue(0.0 <= u <= 1.0)


# ---------------------------------------------------------------------------
# Self-reflection
# ---------------------------------------------------------------------------

class TestSelfReflection(unittest.TestCase):
    def test_classifies_genuine_conflict(self):
        claims = [make_claim("A", Claim.FAKE, 0.8), make_claim("B", Claim.REAL, 0.75)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        self.assertEqual(classify_disagreement(claims, critic_verdicts), DisagreementType.GENUINE_CONFLICT)

    def test_classifies_weak_evidence(self):
        claims = [make_claim("A", Claim.FAKE, 0.2), make_claim("B", Claim.REAL, 0.15)]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        self.assertEqual(classify_disagreement(claims, critic_verdicts), DisagreementType.WEAK_EVIDENCE)

    def test_classifies_novel_pattern_when_retrieval_uninformative(self):
        claims = [
            make_claim("ForensicAnalysisAgent", Claim.FAKE, 0.65),
            make_claim("RetrievalComparisonAgent", Claim.UNCERTAIN, 0.2),
        ]
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        self.assertEqual(classify_disagreement(claims, critic_verdicts), DisagreementType.NOVEL_PATTERN)

    def test_max_reanalysis_attempts_forces_human_review(self):
        claims = [make_claim("A", Claim.FAKE, 0.2), make_claim("B", Claim.REAL, 0.15)]  # weak evidence
        critic_verdicts = RuleBasedCritic().cross_examine(claims)
        report_first_try = generate_dissent_report(claims, critic_verdicts, 0.6, reanalysis_attempts=0)
        report_after_max = generate_dissent_report(claims, critic_verdicts, 0.6, reanalysis_attempts=1)
        self.assertEqual(report_first_try.recommended_action, RemediationAction.REQUEST_REANALYSIS)
        self.assertEqual(report_after_max.recommended_action, RemediationAction.FLAG_HUMAN_REVIEW)


# ---------------------------------------------------------------------------
# Reliability store
# ---------------------------------------------------------------------------

class TestReliabilityStore(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)  # start from a clean, non-existent file

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_unknown_agent_returns_default_prior(self):
        store = ReliabilityStore(self.path)
        self.assertEqual(store.get("NewAgent"), 0.6)

    def test_update_moves_score_toward_target(self):
        store = ReliabilityStore(self.path)
        before = store.get("Agent", "case_a")
        after_correct = store.update("Agent", "case_a", was_correct=True)
        self.assertGreater(after_correct, before)

        after_incorrect = store.update("Agent", "case_a", was_correct=False)
        self.assertLess(after_incorrect, after_correct)

    def test_score_never_reaches_zero_or_one(self):
        store = ReliabilityStore(self.path)
        for _ in range(200):
            store.update("Agent", "case_a", was_correct=True)
        self.assertLess(store.get("Agent", "case_a"), 1.0)

        for _ in range(200):
            store.update("Agent", "case_a", was_correct=False)
        self.assertGreater(store.get("Agent", "case_a"), 0.0)

    def test_persists_across_instances(self):
        store1 = ReliabilityStore(self.path)
        store1.update("Agent", "case_a", was_correct=True)

        store2 = ReliabilityStore(self.path)  # simulates a fresh process/run
        self.assertEqual(store1.get("Agent", "case_a"), store2.get("Agent", "case_a"))

    def test_case_types_are_independent(self):
        store = ReliabilityStore(self.path)
        store.update("Agent", "face_swap", was_correct=True)
        store.update("Agent", "meme_edit", was_correct=False)
        self.assertNotEqual(store.get("Agent", "face_swap"), store.get("Agent", "meme_edit"))


# ---------------------------------------------------------------------------
# Full agent, end-to-end
# ---------------------------------------------------------------------------

class TestEvidenceFusionDebateAgent(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)
        self.agent = EvidenceFusionDebateAgent(reliability_store=ReliabilityStore(self.path))

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_empty_claims_list_does_not_crash(self):
        result = self.agent.run([], case_type="default")
        self.assertEqual(result.verdict, Claim.UNCERTAIN)

    def test_agreeing_agents_produce_no_dissent(self):
        claims = [
            make_claim("ForensicAnalysisAgent", Claim.FAKE, 0.85),
            make_claim("SemanticContextAgent", Claim.FAKE, 0.8),
        ]
        result = self.agent.run(claims, case_type="face_swap")
        self.assertEqual(result.verdict, Claim.FAKE)
        self.assertIsNone(result.dissent)
        self.assertEqual(result.risk_level, "high")

    def test_conflicting_agents_produce_dissent_and_no_forced_verdict_confidence(self):
        claims = [
            make_claim("ForensicAnalysisAgent", Claim.FAKE, 0.8),
            make_claim("RetrievalComparisonAgent", Claim.REAL, 0.78),
        ]
        result = self.agent.run(claims, case_type="face_swap")
        self.assertIsNotNone(result.dissent)
        self.assertIn(result.dissent.recommended_action,
                       (RemediationAction.REQUEST_REANALYSIS, RemediationAction.FLAG_HUMAN_REVIEW))

    def test_record_feedback_updates_store_and_is_reflected_next_run(self):
        claims = [make_claim("ForensicAnalysisAgent", Claim.FAKE, 0.8, reliability_prior=0.5)]
        self.agent.run(claims, case_type="face_swap")
        updated = self.agent.record_feedback("ForensicAnalysisAgent", "face_swap", was_correct=True)
        self.assertGreater(updated, 0.6)  # default prior is 0.6; should move up
        self.assertEqual(self.agent.reliability_store.get("ForensicAnalysisAgent", "face_swap"), updated)

    def test_evidence_summary_preserves_all_upstream_evidence(self):
        e1 = EvidenceItem(kind="bbox", description="lighting mismatch")
        e2 = EvidenceItem(kind="similarity_score", description="db match", value=0.9)
        claims = [
            make_claim("A", Claim.FAKE, 0.7, evidence=[e1]),
            make_claim("B", Claim.FAKE, 0.6, evidence=[e2]),
        ]
        result = self.agent.run(claims)
        self.assertIn(e1, result.evidence_summary)
        self.assertIn(e2, result.evidence_summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)

