"""
Tests for Email Triage OpenEnv.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ---------------------------------------------------------------------------
# Data tests
# ---------------------------------------------------------------------------
class TestEmailData:
    def test_email_counts(self):
        from data.emails import TASK_EMAILS
        assert len(TASK_EMAILS["task_easy"])   == 10
        assert len(TASK_EMAILS["task_medium"]) == 10
        assert len(TASK_EMAILS["task_hard"])   == 15

    def test_all_gold_labels_present(self):
        from data.emails import TASK_EMAILS
        for tid, emails in TASK_EMAILS.items():
            for e in emails:
                assert "gold" in e, f"{e['email_id']} missing gold label"
                assert "priority" in e["gold"]
                assert "category" in e["gold"]
                assert "action" in e["gold"]

    def test_valid_priorities(self):
        from data.emails import TASK_EMAILS
        valid = {"urgent", "high", "normal", "low", "spam"}
        for tid, emails in TASK_EMAILS.items():
            for e in emails:
                assert e["gold"]["priority"] in valid, \
                    f"{e['email_id']} invalid priority: {e['gold']['priority']}"

    def test_required_fields(self):
        from data.emails import TASK_EMAILS
        required = {"email_id", "subject", "sender", "sender_domain", "body",
                    "timestamp", "thread_length", "has_attachments", "gold"}
        for tid, emails in TASK_EMAILS.items():
            for e in emails:
                for field in required:
                    assert field in e, f"{e['email_id']} missing field: {field}"


# ---------------------------------------------------------------------------
# Priority scoring tests
# ---------------------------------------------------------------------------
class TestPriorityScore:
    def test_exact_match(self):
        from data.emails import priority_partial_score
        for p in ["urgent", "high", "normal", "low", "spam"]:
            assert priority_partial_score(p, p) == 1.0

    def test_adjacent(self):
        from data.emails import priority_partial_score
        assert priority_partial_score("high", "urgent")  == 0.5
        assert priority_partial_score("urgent", "high")  == 0.5
        assert priority_partial_score("normal", "high")  == 0.5
        assert priority_partial_score("low", "normal")   == 0.5

    def test_two_away(self):
        from data.emails import priority_partial_score
        assert priority_partial_score("normal", "urgent") == 0.2
        assert priority_partial_score("low", "high")      == 0.2

    def test_far(self):
        from data.emails import priority_partial_score
        assert priority_partial_score("spam", "urgent") == 0.0
        assert priority_partial_score("urgent", "spam") == 0.0

    def test_invalid_returns_zero(self):
        from data.emails import priority_partial_score
        assert priority_partial_score("INVALID", "urgent") == 0.0


# ---------------------------------------------------------------------------
# Grader tests — easy
# ---------------------------------------------------------------------------
class TestEasyGrader:
    def setup_method(self):
        from data.emails import TASK_EMAILS
        self.emails = TASK_EMAILS["task_easy"]

    def _action(self, **kw):
        from app.graders import TriageAction
        defaults = dict(priority="normal", category="other", action="archive",
                        reply_draft=None, forward_to=None, tags=[])
        defaults.update(kw)
        return type("TriageAction", (), defaults)()

    def test_exact_priority_reward_1(self):
        from app.graders import grade_easy
        email = self.emails[0]  # gold: urgent
        a = self._action(priority="urgent")
        r = grade_easy(a, email)
        assert r.reward == 1.0

    def test_adjacent_priority_reward_05(self):
        from app.graders import grade_easy
        email = self.emails[0]  # gold: urgent
        a = self._action(priority="high")
        r = grade_easy(a, email)
        assert r.reward == 0.5

    def test_spam_reply_penalty(self):
        from app.graders import grade_easy
        email = self.emails[1]  # gold: spam
        a = self._action(priority="spam", action="reply")
        r = grade_easy(a, email)
        assert "Replying to spam email" in r.penalties
        assert r.reward < 1.0

    def test_reward_in_range(self):
        from app.graders import grade_easy
        for email in self.emails:
            a = self._action(priority=email["gold"]["priority"])
            r = grade_easy(a, email)
            assert 0.0 <= r.reward <= 1.0, f"reward out of range: {r.reward}"


# ---------------------------------------------------------------------------
# Grader tests — medium
# ---------------------------------------------------------------------------
class TestMediumGrader:
    def setup_method(self):
        from data.emails import TASK_EMAILS
        self.emails = TASK_EMAILS["task_medium"]

    def _action(self, **kw):
        defaults = dict(priority="normal", category="other", action="archive",
                        reply_draft=None, forward_to=None, tags=[])
        defaults.update(kw)
        return type("TriageAction", (), defaults)()

    def test_perfect_score(self):
        from app.graders import grade_medium
        email = self.emails[0]  # gold: urgent, legal, escalate
        a = self._action(priority="urgent", category="legal", action="escalate")
        r = grade_medium(a, email)
        assert r.reward == 1.0

    def test_delete_legal_penalty(self):
        from app.graders import grade_medium
        email = self.emails[0]  # gold: legal
        a = self._action(priority="urgent", category="legal", action="delete")
        r = grade_medium(a, email)
        assert "Deleting legal email" in r.penalties
        assert r.action_score == 0.0

    def test_partial_credit_action(self):
        from app.graders import grade_medium
        email = self.emails[0]  # gold: escalate
        a = self._action(priority="urgent", category="legal", action="forward")
        r = grade_medium(a, email)
        assert r.action_score == 0.6  # acceptable alternative

    def test_reward_range_all_emails(self):
        from app.graders import grade_medium
        for email in self.emails:
            g = email["gold"]
            a = self._action(priority=g["priority"], category=g["category"], action=g["action"])
            r = grade_medium(a, email)
            assert 0.0 <= r.reward <= 1.0


# ---------------------------------------------------------------------------
# Grader tests — hard
# ---------------------------------------------------------------------------
class TestHardGrader:
    def setup_method(self):
        from data.emails import TASK_EMAILS
        self.emails = TASK_EMAILS["task_hard"]

    def _action(self, **kw):
        defaults = dict(priority="normal", category="other", action="archive",
                        reply_draft=None, forward_to=None, tags=[])
        defaults.update(kw)
        return type("TriageAction", (), defaults)()

    def test_phishing_detected_high_score(self):
        from app.graders import grade_hard
        phishing_email = self.emails[3]  # h004 — phishing
        a = self._action(priority="spam", category="spam", action="delete")
        r = grade_hard(a, phishing_email)
        assert r.reward > 0.7, f"Phishing detection should score well, got {r.reward}"

    def test_reply_required_no_reply_action(self):
        from app.graders import grade_hard
        email = self.emails[1]  # h002 — reply required
        a = self._action(priority="urgent", category="complaint", action="archive")
        r = grade_hard(a, email)
        assert any("Reply required" in p for p in r.penalties)

    def test_good_reply_quality(self):
        from app.graders import grade_hard
        email = self.emails[1]  # h002 — needs apologize, refund, resolve, escalate, 48 hours
        a = self._action(
            priority="urgent", category="complaint", action="reply",
            reply_draft="We sincerely apologize for this delay on your refund request. "
                        "We will resolve this and escalate to our billing team within 48 hours."
        )
        r = grade_hard(a, email)
        assert r.reply_score is not None
        assert r.reply_score > 0.5

    def test_reward_in_range_all(self):
        from app.graders import grade_hard
        for email in self.emails:
            g = email["gold"]
            a = self._action(priority=g["priority"], category=g["category"], action=g["action"])
            r = grade_hard(a, email)
            assert 0.0 <= r.reward <= 1.0, f"{email['email_id']}: reward {r.reward} out of range"