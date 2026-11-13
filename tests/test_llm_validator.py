"""Unit tests for the LLM-report validator (run: python3 -m unittest discover -s tests -p 'test_*.py')."""
import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import llm_report as lr  # noqa: E402


def make_alerts(path):
    rows = []
    for v in (0, 1, 2):
        for k in range(10):
            rows.append({"ts_us": 1_700_000_000_000_000 + k, "vehicle": v, "role": "OBU",
                         "class": "DoS", "prob": 0.99, "top_can_id": "000", "id_entropy": 1.0})
    for k in range(4):
        rows.append({"ts_us": 1_700_000_000_100_000 + k, "vehicle": 5, "role": "OBU",
                     "class": "RPM", "prob": 0.9, "top_can_id": "316", "id_entropy": 4.0})
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        make_alerts(self.path)
        self.ev = lr.build_evidence(self.path)

    def tearDown(self):
        os.remove(self.path)

    def test_evidence(self):
        self.assertEqual(self.ev["total_alerts"], 34)
        self.assertEqual(self.ev["attacks"]["DoS"]["vehicles"], [0, 1, 2])
        self.assertEqual(self.ev["attacks"]["DoS"]["severity"], "critical")

    def test_grounded_draft_passes(self):
        checks = lr.validate(lr.template_draft(self.ev), self.ev)
        self.assertTrue(all(ok for _, ok, _ in checks), checks)

    def test_hallucinations_are_flagged(self):
        draft = lr.inject_errors(lr.template_draft(self.ev), self.ev)
        failed = [c for c, ok, _ in lr.validate(draft, self.ev) if not ok]
        joined = " ".join(failed)
        self.assertIn("FUZZING", joined)        # invented attack type
        self.assertIn("vehicles", joined)       # vehicle 99
        self.assertIn("alert_count", joined)    # wrong count
        self.assertIn("CAN IDs", joined)        # 7FF
        self.assertIn("summary numbers", joined)  # 4242

    def test_missing_attack_is_flagged(self):
        draft = lr.template_draft(self.ev)
        draft["incidents"] = [i for i in draft["incidents"] if i["attack_type"] != "RPM"]
        failed = [c for c, ok, _ in lr.validate(draft, self.ev) if not ok]
        self.assertTrue(any("completeness" in c for c in failed))

    def test_wrong_severity_is_flagged(self):
        draft = copy.deepcopy(lr.template_draft(self.ev))
        draft["incidents"][0]["severity"] = "low"
        failed = [c for c, ok, _ in lr.validate(draft, self.ev) if not ok]
        self.assertTrue(any("severity" in c for c in failed))

    def test_malformed_llm_output_is_flagged_not_crash(self):
        # real case seen with llama3.2:1b: "vehicles": 5 instead of a list
        draft = lr.template_draft(self.ev)
        draft["incidents"][0]["vehicles"] = 5
        draft["incidents"][0]["alert_count"] = "thirty"
        draft["incidents"][0]["top_can_ids"] = "000"
        draft["incidents"].append("not an object")
        checks = lr.validate(draft, self.ev)
        failed = " ".join(c for c, ok, _ in checks if not ok)
        self.assertIn("vehicles is a list", failed)
        self.assertIn("alert_count is an integer", failed)
        self.assertIn("top_can_ids is a list", failed)
        self.assertIn("is an object", failed)
        lr.render(draft, self.ev, checks, "test")   # rendering must not crash either

    def test_draft_without_incidents_list(self):
        checks = lr.validate({"summary": "x", "incidents": "none"}, self.ev)
        self.assertFalse(checks[0][1])


if __name__ == "__main__":
    unittest.main()
