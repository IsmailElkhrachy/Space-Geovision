import unittest
import numpy as np
import validation as V

GROUPS = {"change": [2, 3], "no_change": [1]}
COUNTS = {1: 6000, 2: 540, 3: 190}


def make_points(rng, p_true, n=100, audit_frac=0.25, p_unc=(0.2, 0.45, 0.5)):
    """Phase 1: n points per class; 'uncertain' ones are more often truly change; all uncertain + a random audit get a hand label."""
    pts = []
    for h, pt in zip((1, 2, 3), p_true):
        for _ in range(n):
            unc = rng.random() < p_unc[h - 1]
            ch = rng.random() < (min(0.95, pt * 1.6) if unc else pt * 0.5)   # uncertain points: more change
            pts.append(dict(map_class=h, status="uncertain" if unc else "decided", truth="change" if ch else "no_change"))
    return pts


class TwoPhase(unittest.TestCase):
    def test_all_hand_equals_plain_estimate(self):
        rng = np.random.default_rng(0)
        pts = make_points(rng, (0.1, 0.5, 0.7))
        for p in pts:
            p["label"] = p["truth"]
        r = V.evaluate_two_phase(COUNTS, GROUPS, pts, 1.0, n_boot=200)
        samples = [(p["map_class"], p["truth"]) for p in pts]
        ref = V.evaluate(COUNTS, GROUPS, samples, 1.0)
        self.assertAlmostEqual(100 * r["oa"], 100 * ref["overall_accuracy"], places=6)

    def test_reweighting_removes_bias_of_hand_only_subset(self):
        rng = np.random.default_rng(1)
        pts = make_points(rng, (0.1, 0.5, 0.7), n=400)
        full = V.evaluate_two_phase(COUNTS, GROUPS, [dict(p, label=p["truth"]) for p in pts], 1.0, n_boot=50)
        sub = []
        for p in pts:
            hand = p["status"] == "uncertain" or rng.random() < 0.25
            sub.append(dict(p, label=p["truth"] if hand else None))
        r = V.evaluate_two_phase(COUNTS, GROUPS, sub, 1.0, n_boot=300)
        j = r["names"].index("change")
        self.assertAlmostEqual(r["ua"][j], full["ua"][j], delta=0.06)       # close to the all-labelled truth
        # naive pooling of hand points only would be biased upwards (uncertain points over-represented)
        hand_only = [(p["map_class"], p["label"]) for p in sub if p["label"]]
        naive = V.evaluate(COUNTS, GROUPS, hand_only, 1.0)
        nj = naive["names"].index("change")
        self.assertGreater(abs(naive["users_accuracy"][nj] - full["ua"][j]), abs(r["ua"][j] - full["ua"][j]))
        lo, hi = r["ua_ci95"][0][j], r["ua_ci95"][1][j]
        self.assertTrue(lo <= r["ua"][j] <= hi)

    def test_empty_cell_raises(self):
        pts = [dict(map_class=1, status="decided", label=None)] * 3 + [dict(map_class=2, status="decided", label="change")]
        with self.assertRaisesRegex(ValueError, "No hand label"):
            V.evaluate_two_phase({1: 10, 2: 5}, {"change": [2], "no_change": [1]}, pts, 1.0, n_boot=10)

    def test_format_and_unknown_label(self):
        pts = [dict(map_class=1, status="decided", label="no_change"), dict(map_class=2, status="decided", label="change")]
        r = V.evaluate_two_phase({1: 10, 2: 5}, {"change": [2], "no_change": [1]}, pts, 1.0, n_boot=20)
        self.assertIn("two-phase", V.format_two_phase(r))
        with self.assertRaises(ValueError):
            V.evaluate_two_phase({1: 10}, {"change": [2], "no_change": [1]},
                                 [dict(map_class=1, status="decided", label="water")], 1.0, n_boot=5)


if __name__ == "__main__":
    unittest.main()
