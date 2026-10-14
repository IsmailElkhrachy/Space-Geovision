import csv, os, tempfile, types, unittest
import ee_label as E


def probs(**kw):
    p = {b: 0.0 for b in E.DW_BANDS}; p.update(kw); return p


class Decide(unittest.TestCase):
    def test_rules(self):
        built = probs(built=0.8, bare=0.2); bare = probs(bare=0.9, built=0.1)
        veg = probs(trees=0.3, grass=0.3, crops=0.2, bare=0.2)   # grouped vegetation 0.8
        self.assertEqual(E.decide(bare, built)[0], "change")
        self.assertEqual(E.decide(built, built)[0], "no_change")
        self.assertEqual(E.decide(veg, veg)[0], "no_change")          # different DW classes, same group
        self.assertEqual(E.decide(veg, built)[0], "change")
        self.assertEqual(E.decide(probs(built=0.45, bare=0.4, water=0.15), built)[0], "uncertain")
        self.assertEqual(E.decide(None, built)[0], "uncertain")
        self.assertIn("built", E.decide(bare, built)[1])

    def test_missing_band_is_uncertain(self):
        p = probs(built=1.0); p["bare"] = None
        self.assertEqual(E.decide(p, probs(built=1.0))[0], "uncertain")

    def test_keeps_human_labels(self):
        rows = [dict(id=1, lon=0, lat=0, ref_label="change", note=""), dict(id=2, lon=0, lat=0, ref_label="", note="")]
        P1 = {1: probs(bare=1.0), 2: probs(bare=1.0)}; P2 = {1: probs(bare=1.0), 2: probs(built=1.0)}
        cnt = E.label_rows(rows, P1, P2)
        self.assertEqual(rows[0]["ref_label"], "change"); self.assertEqual(rows[1]["ref_label"], "change")
        self.assertEqual(cnt["kept (already labelled)"], 1)
        E.label_rows(rows, P1, P2, overwrite=True)
        self.assertEqual(rows[0]["ref_label"], "no_change")


class FakeEE(unittest.TestCase):
    def make_ee(self, table, n=3):
        ee = types.SimpleNamespace(); seen = {}
        class V:
            def __init__(s, v): s.v = v
            def getInfo(s): return s.v
        class Col:
            def __init__(s, start=None): s.start = start
            def filterDate(s, a, b): seen["win"] = (a, b); return Col(a)
            def filterBounds(s, g): seen["bounds"] = g; return s
            def size(s): return V(n)
            def select(s, b): return s
            def mean(s): return Img(s.start)
        class Img:
            def __init__(s, st): s.st = st
            def reduceRegions(s, collection, reducer, scale):
                return V({"features": [{"properties": dict(table[s.st][f.props["pid"]], pid=f.props["pid"])} for f in collection.feats]})
        class Geom:
            def __init__(s, c): s.c = c
            def buffer(s, m): seen["buf"] = m; return s
        ee.Geometry = types.SimpleNamespace(Point=lambda c: Geom(c), MultiPoint=lambda c: c)
        ee.Feature = lambda g, p: types.SimpleNamespace(props=p)
        ee.FeatureCollection = lambda f: types.SimpleNamespace(feats=f)
        ee.Reducer = types.SimpleNamespace(mean=lambda: None)
        ee.ImageCollection = lambda name: Col()
        return ee, seen

    def test_fetch(self):
        table = {"2020-01-01": {1: probs(bare=1.0), 2: probs(built=1.0)}}
        ee, seen = self.make_ee(table)
        rows = [dict(id=1, lon=44.5, lat=17.6), dict(id=2, lon=44.6, lat=17.5)]
        n, out = E.fetch_probs(ee, rows, "2020-02-15", 45, 15)
        self.assertEqual(n, 3); self.assertEqual(seen["win"], ("2020-01-01", "2020-04-01"))
        self.assertEqual(seen["buf"], 15); self.assertEqual(out[2]["built"], 1.0)

    def test_no_images(self):
        ee, _ = self.make_ee({}, n=0)
        self.assertEqual(E.fetch_probs(ee, [dict(id=1, lon=0, lat=0)], "2020-02-15", 45, 15), (0, {}))

    def test_uncertain_summary(self):
        rows = [dict(id=1, ref_label="uncertain"), dict(id=2, ref_label="change"), dict(id=3, ref_label="uncertain")]
        P1 = {1: probs(built=0.4, bare=0.35, water=0.25), 3: None}; P2 = {1: probs(bare=0.9)}
        out = E.uncertain_summary(rows, P1, P2)
        self.assertIn("built -> bare: 1", out); self.assertIn("no data: 1", out)

    def test_audit_todo_merge(self):
        rows = [dict(id=i, lon=0, lat=0, ref_label=l, note="DW x")
                for i, l in enumerate(["change", "no_change", "uncertain", "change", "no_change", "uncertain"], 1)]
        audit = E.pick_audit(rows, 2)
        self.assertEqual(len(audit), 2); self.assertTrue(all(rows[i - 1]["ref_label"] != "uncertain" for i in audit))
        todo = E.manual_todo(rows, audit)
        self.assertEqual(len(todo), 4)
        self.assertTrue(all(r["ref_label"] == "" and r["note"] == "" for r in todo))
        hand = [dict(r, ref_label="change") for r in todo]
        final, st = E.merge_labels(rows, hand, audit)
        self.assertEqual(st["sources"], {"hand": 4, "automatic": 2})
        self.assertEqual(st["audit_n"], 2)
        exp = sum(1 for i in audit if rows[i - 1]["ref_label"] == "change")
        self.assertEqual(st["audit_agree"], exp)
        self.assertEqual(final[2]["ref_label"], "change")        # uncertain resolved by hand
        self.assertIn("agreement", E.format_audit(st))

    def test_crosstab(self):
        d = tempfile.mkdtemp(); k = os.path.join(d, "k.csv")
        with open(k, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "row", "col", "map_class"]); w.writerow([1, 0, 0, 2]); w.writerow([2, 0, 0, 1])
        rows = [dict(id=1, ref_label="change"), dict(id=2, ref_label="no_change")]
        t = E.crosstab(rows, k)
        self.assertIn("map class", t); self.assertEqual(len(t.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
