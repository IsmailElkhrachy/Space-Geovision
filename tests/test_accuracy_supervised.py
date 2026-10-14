import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import accuracy as acc
import supervised as sup

# Table 4 of the rejected manuscript (rows = reference, columns = map)
T4 = np.array([[143, 2, 0, 8, 1], [4, 86, 1, 2, 9], [0, 0, 47, 1, 0], [11, 1, 2, 125, 3], [2, 7, 0, 1, 44]])


class TestAccuracy(unittest.TestCase):
    def test_paper_table4_is_89_percent_not_94(self):
        s = acc.summarize(T4)
        self.assertEqual(s["n"], 500)
        self.assertAlmostEqual(s["overall_accuracy"], 0.890, places=3)
        self.assertAlmostEqual(s["kappa"], 0.856, places=3)

    def test_pa_ua_definitions(self):
        s = acc.summarize(T4)
        self.assertAlmostEqual(s["producers_accuracy"][0], 143 / 154)   # row total
        self.assertAlmostEqual(s["users_accuracy"][0], 143 / 160)       # column total

    def test_confusion_matrix_from_labels(self):
        cm, labels = acc.confusion_matrix([1, 1, 2, 2, 3], [1, 2, 2, 2, 1])
        self.assertEqual(labels, [1, 2, 3])
        self.assertEqual(cm.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])

    def test_perfect_and_chance_kappa(self):
        self.assertAlmostEqual(acc.kappa(np.diag([10, 20, 30])), 1.0)
        self.assertAlmostEqual(acc.kappa(np.full((2, 2), 25)), 0.0)

    def test_wilson_interval_contains_estimate(self):
        lo, hi = acc.wilson_interval(445, 500)
        self.assertLess(lo, 0.89); self.assertGreater(hi, 0.89)

    def test_report_and_csv(self):
        import tempfile, os
        txt = acc.format_report(T4, [1, 2, 3, 4, 5], {1: "Urban"})
        self.assertIn("Overall accuracy: 89.0 %", txt)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cm.csv"); acc.save_csv(T4, [1, 2, 3, 4, 5], p)
            with open(p) as fh:
                self.assertIn("kappa,0.8", fh.read())


class TestBlockedSplit(unittest.TestCase):
    def test_no_block_shared_between_train_and_test(self):
        rng = np.random.default_rng(0)
        rows, cols = rng.integers(0, 1000, 5000), rng.integers(0, 1000, 5000)
        tr, te = acc.blocked_split(rows, cols, np.zeros(5000), block_px=100, test_fraction=0.3)
        ids = acc.block_ids(rows, cols, 100)
        self.assertFalse(set(ids[tr]) & set(ids[te]))
        self.assertEqual(len(tr) + len(te), 5000)

    def test_single_block_rejected(self):
        with self.assertRaises(ValueError):
            acc.blocked_split(np.arange(10), np.arange(10), np.zeros(10), block_px=1000)


class TestSupervised(unittest.TestCase):
    def synthetic(self, seed=0):
        """Two-band scene: left half class 1, right half class 2, plus noise."""
        rng = np.random.default_rng(seed)
        h, w = 60, 80
        stack = np.zeros((2, h, w), dtype=np.float32)
        stack[0, :, :40] = 0.2; stack[0, :, 40:] = 0.8
        stack[1, :, :40] = 0.7; stack[1, :, 40:] = 0.3
        stack += rng.normal(0, 0.05, stack.shape).astype(np.float32)
        truth = np.zeros((h, w), dtype=np.uint8); truth[:, :40] = 1; truth[:, 40:] = 2
        return stack, truth

    def test_rf_learns_and_evaluates_on_heldout(self):
        stack, truth = self.synthetic()
        lab = np.zeros_like(truth)
        lab[::3, ::3] = truth[::3, ::3]
        X, y, rows, cols = sup.sample_stack(stack, lab)
        tr, te = acc.blocked_split(rows, cols, y, block_px=10, test_fraction=0.3)
        model, ev = sup.train_and_evaluate(X[tr], y[tr], X[te], y[te], sup.ClassifierConfig(model="rf", n_trees=30))
        self.assertGreater(ev.summary["overall_accuracy"], 0.95)
        self.assertEqual(ev.n_val, len(te))
        self.assertIsNotNone(ev.oob_score)
        pred = sup.predict_raster(model, stack)
        self.assertEqual(pred.shape, truth.shape)
        self.assertGreater((pred == truth).mean(), 0.95)

    def test_svm_rbf_tuned(self):
        stack, truth = self.synthetic(1)
        lab = np.zeros_like(truth); lab[::4, ::4] = truth[::4, ::4]
        X, y, rows, cols = sup.sample_stack(stack, lab)
        tr, te = acc.blocked_split(rows, cols, y, block_px=10, test_fraction=0.3)
        _, ev = sup.train_and_evaluate(X[tr], y[tr], X[te], y[te], sup.ClassifierConfig(model="svm_rbf"))
        self.assertGreater(ev.summary["overall_accuracy"], 0.95)
        self.assertIn("svc__C", ev.best_params)

    def test_random_labels_give_chance_accuracy(self):
        """Regression test for the old GUI code path that trained on np.random labels."""
        stack, _ = self.synthetic(2)
        rng = np.random.default_rng(3)
        lab = rng.integers(1, 3, stack.shape[1:]).astype(np.uint8)
        X, y, rows, cols = sup.sample_stack(stack, lab)
        tr, te = acc.blocked_split(rows, cols, y, block_px=10, test_fraction=0.3)
        _, ev = sup.train_and_evaluate(X[tr], y[tr], X[te], y[te], sup.ClassifierConfig(n_trees=30))
        self.assertLess(ev.summary["overall_accuracy"], 0.65)
        self.assertLess(abs(ev.summary["kappa"]), 0.3)

    def test_nan_pixels_are_nodata(self):
        stack, truth = self.synthetic()
        lab = np.zeros_like(truth); lab[::3, ::3] = truth[::3, ::3]
        X, y, *_ = sup.sample_stack(stack, lab)
        model, _ = sup.train_and_evaluate(X, y, X, y, sup.ClassifierConfig(n_trees=10))
        stack[:, 0, 0] = np.nan
        self.assertEqual(sup.predict_raster(model, stack)[0, 0], 0)

    def test_single_class_rejected(self):
        with self.assertRaises(ValueError):
            sup.train_and_evaluate(np.zeros((5, 2)), np.ones(5), np.zeros((2, 2)), np.ones(2), sup.ClassifierConfig())


if __name__ == "__main__":
    unittest.main()


class TestBinaryAndInstance(unittest.TestCase):
    def test_binary_metrics_known_values(self):
        ref = np.zeros((10, 10), bool); ref[:5, :5] = True            # 25 px
        pred = np.zeros((10, 10), bool); pred[:5, :4] = True; pred[8:, 8:] = True   # 20 tp, 4 fp
        m = acc.binary_metrics(pred, ref)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (20, 4, 5, 71))
        self.assertAlmostEqual(m["iou"], 20 / 29); self.assertAlmostEqual(m["precision"], 20 / 24)
        self.assertAlmostEqual(m["recall"], 20 / 25)

    def test_instance_matching(self):
        ref = np.zeros((20, 20), int); ref[2:8, 2:8] = 1; ref[12:18, 12:18] = 2
        pred = np.zeros((20, 20), int); pred[2:8, 2:8] = 7; pred[0:3, 15:18] = 9       # one TP, one FP, one FN
        m = acc.instance_metrics(pred, ref)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))


class TestWater(unittest.TestCase):
    def test_otsu_water_on_synthetic_sigma0(self):
        import water_mapping as wm
        rng = np.random.default_rng(0)
        db = rng.normal(-8, 2, (200, 200)).astype(np.float32)           # land
        truth = np.zeros((200, 200), bool); truth[50:120, 60:150] = True
        db[truth] = rng.normal(-21, 1.5, truth.sum())                     # water
        mask, info = wm.map_water(db, "otsu")
        self.assertTrue(-21 < info["threshold_db"] < -8)
        m = wm.validate_water(mask, truth)
        self.assertGreater(m["iou"], 0.95); self.assertNotIn("warning", info)

    def test_dn_in_db_is_flagged(self):
        import water_mapping as wm
        dn = np.random.default_rng(1).integers(300, 65000, (100, 100)).astype(np.float32)
        _, info = wm.map_water(10 * np.log10(dn), "otsu")
        self.assertIn("warning", info)

    def test_clean_mask_removes_small_objects(self):
        import water_mapping as wm
        m = np.zeros((50, 50), bool); m[5:25, 5:25] = True; m[40, 40] = True
        out = wm.clean_mask(m, min_pixels=10)
        self.assertTrue(out[10, 10]); self.assertFalse(out[40, 40])


class TestFlood(unittest.TestCase):
    def test_flood_classes(self):
        import water_mapping as wm
        before = np.zeros((10, 10), bool); before[:, :3] = True               # river
        after = np.zeros((10, 10), bool); after[:, :6] = True                 # river + flooded strip
        valid = np.ones((10, 10), bool); valid[0, :] = False
        cls, st = wm.flood_extent(before, after, valid, pixel_area_m2=100.0)
        self.assertEqual(st["counts"][1], 9 * 3); self.assertEqual(st["counts"][2], 9 * 3)
        self.assertEqual(st["counts"][3], 0); self.assertTrue((cls[0] == 255).all())
        self.assertAlmostEqual(st["area_ha"][2], 27 * 100 / 1e4)

    def test_receded_and_shape_check(self):
        import water_mapping as wm
        cls, st = wm.flood_extent(np.ones((4, 4), bool), np.zeros((4, 4), bool))
        self.assertEqual(st["counts"][3], 16)
        with self.assertRaises(ValueError):
            wm.flood_extent(np.ones((4, 4)), np.ones((5, 4)))


class TestZeroFill(unittest.TestCase):
    def test_zero_db_fill_does_not_disturb_water_threshold(self):
        """Regression: SNAP's 0.0 no-data fill (44 % of the user's real file) must not enter the histogram."""
        import water_mapping as wm
        rng = np.random.default_rng(0)
        db = rng.normal(-8, 2, (200, 200)).astype(np.float32)
        truth = np.zeros((200, 200), bool); truth[50:120, 20:90] = True
        db[truth] = rng.normal(-21, 1.5, truth.sum())
        clean_mask, clean = wm.map_water(db, "otsu")
        db2 = db.copy(); db2[:, 110:] = 0.0                      # 45 % of the scene is fill
        mask2, info2 = wm.map_water(db2, "otsu")
        self.assertAlmostEqual(info2["threshold_db"], clean["threshold_db"], delta=1.5)
        self.assertEqual(info2["valid_pixels"], int((db2 != 0.0).sum()))
        self.assertGreater(wm.validate_water(mask2, truth, db2 != 0.0)["iou"], 0.9)


class TestWaterMinority(unittest.TestCase):
    """Reproduces the failure seen on the real Sentinel-1 scene: global Otsu splits bright land, not water."""

    def scene(self):
        import water_mapping as wm
        rng = np.random.default_rng(1)
        H = W = 1024
        db = rng.normal(-7.5, 2.8, (H, W)).astype(np.float32)
        truth = np.zeros((H, W), bool)
        truth[300:360, :] = True; truth[640:800, 100:350] = True          # river + lake (about 10 %)
        urb = np.zeros((H, W), bool)
        for _ in range(60):
            r, c = rng.integers(0, H - 100), rng.integers(0, W - 100)
            if truth[max(r - 10, 0):r + 110, max(c - 10, 0):c + 110].any():
                continue
            urb[r:r + 100, c:c + 100] = True
        db[urb] = rng.normal(1.5, 3.5, urb.sum())
        db[truth] = rng.normal(-18, 2.2, truth.sum())
        db[:, 960:] = 0.0                                                   # SNAP no-data fill
        return db, truth, wm

    def test_global_otsu_is_flagged_and_tile_method_works(self):
        db, truth, wm = self.scene()
        valid = db != 0.0
        _, g = wm.map_water(db, "otsu")
        self.assertGreater(g["threshold_db"], -10.0)
        self.assertIn("warning", g)                                         # the implausible result is not silent
        mask, t = wm.map_water(db, "tiles", tile=128)
        self.assertTrue(-17.0 < t["threshold_db"] < -9.0)
        self.assertGreater(wm.validate_water(mask, truth, valid)["iou"], 0.9)
        self.assertNotIn("warning", t)

    def test_tiles_refuses_scene_without_water(self):
        import water_mapping as wm
        db = np.random.default_rng(0).normal(-7.5, 2.8, (512, 512)).astype(np.float32)
        with self.assertRaises(ValueError):
            wm.map_water(db, "tiles", tile=128)

    def test_fixed_threshold_and_sensitivity(self):
        db, truth, wm = self.scene()
        mask, info = wm.map_water(db, "fixed", threshold=-13.0)
        self.assertEqual(info["method"], "fixed")
        self.assertGreater(wm.validate_water(mask, truth, db != 0.0)["iou"], 0.9)
        sens = wm.threshold_sensitivity(db)
        self.assertEqual([s["threshold_db"] for s in sens], [-14.0, -16.0, -18.0, -20.0])
        self.assertTrue(all(a["water_percent"] >= b["water_percent"] for a, b in zip(sens, sens[1:])))
        with self.assertRaises(ValueError):
            wm.map_water(db, "fixed")


class TestFloodFromDb(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(3)
        H = W = 600
        before = rng.normal(-8, 2.0, (H, W)).astype(np.float32)
        after = before + rng.normal(0, 0.8, (H, W)).astype(np.float32)             # same scene, small noise
        river = np.zeros((H, W), bool); river[100:140, :] = True
        flood = np.zeros((H, W), bool); flood[300:400, 150:400] = True              # new water
        shadow = np.zeros((H, W), bool); shadow[450:520, 50:200] = True             # dark on BOTH dates, noise crosses threshold
        before[river] = after[river] = -19
        before[flood] = -8; after[flood] = -18                                       # strong drop
        before[shadow] = -12.3; after[shadow] = -13.4                                # no strong drop
        after[:, 560:] = 0.0                                                         # no-data strip
        return before, after, river, flood, shadow

    def test_flooded_permanent_and_uncertain(self):
        import water_mapping as wm
        before, after, river, flood, shadow = self.scene()
        cls, wb, wa, st = wm.flood_from_db(before, after, threshold=-13.0, min_drop_db=3.0, pixel_area_m2=100.0)
        valid = after != 0.0
        self.assertGreater((cls[flood & valid] == 2).mean(), 0.97)                  # flood found
        self.assertGreater((cls[river & valid] == 1).mean(), 0.97)                  # river stays permanent, is not flood
        self.assertTrue((cls[:, 560:] == 255).all())                                # fill excluded
        self.assertEqual(int(((cls == 2) & river).sum()), 0)
        # dark-on-both-dates area: never counted as flood, shown as uncertain/permanent at most
        self.assertLess(((cls == 2) & shadow).mean() if shadow.any() else 0, 0.02)
        self.assertAlmostEqual(st["area_ha"][2], st["counts"][2] * 100 / 1e4)

    def test_without_drop_test_shadow_noise_becomes_flood(self):
        import water_mapping as wm
        before, after, river, flood, shadow = self.scene()
        cls_on, *_ = wm.flood_from_db(before, after, -13.0, min_drop_db=3.0)
        cls_off, *_ = wm.flood_from_db(before, after, -13.0, min_drop_db=0.0)
        self.assertGreater(int((cls_off == 2).sum()), int((cls_on == 2).sum()))     # the drop test removes false alarms

    def test_common_threshold_between_modes(self):
        import water_mapping as wm
        before, after, *_ = self.scene()
        thr, info = wm.common_tile_threshold([before, after], tile=100)
        self.assertTrue(-19.0 < thr < -9.0)
        self.assertEqual(len(info), 2)

    def test_shape_mismatch(self):
        import water_mapping as wm
        with self.assertRaises(ValueError):
            wm.flood_from_db(np.zeros((4, 4)), np.zeros((5, 4)), -13.0)


class TestFloodModule(unittest.TestCase):
    def test_report_and_area_helpers(self):
        import flood, water_mapping as wm
        from types import SimpleNamespace
        before = np.full((50, 50), -8.0, np.float32); after = before.copy()
        after[10:30, 10:30] = -19.0
        cls, wb, wa, st = wm.flood_from_db(before, after, -14.0, 3.0, 10, 100.0)
        txt = flood.report_text(st, 100.0, False)
        self.assertIn("Flooded", txt); self.assertIn("Not validated", txt)
        tr = SimpleNamespace(a=8.983e-5, e=-8.983e-5)
        a = flood.pixel_area_m2(tr, True, 20.9)
        self.assertTrue(85 < a < 100)                       # about 10 m x 9.3 m at 21 N
        self.assertAlmostEqual(flood.pixel_area_m2(SimpleNamespace(a=10, e=-10), False, 0), 100.0)
        self.assertEqual(set(flood.CLASS_COLOURS), {0, 1, 2, 3, 4})
