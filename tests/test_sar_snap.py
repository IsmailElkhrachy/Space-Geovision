"""Tests for sar_snap.py. Run with:  python -m unittest discover -s tests -v   (or pytest).
A fake `gpt` script stands in for SNAP, so these tests verify our logic (graph, command line,
progress parsing, error and cancel handling) - NOT SNAP's own numerical results."""
import os
import stat
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sar_snap as s  # noqa: E402

GOOD = "S1A_IW_GRDH_1SDV_20210911T025500_20210911T025525_039500_04AAAA_1234.zip"


def make_fake_gpt(dirpath, body):
    """A stand-in for SNAP's gpt. On Windows a .cmd wrapper starts the Python script (a shebang is not enough)."""
    impl = Path(dirpath) / "fake_gpt_impl.py"
    impl.write_text(textwrap.dedent(body))
    if os.name == "nt":
        p = Path(dirpath) / "gpt.cmd"
        p.write_text('@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, impl))
    else:
        p = Path(dirpath) / "gpt"
        p.write_text("#!%s\n" % sys.executable + impl.read_text())
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


class TestProductName(unittest.TestCase):
    def test_valid_dual_pol(self):
        info = s.parse_product_name("/x/" + GOOD)
        self.assertEqual((info.mission, info.mode, info.product_type), ("S1A", "IW", "GRD"))
        self.assertEqual(info.polarisations, ["VV", "VH"])
        self.assertEqual(info.start, "20210911T025500")

    def test_geotiff_export_rejected(self):
        with self.assertRaises(s.SnapError):
            s.parse_product_name("2021-09-11-00_00_2021-09-11-23_59_Sentinel-1_IW.tiff")

    def test_slc_rejected(self):
        with self.assertRaises(s.SnapError):
            s.parse_product_name(GOOD.replace("GRDH", "SLC_").replace("SDV", "SDV"))

    def test_copernicus_safe_dot_zip_name(self):
        n = "S1A_IW_GRDH_1SDV_20240903T110609_20240903T110634_055500_06C589_5A6B.SAFE.zip"
        info = s.parse_product_name("D:\\Vitname\\" + n)
        self.assertEqual((info.start, info.polarisations), ("20240903T110609", ["VV", "VH"]))
        self.assertEqual(s._strip_product_suffix(n), n[:-len(".SAFE.zip")])

    def test_manifest_safe_uses_parent_folder(self):
        info = s.parse_product_name("/x/" + GOOD.replace(".zip", ".SAFE") + "/manifest.safe")
        self.assertTrue(info.is_grd)


class TestParams(unittest.TestCase):
    def test_pol_not_in_product(self):
        info = s.parse_product_name(GOOD)
        with self.assertRaises(s.SnapError):
            s.SnapParams(polarisation="HH").validate(info)

    def test_even_window_rejected(self):
        with self.assertRaises(s.SnapError):
            s.SnapParams(speckle_window=4).validate()

    def test_external_dem_must_exist(self):
        with self.assertRaises(s.SnapError):
            s.SnapParams(dem_name="External DEM", external_dem="/nope.tif").validate()

    def test_bad_subset(self):
        with self.assertRaises(s.SnapError):
            s.SnapParams(subset_wkt="BOX(1 2 3 4)").validate()


class TestGraph(unittest.TestCase):
    def order(self, xml):
        root = ET.fromstring(xml)
        return [n.find("operator").text for n in root.findall("node")]

    def test_full_order_is_physically_correct(self):
        ops = self.order(s.build_graph_xml("in.zip", "out.tif", s.SnapParams(subset_wkt="POLYGON((0 0,1 0,1 1,0 0))")))
        self.assertEqual(ops, ["Read", "Subset", "Apply-Orbit-File", "ThermalNoiseRemoval", "Calibration",
                               "Speckle-Filter", "Terrain-Correction", "LinearToFromdB", "Write"])

    def test_optional_steps_skipped(self):
        ops = self.order(s.build_graph_xml("in.zip", "o.tif", s.SnapParams(
            speckle_filter="None", remove_thermal_noise=False, output_db=False)))
        self.assertEqual(ops, ["Read", "Apply-Orbit-File", "Calibration", "Terrain-Correction", "Write"])

    def test_every_source_refers_to_existing_node(self):
        root = ET.fromstring(s.build_graph_xml("in.zip", "o.tif", s.SnapParams()))
        ids = {n.get("id") for n in root.findall("node")}
        for ref in root.iter("sourceProduct"):
            self.assertIn(ref.get("refid"), ids)

    def test_sigma0_and_sea_flag(self):
        x = s.build_graph_xml("in.zip", "o.tif", s.SnapParams(polarisation="VH"))
        self.assertIn("<outputSigmaBand>true</outputSigmaBand>", x)
        self.assertIn("<sourceBands>Sigma0_VH</sourceBands>", x)
        self.assertIn("<nodataValueAtSea>false</nodataValueAtSea>", x)
        self.assertIn("<outputImageScaleInDb>false</outputImageScaleInDb>", x)  # dB only after correction


class TestSubsetModes(unittest.TestCase):
    WKT = "POLYGON((106.3 20.7, 106.7 20.7, 106.7 21.1, 106.3 21.1, 106.3 20.7))"

    def ops(self, **kw):
        x = s.build_graph_xml("in.zip", "o.tif", s.SnapParams(subset_wkt=self.WKT, **kw))
        return [n.find("operator").text for n in ET.fromstring(x).findall("node")], x

    def test_operator_mode(self):
        ops, _ = self.ops(subset_mode="operator")
        self.assertEqual(ops[:3], ["Read", "Subset", "Apply-Orbit-File"])

    def test_read_mode_uses_read_geometry_region(self):
        ops, x = self.ops(subset_mode="read")
        self.assertNotIn("Subset", ops)
        self.assertIn("<geometryRegion>", x); self.assertIn("<useAdvancedOptions>true</useAdvancedOptions>", x)

    def test_after_mode_crops_after_terrain_correction(self):
        ops, _ = self.ops(subset_mode="after")
        self.assertEqual(ops[ops.index("Terrain-Correction") + 1], "Subset")
        self.assertEqual(ops[-2:], ["LinearToFromdB", "Write"])

    def test_default_projection_is_snaps_own_default(self):
        _, x = self.ops()
        self.assertIn("<mapProjection>WGS84(DD)</mapProjection>", x)
        self.assertNotIn("<mapProjection>", s.build_graph_xml("i.zip", "o.tif", s.SnapParams(map_projection="")))

    def test_bad_mode(self):
        with self.assertRaises(s.SnapError):
            s.SnapParams(subset_mode="x").validate()


class TestProbe(unittest.TestCase):
    def test_fast_failure_and_init_ok(self):
        with tempfile.TemporaryDirectory() as d:
            gpt = make_fake_gpt(d, """
                import sys, time
                g = open(sys.argv[1]).read()
                if "EPSG:4326" in g:
                    print("java.lang.ArithmeticException: / by zero")
                    print("        at javax.media.jai.PlanarImage.XToTileX(Unknown Source)")
                    sys.exit(1)
                print("Executing processing graph", flush=True)
                time.sleep(30)
            """)
            bad, good = Path(d) / "bad.xml", Path(d) / "good.xml"
            bad.write_text("<mapProjection>EPSG:4326</mapProjection>"); good.write_text("<mapProjection>WGS84(DD)</mapProjection>")
            r1 = s.probe_graph_init(gpt, str(bad), "bad", wait_s=5)
            self.assertEqual(r1.status, "failed"); self.assertIn("by zero", r1.detail); self.assertNotIn("XToTileX", r1.detail)
            r2 = s.probe_graph_init(gpt, str(good), "good", wait_s=1.5)
            self.assertEqual(r2.status, "init-ok"); self.assertLess(r2.seconds, 10)


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prod = Path(self.tmp.name) / GOOD
        self.prod.write_bytes(b"x")

    def test_success_and_progress(self):
        gpt = make_fake_gpt(self.tmp.name, """
            import sys, re
            g = open(sys.argv[1]).read()
            out = re.search(r"<file>([^<]*\\.tif)</file>", g[g.rfind("<operator>Write"):]).group(1)
            for p in (10, 40, 100):
                print("....%d%%" % p, flush=True)
            open(out, "wb").write(b"II*\\x00")
        """)
        seen = []
        res = s.run_full_chain(str(self.prod), self.tmp.name, s.SnapParams(), gpt=gpt,
                               progress=lambda v, m: seen.append(v))
        self.assertTrue(Path(res.geotiff).is_file())
        self.assertEqual(res.band_name, "Sigma0_VV_db")
        self.assertEqual(seen[:3], [10, 40, 100])
        self.assertTrue(Path(self.tmp.name, res.geotiff).exists())

    def test_gpt_failure_reports_tail(self):
        gpt = make_fake_gpt(self.tmp.name, """
            import sys
            print("Error: [NodeId: Apply-Orbit-File] orbit file not available")
            sys.exit(1)
        """)
        with self.assertRaises(s.SnapError) as cm:
            s.run_full_chain(str(self.prod), self.tmp.name, s.SnapParams(), gpt=gpt)
        self.assertIn("orbit file not available", str(cm.exception))

    def test_java_stack_trace_shows_real_message_and_saves_log(self):
        gpt = make_fake_gpt(self.tmp.name, """
            import sys
            print("INFO: starting")
            print("Error: [NodeId: Apply-Orbit-File] Cannot download orbit file: connection timed out")
            print("org.esa.snap.core.gpf.OperatorException: Cannot download orbit file")
            for i in range(30):
                print("        at java.base/jdk.internal.reflect.Frame%d.invoke(Native Method)" % i)
            sys.exit(1)
        """)
        with self.assertRaises(s.SnapError) as cm:
            s.run_full_chain(str(self.prod), self.tmp.name, s.SnapParams(), gpt=gpt)
        msg = str(cm.exception)
        self.assertIn("Cannot download orbit file", msg)
        self.assertNotIn("Frame29", msg)
        log = next(Path(self.tmp.name).glob("*_gpt.log"))
        self.assertIn("Frame29", log.read_text())

    def test_missing_output_is_detected(self):
        gpt = make_fake_gpt(self.tmp.name, "print('done')")
        with self.assertRaises(s.SnapError):
            s.run_full_chain(str(self.prod), self.tmp.name, s.SnapParams(), gpt=gpt)

    def test_cancel(self):
        gpt = make_fake_gpt(self.tmp.name, """
            import time
            for i in range(200):
                print("....%d%%" % (i % 100), flush=True)
                time.sleep(0.05)
        """)
        with self.assertRaises(s.Cancelled):
            s.run_full_chain(str(self.prod), self.tmp.name, s.SnapParams(), gpt=gpt, cancelled=lambda: True)

    @unittest.skipIf(os.name == "nt", "PATH manipulation is not meaningful on Windows")
    def test_gpt_not_found(self):
        old = os.environ.get("PATH", "")
        os.environ["PATH"] = ""
        try:
            os.environ.pop("SNAP_GPT", None)
            if not any(p.is_file() for p in s._candidate_gpt_paths()):
                with self.assertRaises(s.SnapError):
                    s.find_gpt()
        finally:
            os.environ["PATH"] = old

    def test_verify_parameter_names(self):
        gpt = make_fake_gpt(self.tmp.name, """
            import sys
            if sys.argv[1] == "Terrain-Correction":
                print("  -PdemName=  DEM\\n  -PmapProjection=  proj")
            else:
                print("  -ParamA=x")
        """)
        warns = s.verify_parameter_names(gpt, s.SnapParams())
        self.assertTrue(any("Terrain-Correction" in w and "pixelSpacingInMeter" in w for w in warns))


FAKE_STEP_GPT = """
    import sys, os
    a = sys.argv[1:]
    op = a[0]
    t = a[a.index("-t") + 1]
    src = [x for x in a if x.startswith("-Ssource=")][0].split("=", 1)[1]
    log = os.environ["FAKE_LOG"]
    open(log, "a").write(op + "|" + src + "|" + t + "|" + a[a.index("-f") + 1] + "\\n")
    if os.environ.get("FAIL_AT") == op:
        print("Error: / by zero"); sys.exit(1)
    print("....50%....100%")
    os.makedirs(os.path.dirname(t), exist_ok=True)
    open(t, "wb").write(b"x")
"""


class TestStepwise(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.prod = Path(self.tmp.name) / GOOD; self.prod.write_bytes(b"x")
        self.gpt = make_fake_gpt(self.tmp.name, FAKE_STEP_GPT)
        self.log = str(Path(self.tmp.name) / "calls.txt")
        os.environ["FAKE_LOG"] = self.log
        self.addCleanup(lambda: (os.environ.pop("FAKE_LOG", None), os.environ.pop("FAIL_AT", None)))

    def calls(self):
        return [l.split("|") for l in Path(self.log).read_text().split()]

    def test_chain_order_and_data_flow(self):
        res = s.run_stepwise(str(self.prod), self.tmp.name, s.SnapParams(), gpt=self.gpt)
        ops = [c[0] for c in self.calls()]
        self.assertEqual(ops, ["Apply-Orbit-File", "ThermalNoiseRemoval", "Calibration", "Speckle-Filter",
                               "Terrain-Correction", "LinearToFromdB"])
        calls = self.calls()
        self.assertEqual(calls[0][1], str(self.prod))
        for prev, nxt in zip(calls, calls[1:]):
            self.assertEqual(nxt[1], prev[2])               # every step reads the previous step's output
        self.assertEqual(calls[-1][3], "GeoTIFF"); self.assertTrue(Path(res.geotiff).is_file())

    def test_subset_at_end_goes_after_terrain_correction(self):
        wkt = "POLYGON((106.3 20.7, 106.7 20.7, 106.7 21.1, 106.3 21.1, 106.3 20.7))"
        import zipfile
        z = Path(self.tmp.name) / "z" / GOOD; z.parent.mkdir()
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("x.SAFE/manifest.safe", "<gml:coordinates>20.0,106.0 20.4,108.0 22.4,107.6 22.0,105.6 20.0,106.0</gml:coordinates>")
        s.run_stepwise(str(z), self.tmp.name, s.SnapParams(subset_wkt=wkt, subset_stage="end"), gpt=self.gpt)
        ops = [c[0] for c in self.calls()]
        self.assertEqual(ops.index("Subset"), ops.index("Terrain-Correction") + 1)

    def test_failure_names_the_step_and_stops(self):
        os.environ["FAIL_AT"] = "Terrain-Correction"
        with self.assertRaises(s.SnapError) as cm:
            s.run_stepwise(str(self.prod), self.tmp.name, s.SnapParams(), gpt=self.gpt)
        self.assertIn("FAILED: Terrain-Correction", str(cm.exception))
        self.assertIn("by zero", str(cm.exception))
        self.assertEqual([c[0] for c in self.calls()][-1], "Terrain-Correction")   # nothing ran after it


class TestSanity(unittest.TestCase):
    def test_flags_dn_in_db(self):
        # 10*log10(raw uint16 DN): the mistake seen in the old workflow (24.8 - 48.2 dB)
        dn = np.random.default_rng(0).integers(300, 65000, 10000).astype("float32")
        self.assertTrue(s.sanity_check_db(10 * np.log10(dn)))

    def test_accepts_real_sigma0(self):
        v = np.random.default_rng(1).normal(-14, 5, 10000).astype("float32")
        self.assertEqual(s.sanity_check_db(v), [])

    def test_empty(self):
        self.assertTrue(s.sanity_check_db(np.array([np.nan])))


class TestProvenance(unittest.TestCase):
    def test_sidecar_written(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            gpt = make_fake_gpt(d, """
                import sys, re
                g = open(sys.argv[1]).read()
                out = re.search(r"<file>([^<]*\\.tif)</file>", g[g.rfind("<operator>Write"):]).group(1)
                open(out, "wb").write(b"II*")
            """)
            prod = Path(d) / GOOD
            prod.write_bytes(b"x")
            res = s.run_full_chain(str(prod), d, s.SnapParams(speckle_window=7), gpt=gpt)
            side = s.write_provenance(res, gpt, str(prod))
            doc = json.loads(Path(side).read_text())
            self.assertEqual(doc["parameters"]["speckle_window"], 7)
            self.assertIn("Terrain-Correction", doc["graph_xml"])
            self.assertEqual(doc["input_product"], GOOD)


class TestSafeFolder(unittest.TestCase):
    def test_unzipped_safe_folder_resolves_to_manifest(self):
        name = "S1A_IW_GRDH_1SDV_20240829T105808_20240829T105833_055427_06C2BB_954F.SAFE"
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / name
            folder.mkdir()
            (folder / "manifest.safe").write_text('<s1:pass>ASCENDING</s1:pass><gml:coordinates>20.0,104.0 20.2,107.0 22.2,106.6 22.0,103.6 20.0,104.0</gml:coordinates>')
            info = s.parse_product_name(str(folder))
            self.assertTrue(info.path.endswith("manifest.safe"))      # this is what goes into the SNAP Read operator
            self.assertEqual((info.abs_orbit, info.relative_orbit), (55427, 55))
            self.assertEqual(s.read_pass_direction(str(folder)), "ASCENDING")
            self.assertEqual(s.footprint_bbox(s.read_footprint(str(folder)))[0], 103.6)
            self.assertIn("manifest.safe", s.build_graph_xml(info.path, "o.tif", s.SnapParams()))


class TestBoxWkt(unittest.TestCase):
    def test_box_matches_cli_example_and_is_inside_footprint(self):
        wkt = s.box_wkt(106.5, 20.9, 40)
        self.assertEqual(wkt, "POLYGON((106.3071 20.7198, 106.6929 20.7198, 106.6929 21.0802, 106.3071 21.0802, 106.3071 20.7198))")
        with self.assertRaises(s.SnapError):
            s.box_wkt(106.5, 20.9, 0)


class TestDemCoverage(unittest.TestCase):
    def test_high_latitude_needs_another_dem(self):
        far_north = s.box_wkt(15.0, 68.0, 40)
        with self.assertRaises(s.SnapError):
            s.check_dem_coverage(s.SnapParams(subset_wkt=far_north), "x")
        s.check_dem_coverage(s.SnapParams(subset_wkt=far_north, dem_name="Copernicus 30m Global DEM"), "x")   # allowed
        s.check_dem_coverage(s.SnapParams(subset_wkt=s.box_wkt(106.5, 20.9, 40)), "x")                       # Vietnam: fine
        with self.assertRaises(s.SnapError):
            s.check_dem_coverage(s.SnapParams(subset_wkt=s.box_wkt(-70.0, -60.0, 40)), "x")


class TestSubsetSpec(unittest.TestCase):
    def make_zip(self, d):
        import zipfile
        z = Path(d) / GOOD
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("x.SAFE/manifest.safe", "<gml:coordinates>20.0,104.0 20.4,107.0 22.4,106.6 22.0,103.6 20.0,104.0</gml:coordinates>")
        return str(z)

    def test_modes_produce_wkt(self):
        self.assertIsNone(s.resolve_subset(s.SubsetSpec("full")))
        w = s.resolve_subset(s.SubsetSpec("bbox", bbox=(106.3, 20.7, 106.7, 21.1)))
        self.assertEqual(w, "POLYGON((106.3000 20.7000, 106.7000 20.7000, 106.7000 21.1000, 106.3000 21.1000, 106.3000 20.7000))")
        c = s.resolve_subset(s.SubsetSpec("center", center=(106.5, 20.9), radius_km=20))
        self.assertEqual(c, s.box_wkt(106.5, 20.9, 40))
        p = s.resolve_subset(s.SubsetSpec("polygon", wkt="POLYGON((1 2, 3 2, 3 4, 1 4, 1 2))"))
        self.assertTrue(p.startswith("POLYGON"))

    def test_works_anywhere_on_earth(self):
        for lon, lat in [(44.2, 17.5), (-60.0, -3.1), (151.2, -33.9), (15.0, 50.0)]:     # Najran, Amazon, Sydney, Europe
            w = s.resolve_subset(s.SubsetSpec("center", center=(lon, lat), radius_km=15))
            x0, y0, x1, y1 = s._wkt_bbox(w)
            self.assertTrue(x0 < lon < x1 and y0 < lat < y1)

    def test_invalid_input_is_refused(self):
        bad = [s.SubsetSpec("bbox", bbox=(106.7, 20.7, 106.3, 21.1)),            # min > max
               s.SubsetSpec("bbox", bbox=(20.7, 106.3, 21.1, 106.7)),            # lat/lon swapped -> lat > 90
               s.SubsetSpec("bbox", bbox=(-100, 0, 100, 10)),                    # crosses the antimeridian rule (width > 180)
               s.SubsetSpec("center", center=(106.5, 20.9), radius_km=0),
               s.SubsetSpec("center", center=(106.5, 20.9)),
               s.SubsetSpec("polygon", wkt="LINESTRING(1 2, 3 4)"),
               s.SubsetSpec("polygon", wkt="POLYGON((1 2, 3 4))"),
               s.SubsetSpec("nonsense")]
        for sp in bad:
            with self.assertRaises(s.SnapError, msg=str(sp)):
                s.resolve_subset(sp)

    def test_box_covering_the_scene_means_no_cut(self):
        with tempfile.TemporaryDirectory() as d:
            z = self.make_zip(d)
            self.assertIsNone(s.resolve_subset(s.SubsetSpec("bbox", bbox=(100, 10, 120, 30)), z))
            self.assertIsNotNone(s.resolve_subset(s.SubsetSpec("bbox", bbox=(105, 20.5, 106, 21.5)), z))

    def test_pixel_estimate_and_advice(self):
        with tempfile.TemporaryDirectory() as d:
            z = self.make_zip(d)
            px = s.estimate_output_pixels(s.box_wkt(106.0, 21.0, 40), z, 10.0)
            self.assertTrue(12e6 < px < 20e6)                                    # 40 km x 40 km at 10 m = 16 MP
            self.assertIn("light", s.size_advice(px))
            self.assertIn("VERY large", s.size_advice(s.estimate_output_pixels(None, z, 10.0)))
            self.assertEqual(s.estimate_output_pixels(s.box_wkt(10.0, 50.0, 10), z, 10.0), 0)   # outside the scene


class TestOrbits(unittest.TestCase):
    def test_relative_orbits_of_the_vietnam_scenes(self):
        names = {"S1A_IW_GRDH_1SDV_20240903T110609_20240903T110634_055500_06C589_5A6B.SAFE.zip": 128,
                 "S1A_IW_GRDH_1SDV_20240910T105808_20240910T105833_055602_06C990_A1D7.SAFE.zip": 55,
                 "S1A_IW_GRDH_1SDV_20240910T105833_20240910T105858_055602_06C990_104D.SAFE.zip": 55,
                 "S1A_IW_GRDH_1SDV_20240910T230726_20240910T230751_055609_06C9DF_1FDD.SAFE.zip": 62}
        for n, rel in names.items():
            self.assertEqual(s.parse_product_name(n).relative_orbit, rel, n)

    def test_consecutive_slices_share_datatake(self):
        a = s.parse_product_name("S1A_IW_GRDH_1SDV_20240910T105808_20240910T105833_055602_06C990_A1D7.SAFE.zip")
        b = s.parse_product_name("S1A_IW_GRDH_1SDV_20240910T105833_20240910T105858_055602_06C990_104D.SAFE.zip")
        self.assertEqual((a.abs_orbit, a.datatake), (b.abs_orbit, b.datatake))

    def test_pass_direction(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / GOOD
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("x.SAFE/manifest.safe", "<s1:pass>DESCENDING</s1:pass>")
            self.assertEqual(s.read_pass_direction(str(z)), "DESCENDING")


class TestPointInFootprint(unittest.TestCase):
    def test_slanted_polygon(self):
        poly = [(106.09, 21.07), (108.57, 21.7), (107.83, 23.01), (105.83, 22.4), (106.09, 21.07)]
        self.assertTrue(s.footprint_contains(poly, 107.0, 22.0))
        self.assertFalse(s.footprint_contains(poly, 105.85, 21.03))   # west/south of the swath
        self.assertFalse(s.footprint_contains(poly, 107.0, 24.0))


class TestSubsetCoverage(unittest.TestCase):
    def test_partial_and_full(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / GOOD
            with zipfile.ZipFile(z, "w") as zf:      # slanted square-ish swath
                zf.writestr("x.SAFE/manifest.safe", "<gml:coordinates>20.0,104.0 20.4,107.0 22.4,106.6 22.0,103.6 20.0,104.0</gml:coordinates>")
            full = "POLYGON((105 21, 106 21, 106 21.5, 105 21.5, 105 21))"
            part = "POLYGON((103.3 21, 104.3 21, 104.3 21.5, 103.3 21.5, 103.3 21))"
            self.assertEqual(s.check_subset_in_footprint(full, str(z)), 1.0)
            f = s.check_subset_in_footprint(part, str(z))
            self.assertTrue(0 < f < 1)


class TestFootprint(unittest.TestCase):
    MANIFEST = ('<x><safe:footPrint srsName="epsg4326"><gml:coordinates>'
                '20.0,104.0 20.2,107.0 22.2,106.6 22.0,103.6 20.0,104.0</gml:coordinates></safe:footPrint></x>')

    def make_zip(self, d):
        import zipfile
        z = Path(d) / GOOD
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr(GOOD.replace(".zip", "") + ".SAFE/manifest.safe", self.MANIFEST)
        return str(z)

    def test_read_footprint_lon_lat_order(self):
        with tempfile.TemporaryDirectory() as d:
            pts = s.read_footprint(self.make_zip(d))
            self.assertEqual(pts[0], (104.0, 20.0))          # (lon, lat), not (lat, lon)
            self.assertEqual(s.footprint_bbox(pts), (103.6, 20.0, 107.0, 22.2))

    def test_subset_inside_ok_outside_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            z = self.make_zip(d)
            s.check_subset_in_footprint("POLYGON((105.5 20.8, 106.0 20.8, 106.0 21.2, 105.5 21.2, 105.5 20.8))", z)
            with self.assertRaises(s.SnapError):             # Najran is nowhere near this scene
                s.check_subset_in_footprint("POLYGON((44.0 17.4, 44.3 17.4, 44.3 17.6, 44.0 17.6, 44.0 17.4))", z)
            with self.assertRaises(s.SnapError):             # swapped lat/lon
                s.check_subset_in_footprint("POLYGON((21.0 105.5, 21.2 105.5, 21.2 106.0, 21.0 106.0, 21.0 105.5))", z)

    def test_run_stops_early_on_wrong_area(self):
        with tempfile.TemporaryDirectory() as d:
            z = self.make_zip(d)
            gpt = make_fake_gpt(d, "print('should never run')")
            with self.assertRaises(s.SnapError) as cm:
                s.run_full_chain(z, d, s.SnapParams(subset_wkt="POLYGON((44 17, 45 17, 45 18, 44 18, 44 17))"), gpt=gpt)
            self.assertIn("does not overlap", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
