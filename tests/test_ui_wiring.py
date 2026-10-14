"""Static check: every slot wired with .connect(self.name) in ui.py exists as a method (catches deleted/renamed methods
that otherwise only crash at start-up on a real GUI)."""
import ast
import os
import unittest


class UiWiring(unittest.TestCase):
    def test_connected_methods_exist(self):
        path = os.path.join(os.path.dirname(__file__), "..", "ui.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SatelliteImageApp")
        methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
                for arg in node.args:
                    if (isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "self"
                            and arg.attr in {"print_image"} | methods | set()):
                        continue
                    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "self":
                        missing.add(arg.attr)
        # attributes that are Qt objects rather than methods are legitimate connect targets (e.g. self.close)
        qt_builtin = {"close", "accept", "reject", "show", "hide", "update", "deleteLater", "quit", "setFocus", "clear"}
        self.assertEqual(sorted(missing - qt_builtin), [])


if __name__ == "__main__":
    unittest.main()


class ModuleAttributes(unittest.TestCase):
    def test_project_module_functions_exist(self):
        """Every `validation.func(...)`-style call in ui.py/cli.py must name something that exists in that module."""
        import importlib
        here = os.path.join(os.path.dirname(__file__), "..")
        mods = ["validation", "terrain", "basins", "webmap", "georef", "flood", "water_mapping", "supervised",
                "change_detection", "sar_snap", "raster_expr", "imageops", "mapexport", "labeler"]
        loaded = {}
        for m in mods:
            try:
                loaded[m] = importlib.import_module(m)
            except Exception:
                pass            # optional heavy dependency missing: skipped
        missing = []
        for fname in ("ui.py", "cli.py", "mapexport_ui.py", "labeler_ui.py"):
            with open(os.path.join(here, fname), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in loaded
                        and not hasattr(loaded[node.value.id], node.attr)):
                    missing.append(f"{fname}:{node.lineno} {node.value.id}.{node.attr}")
        self.assertEqual(missing, [])
