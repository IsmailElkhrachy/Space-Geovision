"""Safe raster-calculator expression evaluator (no Qt, no rasterio).

Layer names are written in double quotes, e.g. ``("NIR" - "Red") / ("NIR" + "Red")``. The expression is parsed
with ``ast`` and only a whitelist of syntax is evaluated: arithmetic, comparisons, ``& | ^ ~`` for masks, and a few
numpy functions. Attribute access, subscripts, imports, lambdas and unknown names are rejected, so a formula can
never run arbitrary code. Division by zero gives NaN (no epsilon bias) and infinities become NaN.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, Mapping

import numpy as np

FUNCTIONS = {
    "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arctan": np.arctan, "abs": np.abs, "where": np.where, "minimum": np.minimum, "maximum": np.maximum,
    "clip": np.clip, "floor": np.floor, "ceil": np.ceil, "isnan": np.isnan, "nan_to_num": np.nan_to_num,
}
_BIN = {ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor}
_CMP = {ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq}
_UNARY = {ast.USub, ast.UAdd, ast.Invert, ast.Not}
MAX_FORMULA_CHARS = 2000


class FormulaError(ValueError):
    pass


def _div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b == 0, np.nan, a / b)


def _floordiv(a, b):
    return np.floor(_div(a, b))


class _Check(ast.NodeTransformer):
    def __init__(self, names):
        self.names = set(names)

    def visit_Expression(self, node):
        node.body = self.visit(node.body)
        return node

    def visit_BinOp(self, node):
        if type(node.op) is ast.Div:
            return ast.copy_location(ast.Call(ast.Name("_div", ast.Load()), [self.visit(node.left), self.visit(node.right)], []), node)
        if type(node.op) is ast.FloorDiv:
            return ast.copy_location(ast.Call(ast.Name("_floordiv", ast.Load()), [self.visit(node.left), self.visit(node.right)], []), node)
        if type(node.op) not in _BIN:
            raise FormulaError(f"Operator {type(node.op).__name__} is not allowed")
        node.left, node.right = self.visit(node.left), self.visit(node.right)
        return node

    def visit_UnaryOp(self, node):
        if type(node.op) not in _UNARY:
            raise FormulaError("Unary operator not allowed")
        node.operand = self.visit(node.operand)
        return node

    def visit_Compare(self, node):
        if any(type(o) not in _CMP for o in node.ops):
            raise FormulaError("Comparison operator not allowed")
        if len(node.ops) > 1:
            raise FormulaError("Chained comparisons are not supported; combine with & and parentheses")
        node.left = self.visit(node.left)
        node.comparators = [self.visit(c) for c in node.comparators]
        return node

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise FormulaError(f"Unknown function. Allowed: {', '.join(sorted(FUNCTIONS))}")
        if node.keywords:
            raise FormulaError("Keyword arguments are not allowed")
        node.args = [self.visit(a) for a in node.args]
        return node

    def visit_Name(self, node):
        if node.id in self.names or node.id in ("_div", "_floordiv"):
            return node
        raise FormulaError(f"Unknown name '{node.id}'. Put layer names in double quotes.")

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, bool)):
            return node
        raise FormulaError("Only numeric constants are allowed")

    def generic_visit(self, node):
        raise FormulaError(f"Syntax '{type(node).__name__}' is not allowed in a formula")


def layer_names(formula: str):
    """Distinct quoted layer names in order of first use."""
    seen = []
    for n in re.findall(r'"([^"]+)"', formula):
        if n not in seen:
            seen.append(n)
    return seen


def evaluate(formula: str, layers: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate ``formula`` with ``layers`` {name: array}. All layers used must have the same shape.
    Returns float32 with NaN where the result is undefined (division by zero, infinities)."""
    formula = (formula or "").strip()
    if not formula:
        raise FormulaError("The formula is empty")
    if len(formula) > MAX_FORMULA_CHARS:
        raise FormulaError("The formula is too long")
    names = layer_names(formula)
    if not names:
        raise FormulaError('The formula uses no layers. Click a layer to insert it as "name".')
    missing = [n for n in names if n not in layers]
    if missing:
        raise FormulaError("Missing layers: " + ", ".join(missing))
    env: Dict[str, object] = {}
    expr = formula
    shapes = set()
    for i, n in enumerate(names):
        var = f"L{i}"
        expr = expr.replace(f'"{n}"', var)
        arr = np.asarray(layers[n])
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        env[var] = arr.astype(np.float64)
        shapes.add(arr.shape)
    if len(shapes) > 1:
        raise FormulaError("Layers have different dimensions: " + ", ".join(str(s) for s in sorted(shapes)))
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Invalid syntax: {exc.msg}") from None
    tree = ast.fix_missing_locations(_Check(env).visit(tree))
    scope = dict(FUNCTIONS); scope.update(env); scope["_div"] = _div; scope["_floordiv"] = _floordiv
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, scope)   # whitelisted AST only
    out = np.asarray(out, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)
