"""Safe arithmetic for agent calculator tools.

The tools used to call eval() on model-written expressions. eval() with an
empty __builtins__ is not a sandbox: attribute access such as
().__class__.__bases__[0].__subclasses__() reaches os/subprocess, so a
prompt-injected expression could run arbitrary code, and 9**9**9 can hang
the process. This walks the parsed expression and allows only numbers,
lists/tuples of numbers, arithmetic, comparisons and a few named functions.
"""
import ast
import operator

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CMP = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_MAX_EXPONENT = 1000
_MAX_LEN = 500


def safe_eval(expression: str, functions: dict):
    """Evaluate an arithmetic expression; raise ValueError for anything else."""
    if len(expression) > _MAX_LEN:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare) and all(type(op) in _CMP for op in node.ops):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and not node.keywords):
            return functions[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return ev(tree)
