import ast
import operator as op
import re

from app.mcp_server.tools.base import Tool


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


class CalculatorTool(Tool):
    name = "calculator"
    description = "Perform safe mathematical calculations."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A valid mathematical expression to evaluate"
            }
        },
        "required": ["expression"],
    }

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return _ALLOWED_OPERATORS[type(node.op)](
                self._eval(node.left),
                self._eval(node.right),
            )
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_OPERATORS:
                raise ValueError("Unary operator not allowed")
            return _ALLOWED_OPERATORS[type(node.op)](
                self._eval(node.operand)
            )
        raise ValueError("Unsupported expression")

    async def run(self, args: dict) -> str:
        """
        Executes the calculator tool.

        IMPORTANT RULE:
        - This method MUST NEVER raise an exception.
        - On any error, it must return a string.
        """

        try:
            expr = args.get("expression", "")
            if not isinstance(expr, str):
                return "Calculator not used: invalid expression type."

            expr = expr.replace("^", "**").strip()

            if not re.search(r"\d", expr):
                return "Calculator not used: no numeric expression detected."

            clean_expr = re.sub(r"[^0-9+\-*/(). **]", "", expr)

            if not clean_expr:
                return "Calculator not used: expression is empty after sanitization."

            tree = ast.parse(clean_expr, mode="eval")
            result = self._eval(tree.body)

            return f"{clean_expr} = {result}"

        except Exception:
            return "Calculator could not evaluate the given expression."
