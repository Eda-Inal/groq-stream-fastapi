import ast
import operator as op
import re
from app.services.tools.base import Tool

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
    description = (
        "Use ONLY for complex math, multi-step equations, or large numbers. "
        "Do NOT use for simple counting, basic word problems, or conceptual questions."
    )

    def safe_eval(self, expr: str) -> float:
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                return _ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
            raise ValueError("Unsupported expression")

        # 1. Handle the power symbol ^ -> **
        expr = expr.replace('^', '**')
        # 2. Clean the expression: keep only math-safe characters
        clean_expr = re.sub(r'[^0-9+\-*/.**() ]', '', expr)
        
        try:
            tree = ast.parse(clean_expr, mode="eval")
            return _eval(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid math syntax: {str(e)}")

    async def run(self, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        
        # Extract potential math expression
        math_matches = re.findall(r'[0-9+\-*/.^() \*\*]+', content)
        if not math_matches:
            return "Error: No mathematical expression detected in the prompt."

        expr = max(math_matches, key=len).strip()
        
        if expr.isdigit():
            return f"Note: '{expr}' is just a number, no calculation needed."

        try:
            result = self.safe_eval(expr)
            return f"Calculator result: {expr} = {result}"
        except Exception as e:
            return f"Calculator error: {str(e)}"