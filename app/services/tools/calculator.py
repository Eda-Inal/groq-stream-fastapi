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
    description = "Evaluates complete mathematical expressions including parentheses."

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
        
        tree = ast.parse(clean_expr, mode="eval")
        return _eval(tree.body)

    async def should_run(self, messages: list[dict[str, str]]) -> bool:
        last = messages[-1]
        if last.get("role") != "user":
            return False
        content = last.get("content", "")
       
        return bool(re.search(r'\d+\s*[+\-*/^]\s*\d+', content)) or "(" in content

    async def run(self, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        
        # This finds the longest string of math characters
        # It allows numbers, spaces, and all operators including parentheses
        math_matches = re.findall(r'[0-9+\-*/.^() \*\*]+', content)
        if not math_matches:
            return "Error: No math found."

        # Pick the longest match (to get the full equation, not just parts)
        expr = max(math_matches, key=len).strip()
        
        # If the match is just a single number, skip it
        if expr.isdigit():
            return "Error: Not a calculation."

        try:
            result = self.safe_eval(expr)
            return f"Calculator result: {expr} = {result}"
        except Exception as e:
            return f"Calculator error evaluating '{expr}': {str(e)}"