"""
Calculator tool — safe math evaluation.
"""

import ast
import operator
from dataclasses import dataclass


# Supported binary operators
BINOP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

# Supported unary operators
UNARYOP_MAP = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class _SafeEval(ast.NodeVisitor):
    """Safe AST evaluator for mathematical expressions."""

    def visit_Constant(self, node):
        """Handle numeric constants."""
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Num(self, node):
        """Handle legacy Num nodes (Python < 3.8)."""
        return node.n

    def visit_BinOp(self, node):
        """Handle binary operations like 2 + 3."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_func = BINOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(left, right)

    def visit_UnaryOp(self, node):
        """Handle unary operations like -5."""
        operand = self.visit(node.operand)
        op_func = UNARYOP_MAP.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand)

    def visit_Expr(self, node):
        """Handle expression statements."""
        return self.visit(node.value)

    def generic_visit(self, node):
        """Reject unknown node types for safety."""
        raise ValueError(f"Unsupported node type: {type(node).__name__}")


def _safe_eval(expr: str) -> float:
    """Safely evaluate a math expression without using eval()."""
    node = ast.parse(expr.strip(), mode='eval')
    evaluator = _SafeEval()
    result = evaluator.visit(node.body)
    return result


@dataclass
class CalculatorTool:
    """
    Safely evaluate mathematical expressions.

    Supports: +, -, *, /, **, //, %
    Uses AST parsing for security (no arbitrary code execution).
    """

    def run(self, expression: str) -> str:
        """
        Evaluate a math expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Result as string, or error message
        """
        try:
            expr = expression.strip()

            # Basic input validation - only allow digits, operators, parentheses, spaces, dots
            allowed = set("0123456789+-*/(). **//% \t")
            if not all(c in allowed for c in expr):
                return f"[Error: Invalid characters in expression]"

            result = _safe_eval(expr)

            # Format the result
            if isinstance(result, float):
                # Remove trailing zeros
                result = f"{result:.10g}"

            return str(result)

        except ZeroDivisionError:
            return "[Error: Division by zero]"
        except ValueError as e:
            return f"[Error: {e}]"
        except SyntaxError as e:
            return f"[Error: Syntax error in expression]"
        except Exception as e:
            return f"[Error calculating '{expression}': {e}]"


# Decorator-based tool
def calculator_tool(expression: str) -> str:
    """Calculate math expression. Args: expression (str)."""
    tool = CalculatorTool()
    return tool.run(expression)
