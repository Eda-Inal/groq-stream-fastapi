"""
Prompt construction and response parsing for the text-based ReAct agent.

Unlike ChatService, this agent does not use native function-calling
(`tools=`). Tool descriptions are embedded as plain text in the system
prompt, and the model is expected to reply in a fixed
Thought/Action/Action Input/Final Answer format, which is parsed with
`parse_react_response`.
"""

from __future__ import annotations

import json
import re


_ROLE_NO_TOOLS = (
    "You are a helpful assistant. Before answering, think step by step "
    "about the user's question.\n\n"
    "Respond using exactly this format:\n\n"
    "Thought: <your reasoning about the question>\n"
    "Final Answer: <your answer to the user>\n\n"
    "Rules:\n"
    "- Always start your response with \"Thought:\".\n"
    "- Always follow it with \"Final Answer:\" containing the answer to give the user.\n"
    "- Do not skip either section."
)


_ROLE_WITH_TOOLS = (
    "You are a helpful assistant that can use tools to answer questions. "
    "You work in a loop of Thought, Action, and Observation steps until you "
    "have enough information to give a Final Answer.\n\n"
    "Respond using exactly this format:\n\n"
    "Thought: <reasoning about what to do next>\n"
    "Action: <the name of one tool to call>\n"
    "Action Input: <a JSON object with the tool's arguments>\n\n"
    "After an Action, you will be given:\n"
    "Observation: <the tool's result>\n\n"
    "You can repeat Thought/Action/Observation as many times as needed. "
    "When you have enough information, respond instead with:\n\n"
    "Thought: <final reasoning>\n"
    "Final Answer: <the answer to give the user>\n\n"
    "Rules:\n"
    "- Always start your response with \"Thought:\".\n"
    "- Produce at most one Action per response, then stop and wait for its Observation.\n"
    "- Never write an \"Observation:\" section yourself — it will be provided to you.\n"
    "- If a tool returned a result, base your Final Answer on that result. "
    "Do not contradict it with your own knowledge.\n"
    "- If no Action is needed, skip directly to \"Final Answer:\"."
)


_EXAMPLE_TEMPLATE = (
    "\n\nExample:\n\n"
    "Question: What is the current EUR/USD exchange rate?\n"
    "Thought: This requires up-to-date information, so I should search for it.\n"
    "Action: {tool_name}\n"
    "Action Input: {{\"query\": \"EUR/USD exchange rate\"}}\n"
    "Observation: 1 EUR = 1.0842 USD (2026-06-15)\n"
    "Thought: I now have the exchange rate.\n"
    "Final Answer: The current EUR/USD rate is 1.0842."
)


def _format_tool(tool: dict) -> str:
    fn = tool.get("function", {}) if isinstance(tool, dict) else {}
    name = fn.get("name", "unknown")
    description = fn.get("description", "")
    parameters = fn.get("parameters", {})
    return f"- {name}: {description}\n  Parameters: {json.dumps(parameters)}"


def build_system_prompt(available_tools: list[dict]) -> str:
    """
    Build the ReAct system prompt.

    If `available_tools` is empty, the prompt only describes the
    Thought -> Final Answer format (no Action/Observation section).
    Otherwise it includes the full Thought/Action/Observation loop, the
    list of available tools, and a worked example using the first tool.
    """
    if not available_tools:
        return _ROLE_NO_TOOLS

    tools_section = "\n\nAvailable tools:\n" + "\n".join(
        _format_tool(t) for t in available_tools
    )

    first_tool = available_tools[0]
    first_tool_name = (
        first_tool.get("function", {}).get("name", "tool")
        if isinstance(first_tool, dict)
        else "tool"
    )
    example = _EXAMPLE_TEMPLATE.format(tool_name=first_tool_name)

    return _ROLE_WITH_TOOLS + tools_section + example


# Section markers recognised in a ReAct completion, in the order they are
# searched for. "Action Input" is listed separately from "Action" since both
# are valid standalone markers (`Action:` does not match `Action Input:`).
_SECTION_MARKERS = ["Thought", "Action Input", "Action", "Observation", "Final Answer"]


def _find_sections(text: str) -> list[tuple[str, int, int]]:
    """Find all ReAct section markers in `text`.

    Returns a list of (marker, marker_start, content_start) tuples sorted by
    position of occurrence.
    """
    sections: list[tuple[str, int, int]] = []
    for marker in _SECTION_MARKERS:
        for m in re.finditer(rf"(?:^|\n){re.escape(marker)}:[ \t]*", text):
            sections.append((marker, m.start(), m.end()))
    sections.sort(key=lambda s: s[1])
    return sections


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced `{...}` substring in `text`, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_action_input(content: str) -> dict | str | None:
    if not content:
        return None
    obj_str = _extract_json_object(content)
    if obj_str is None:
        return content
    try:
        return json.loads(obj_str)
    except json.JSONDecodeError:
        return content


def parse_react_response(text: str) -> dict:
    """
    Parse a model completion into ReAct fields.

    Returns a dict with keys `thought`, `action`, `action_input`, and
    `final_answer`. Any field not present in `text` is None. `action_input`
    is parsed as JSON when it contains a valid JSON object, otherwise
    returned as the raw trimmed string.
    """
    result: dict = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    sections = _find_sections(text)
    for i, (marker, _marker_start, content_start) in enumerate(sections):
        content_end = sections[i + 1][1] if i + 1 < len(sections) else len(text)
        content = text[content_start:content_end].strip()

        if marker == "Thought" and result["thought"] is None:
            result["thought"] = content
        elif marker == "Action" and result["action"] is None:
            result["action"] = content.splitlines()[0].strip() if content else None
        elif marker == "Action Input" and result["action_input"] is None:
            result["action_input"] = _parse_action_input(content)
        elif marker == "Final Answer" and result["final_answer"] is None:
            result["final_answer"] = content

    return result
