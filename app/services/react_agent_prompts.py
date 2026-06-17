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
    "- In your first Thought, write a numbered checklist of every distinct piece of "
    "information you need to answer the question. Use this exact format:\n"
    "  [ ] 1. <what you need>\n"
    "  [ ] 2. <what you need>\n"
    "  Then state which item you are searching for first.\n"
    "- In every Thought after an Observation, repeat the checklist and mark completed "
    "items [x]. State which item you are addressing next.\n"
    "- You may only write Final Answer when every item in your checklist is marked [x]. "
    "If any item is still [ ], run that search before writing Final Answer.\n"
    "- Produce at most one Action per response, then stop and wait for its Observation.\n"
    "- Never write an \"Observation:\" section yourself — it will be provided to you.\n"
    "- A question is complex if it asks you to (1) apply a legal or technical standard "
    "to a set of facts, (2) compare two things, or (3) construct a counter-argument. "
    "In these cases, each standard or side of the comparison is a separate checklist item "
    "requiring its own search.\n"
    "- If a tool returned a result, base your Final Answer on that result. "
    "Do not contradict it with your own knowledge.\n"
    "- If no Action is needed, skip directly to \"Final Answer:\".\n"
    "- If rag_search returns no results or results that do not answer the "
    "question, reformulate the query (different keywords, broader or more "
    "specific phrasing) and try again with another Action.\n"
    "- If rag_search still returns nothing useful after reformulating, do "
    "not answer from your own knowledge. Say in the Final Answer that the "
    "information was not found in the user's documents.\n"
    "- When your Final Answer is based on retrieved passages, cite the "
    "source filename(s) from the Observation, e.g. \"Source: <filename>\".\n"
)


_EXAMPLE_TEMPLATE = (
    "\n\nExample (simple — single search):\n\n"
    "Question: What does the onboarding guide say about setting up a dev environment?\n"
    "Thought: I need:\n"
    "  [ ] 1. Onboarding guide instructions for dev environment setup\n"
    "  Searching for item 1.\n"
    "Action: {tool_name}\n"
    "Action Input: {{\"query\": \"dev environment setup onboarding guide\"}}\n"
    "Observation: Source: onboarding.pdf (page 2, section Getting Started), uploaded 2026-05-01\n"
    "Similarity: 0.842\n"
    "Content: \"Clone the repository, then run scripts/setup.sh to install dependencies "
    "and configure the local environment.\"\n"
    "Thought:\n"
    "  [x] 1. Onboarding guide instructions for dev environment setup — found.\n"
    "  All items complete. Writing Final Answer.\n"
    "Final Answer: According to the onboarding guide, clone the repository and run "
    "scripts/setup.sh to install dependencies and configure the local environment. "
    "Source: onboarding.pdf\n\n"
    "Example (complex — two searches required):\n\n"
    "Question: What were the key findings of the audit report, and how do they compare "
    "to the compliance requirements?\n"
    "Thought: I need:\n"
    "  [ ] 1. Key findings from the audit report\n"
    "  [ ] 2. The compliance requirements to compare against\n"
    "  Searching for item 1.\n"
    "Action: {tool_name}\n"
    "Action Input: {{\"query\": \"audit report key findings\"}}\n"
    "Observation: Source: audit_2024.pdf (page 4, section Summary), uploaded 2026-01-10\n"
    "Similarity: 0.811\n"
    "Content: \"The audit identified three critical gaps: missing access logs, "
    "unencrypted backups, and outdated password policies.\"\n"
    "Thought:\n"
    "  [x] 1. Key findings — found: three gaps (access logs, backups, passwords).\n"
    "  [ ] 2. Compliance requirements — not yet searched.\n"
    "  Searching for item 2.\n"
    "Action: {tool_name}\n"
    "Action Input: {{\"query\": \"compliance requirements access control encryption password policy\"}}\n"
    "Observation: Source: compliance_policy.pdf (page 2, section Requirements), uploaded 2025-11-03\n"
    "Similarity: 0.794\n"
    "Content: \"All systems must maintain access logs for 12 months, encrypt backups "
    "using AES-256, and enforce password rotation every 90 days.\"\n"
    "Thought:\n"
    "  [x] 1. Key findings — three gaps identified.\n"
    "  [x] 2. Compliance requirements — found.\n"
    "  All items complete. Writing Final Answer.\n"
    "Final Answer: The audit found three gaps — missing access logs, unencrypted backups, "
    "and outdated password policies — each of which directly violates a compliance "
    "requirement: logs must be kept 12 months, backups must use AES-256 encryption, "
    "and passwords must rotate every 90 days. Source: audit_2024.pdf, compliance_policy.pdf"
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
