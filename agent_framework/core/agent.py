"""
Agent module — the core of the Agent framework.

PURPOSE:
  An Agent is an AI system that can:
  1. Receive a task (text input)
  2. Think and decide actions
  3. Use tools to interact with the world
  4. Return a final result

KEY CONCEPTS:
  - BaseAgent: Abstract base class with common functionality
  - ReActAgent: Implements the ReAct (Reasoning + Acting) pattern
  - PlanAndExecuteAgent: Implements Plan-then-Execute pattern

REACT PATTERN (ReActAgent):
  The agent loops through these steps:
    Thought: What should I do?
    Action: Call a tool (or give final answer)
    Observation: See the tool's result
    ...repeat until final answer

  This is inspired by the paper "ReAct: Synergizing Reasoning and Acting in
  Language Models" (2023). It gives the model a way to:
  1. Reason about what to do (not just generate text)
  2. Take actions and see results
  3. Course-correct based on observations

PLAN-AND-EXECUTE PATTERN (PlanAndExecuteAgent):
  More structured, two-phase approach:
    Phase 1: Given the task, make a plan (a list of steps)
    Phase 2: Execute each step in order
    Phase 3: Synthesize results into final answer

  This is better for complex multi-step tasks where
  planning upfront saves effort.

WHY TWO DIFFERENT AGENTS?
  Different tasks suit different patterns:
  - Simple Q&A: ReAct (quick, minimal overhead)
  - Complex workflows: Plan-and-Execute (structured, fewer mid-course corrections)
"""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from .llm import LLM
from .message import Message, MessageRole
from .executor import ActionExecutor
from .memory import SummarizationMemory

# Safety limit to prevent infinite loops
# In practice, if an agent runs this many turns without finishing,
# something is likely wrong (bad prompting, no tools, etc.)
MAX_TURNS_DEFAULT = 50


# =============================================================================
# HELPER FUNCTION — Parsing LLM Output
# =============================================================================

def _parse_thought_output(text: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    Parse the LLM's text output to extract structured action information.

    The LLM generates free-form text, but we need structured data:
    - What was the thought?
    - What action (if any) to take?
    - What arguments for that action?

    EXPECTED FORMAT:
      Thought: I need to check the weather
      Action: get_weather
      Action Args: {"city": "Beijing"}

    OR for final answer:
      Thought: I have all the information
      Final Answer: Beijing is sunny and 25°C

    Args:
        text: The raw text output from the LLM

    Returns:
        A tuple of (thought, action_name, action_args) where:
        - thought: The reasoning text (or "" if not found)
        - action_name: The tool name to call, "FINAL_ANSWER", or None
        - action_args: Dict of arguments (or None if no action)

    REGEX EXPLANATION:
      r"Thought[:\\s]*(.+?)(?=\\n(?:Action|Final Answer)|$)"
      - Thought[:\\s]* : "Thought:" followed by whitespace
      - (.+?) : capture group 1: any characters (non-greedy)
      - (?=\\n(?:Action|Final Answer)|$) : lookahead for end of line + Action/Final Answer

      This pattern extracts the thought text that comes before an Action or end of string.
    """
    # Extract the Thought portion
    # (?=...) is a lookahead — we capture but don't consume what follows
    thought_match = re.search(
        r"Thought[:\s]*(.+?)(?=\n(?:Action|Final Answer)|$)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    # Check if the LLM gave a final answer
    # If yes, we return immediately with "FINAL_ANSWER" as the action
    if re.search(r"Final Answer[:\s]*(.+)", text, re.DOTALL | re.IGNORECASE):
        return thought, "FINAL_ANSWER", None

    # Extract the Action name
    action_match = re.search(r"Action[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    action_name = action_match.group(1).strip() if action_match else None

    # Extract the Action Arguments (as a JSON string)
    args_match = re.search(r"Action Args[:\s]*(.+)", text, re.IGNORECASE)
    action_args = None
    if args_match:
        try:
            # Parse the JSON string into a Python dict
            action_args = json.loads(args_match.group(1).strip())
        except json.JSONDecodeError:
            # If the LLM gave malformed JSON, we can't use the arguments
            # Return empty dict rather than crashing
            action_args = {}

    return thought, action_name, action_args


# =============================================================================
# BASE AGENT — Common functionality for all agents
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all Agent implementations.

    WHY ABSTRACT?
      We want to define the interface (run method) without committing
      to a specific agent strategy (ReAct vs Plan-and-Execute).

      Any class that inherits from BaseAgent MUST implement run().

    SHARED COMPONENTS:
      - LLM: The brain — makes decisions
      - Executor: The hands — runs tools
      - Memory: The context — remembers conversation history
      - max_turns: Safety limit on iterations

    WHAT'S COMMON ACROSS AGENTS:
      - Building the system prompt (listing available tools)
      - Managing the message history
      - The max_turns safety limit
    """

    def __init__(
        self,
        llm: LLM,
        executor: ActionExecutor,
        memory: Optional[SummarizationMemory] = None,
        max_turns: int = MAX_TURNS_DEFAULT,
    ):
        """
        Initialize the base agent.

        Args:
            llm: The LLM to use for decision-making.
            executor: The tool executor for running actions.
            memory: Conversation memory. If None, creates SummarizationMemory.
            max_turns: Maximum iterations before giving up. Default 50.
        """
        self.llm = llm
        self.executor = executor
        # Use provided memory or create a default SummarizationMemory
        self.memory = memory or SummarizationMemory(llm=self.llm)
        self.max_turns = max_turns

    @abstractmethod
    def run(self, task: str) -> str:
        """
        Execute a task and return the result.

        This is the main entry point — subclasses implement specific strategies.
        """
        pass

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt that instructs the LLM how to behave.

        The system prompt is critical — it tells the LLM:
        1. What role it plays (helpful agent)
        2. How to format its responses (Thought/Action/Action Args)
        3. What tools are available

        WHY A SYSTEM PROMPT INSTEAD OF USER PROMPT?
          System prompts set the "behavior" of the model — they're not
          part of the conversation itself. The model processes them
          differently than user messages.

        TOOL LIST:
          We enumerate all available tools with their descriptions.
          The LLM uses this to decide WHICH tool to call and WHEN.
        """
        # Get all registered tools from the executor's registry
        tools = self.executor.registry.list_tools()

        # Format tools as a list: "- tool_name: description"
        tool_desc = "\n".join(
            f"- {t.name}: {t.description}"
            for t in tools
        )

        return (
            "You are a helpful AI agent.\n"
            "When you need to use tools, respond with:\n"
            "Thought: ...\n"          # What you're thinking
            "Action: tool_name\n"    # Which tool to call
            'Action Args: {"arg1": "value1"}\n\n'  # Tool arguments
            "Available tools:\n"
            f"{tool_desc}\n\n"
            "If you have enough information, respond with:\n"
            "Thought: ...\n"
            "Final Answer: ..."
        )

    def _get_messages(self, task: str) -> List[Message]:
        """
        Build the message list for an LLM call.

        Message order (MOST IMPORTANT):
          1. SYSTEM: Instructions and available tools
          2. Memory: Previous conversation (summary + recent messages)
          3. USER: The current task

        This order is critical — it mirrors how humans communicate:
        - First, establish context (who am I, what can I do)
        - Then, remind of past conversation
        - Finally, state the current request
        """
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._build_system_prompt())
        ]
        messages.extend(self.memory.get_messages())
        messages.append(Message(role=MessageRole.USER, content=task))
        return messages


# =============================================================================
# REACT AGENT — Thought -> Action -> Observation loop
# =============================================================================

class ReActAgent(BaseAgent):
    """
    ReAct pattern agent.

    Loop structure:
      1. Send messages to LLM → get response
      2. Parse response for Thought/Action/Args
      3. If FINAL_ANSWER → return it
      4. If action → execute via executor
      5. Add assistant + tool_result to messages
      6. Loop back to step 1

    WHY THIS LOOP WORKS:
      - The LLM can see all previous messages (history)
      - Each tool result becomes a new message for the LLM to consider
      - The LLM can course-correct based on tool results
      - Eventually it has enough info to give a FINAL_ANSWER

    TERMINATION CONDITIONS:
      - LLM outputs "Final Answer:" → return immediately
      - LLM outputs unparseable response → return raw response
      - max_turns reached → give up
    """

    def run(self, task: str) -> str:
        """
        Execute a task using the ReAct loop.
        """
        # Build the initial message list (system prompt + memory + user task)
        messages = self._get_messages(task)
        turns = 0

        while turns < self.max_turns:
            turns += 1

            # Step 1: Ask the LLM for its next action/thought
            response = self.llm.generate(messages)

            # Step 2: Parse the response
            _, action, args = _parse_thought_output(response)

            # Step 3: Check if we're done
            if action == "FINAL_ANSWER":
                # Extract the actual answer text
                match = re.search(
                    r"Final Answer[:\s]*(.+)",
                    response,
                    re.DOTALL | re.IGNORECASE
                )
                answer = match.group(1).strip() if match else response
                # Remember this for future context
                self.memory.add(Message(role=MessageRole.ASSISTANT, content=answer))
                return answer

            # Step 4: Execute the tool if one was requested
            if action and action != "FINAL_ANSWER":
                # Run the tool and get the result
                result = self.executor.run(action, args or {})

                # Add both the assistant's thought AND the tool result to history
                # This way, the LLM sees what it said and what happened
                messages.append(Message(role=MessageRole.ASSISTANT, content=response))
                messages.append(Message(role=MessageRole.TOOL_RESULT, content=result))

            else:
                # LLM didn't give a parseable action.
                # This can happen with a poorly tuned model or wrong prompting.
                # Return what we have rather than looping forever.
                self.memory.add(Message(role=MessageRole.ASSISTANT, content=response))
                return response

        # Safety: if we hit max turns, give up
        return "Max turns reached without final answer."


# =============================================================================
# PLAN-AND-EXECUTE AGENT — Plan first, then execute
# =============================================================================

class PlanAndExecuteAgent(BaseAgent):
    """
    Plan-and-Execute pattern agent.

    Three phases:
      1. PLANNING: Ask LLM to break down the task into steps
      2. EXECUTION: Execute each step (possibly using tools)
      3. SYNTHESIS: Ask LLM to combine results into final answer

    WHY PLAN FIRST?
      For complex tasks, it's often better to plan upfront:
      - The LLM sees the whole picture before acting
      - Less "drift" — each step is aligned with the plan
      - Easier to course-correct — the plan is explicit

      Compare to ReAct, where each step is myopic (only sees immediate history).

    TRADE-OFFS:
      - Pros: More structured, fewer mid-course corrections
      - Cons: More LLM calls (plan + each step + synthesis)
      - Cons: If plan is wrong, later steps may be wasted

    SUITABLE FOR:
      - Complex, multi-step tasks (e.g., "write a report")
      - Tasks where the steps are clearly delineated
      - Tasks where following a plan is more important than reacting
    """

    def run(self, task: str) -> str:
        """
        Execute a task using the Plan-and-Execute strategy.
        """
        messages = self._get_messages(task)

        # =========================================================================
        # PHASE 1: PLANNING
        # =========================================================================
        # Ask the LLM to break down the task into explicit steps
        plan_response = self.llm.generate(messages)

        # Parse the plan
        # Expected format: "Step 1: Do X\nStep 2: Do Y\n..."
        step_matches = re.findall(
            r"Step \d+[:\s]*(.+?)(?=(?:Step \d+)|$)",
            plan_response,
            re.DOTALL | re.IGNORECASE
        )
        steps = [s.strip() for s in step_matches if s.strip()]

        # Fallback: if we couldn't parse any steps, just treat the whole
        # task as a single step
        if not steps:
            steps = [task]

        # =========================================================================
        # PHASE 2: EXECUTION
        # =========================================================================
        # Add the plan to the conversation history
        messages.append(Message(role=MessageRole.ASSISTANT, content=plan_response))

        execution_results: List[str] = []

        for step in steps:
            # Ask the LLM to execute THIS specific step
            # We provide: the full conversation + the step to execute
            step_msg = Message(
                role=MessageRole.USER,
                content=f"Execute this step: {step}"
            )
            step_response = self.llm.generate(messages + [step_msg])
            messages.append(Message(role=MessageRole.ASSISTANT, content=step_response))

            # Try to extract a tool call from the step's response
            _, action, args = _parse_thought_output(step_response)

            if action and action != "FINAL_ANSWER":
                # Execute the tool and capture the result
                result = self.executor.run(action, args or {})
                messages.append(Message(role=MessageRole.TOOL_RESULT, content=result))
                execution_results.append(result)
            else:
                # No tool was called — the response itself is the result
                execution_results.append(step_response)

        # =========================================================================
        # PHASE 3: SYNTHESIS
        # =========================================================================
        # Combine all step results into a final answer
        # The LLM is asked to synthesize rather than just concatenate
        synthesize_prompt = (
            f"Original task: {task}\n"
            f"Execution results:\n" +
            "\n".join(f"- {r}" for r in execution_results)
        )
        final = self.llm.generate([Message(role=MessageRole.USER, content=synthesize_prompt)])
        return final
