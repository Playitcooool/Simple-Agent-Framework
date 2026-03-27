"""
Multi-agent collaboration module.

PURPOSE:
  Sometimes a single agent isn't enough. Complex tasks benefit from
  multiple specialized agents working together.

  Example: Writing a report might need:
  - A "researcher" agent to gather information
  - A "writer" agent to compose the report
  - A "reviewer" agent to check quality

KEY CONCEPTS:
  - SupervisorAgent: A "manager" agent that orchestrates sub-agents
  - Hierarchical decomposition: The supervisor breaks down a task,
    assigns pieces to specialized sub-agents, then synthesizes results

THIS IS THE "HIERARCHICAL" APPROACH TO MULTI-AGENT:
  vs. "Peer-to-peer" where agents negotiate directly
  vs. "Shared message board" where agents read each other's messages

  Hierarchical is simpler and easier to understand —
  the supervisor is the single source of truth for task decomposition.
"""

import re
from typing import List

from ..core.llm import LLM
from ..core.message import Message, MessageRole
from ..core.agent import BaseAgent


__all__ = ["SupervisorAgent"]


class SupervisorAgent:
    """
    Master agent that coordinates multiple specialized sub-agents.

    HOW IT WORKS:
      1. RECEIVE: Get a high-level task from the user
      2. DECOMPOSE: Ask the LLM to break it into subtasks
      3. DISPATCH: Assign each subtask to an appropriate sub-agent
      4. SYNTHESIZE: Combine results from all sub-agents into a final answer

    SUB-AGENT SELECTION:
      The supervisor uses simple keyword matching to pick the right sub-agent.
      For example, if a subtask contains "research", it goes to the agent
      named "researcher".

      LIMITATION: This is a naive approach. In production, consider:
      - Having sub-agents declare their capabilities
      - Using the LLM to decide routing
      - Or having the LLM output structured JSON for routing

    WHY NOT JUST HAVE ONE AGENT?
      Specialization allows each agent to:
      - Have different system prompts (researcher vs writer)
      - Use different tools (web search vs text editing)
      - Be optimized for their specific task

      A single general-purpose agent often does everything "ok" but nothing "great".
    """

    def __init__(self, llm: LLM, sub_agents: List[BaseAgent]):
        """
        Initialize the supervisor.

        Args:
            llm: The LLM used for decomposition and synthesis decisions.
                 Note: This is the supervisor's LLM, not the sub-agents'.
                 Each sub-agent should have its own LLM (often the same one).
            sub_agents: List of agent instances that can handle subtasks.
                       Each agent MUST have a unique `name` attribute.

        Raises:
            ValueError: If sub_agents is empty or contains agents without names.
        """
        if not sub_agents:
            raise ValueError("SupervisorAgent requires at least one sub-agent")

        self.llm = llm
        # Store agents in a dict keyed by name for easy lookup
        self.sub_agents = {a.name: a for a in sub_agents}

    def run(self, task: str) -> str:
        """
        Execute a complex task by decomposing and delegating to sub-agents.

        This is the main entry point — call this with a high-level task
        and get back the synthesized result.
        """
        # =========================================================================
        # PHASE 1: DECOMPOSITION
        # =========================================================================
        # Ask the LLM to break the task into subtasks
        #
        # We use a structured format: "Subtask 1: ...\nSubtask 2: ..."
        # The LLM is instructed to output exactly this format.
        decompose_prompt = (
            f"Break down the following task into subtasks, one per line, "
            f"each starting with 'Subtask N: '.\n"
            f"Task: {task}\n\n"
            f"Subtasks:"
        )
        plan_response = self.llm.generate([
            Message(role=MessageRole.USER, content=decompose_prompt)
        ])

        # Parse the subtasks from the LLM's response
        # Regex matches: "Subtask 1: Do X" -> captures "Do X"
        subtask_lines = re.findall(
            r"Subtask \d+[:\s]+(.+?)(?:\n|$)",
            plan_response,
            re.DOTALL
        )

        # Fallback: if we couldn't parse, treat the whole task as one subtask
        if not subtask_lines:
            subtask_lines = [task]

        # =========================================================================
        # PHASE 2: DISPATCH AND EXECUTION
        # =========================================================================
        # For each subtask, pick the right agent and delegate
        results: List[str] = []

        for line in subtask_lines:
            # Select the appropriate agent for this subtask
            # Uses simple keyword matching: agent name in subtask text
            agent_name = self._select_agent(line)

            if agent_name not in self.sub_agents:
                # This shouldn't happen if _select_agent works correctly
                raise ValueError(f"No sub-agent found for subtask: {line.strip()}")

            # Run the subtask on the selected agent
            # Note: We're calling run() directly on the sub-agent
            result = self.sub_agents[agent_name].run(line.strip())
            results.append(result)

        # =========================================================================
        # PHASE 3: SYNTHESIS
        # =========================================================================
        # Take all the sub-task results and ask the LLM to combine them
        # This is better than just concatenating — the LLM can:
        # - Resolve conflicts between results
        # - Remove redundancies
        # - Present a coherent final answer
        synthesize_prompt = (
            f"Original task: {task}\n\n"
            f"Subtask results:\n" +
            "\n".join(f"- {r}" for r in results) +
            "\n\n"
            "Provide the final answer:"
        )
        final = self.llm.generate([Message(role=MessageRole.USER, content=synthesize_prompt)])
        return final

    def _select_agent(self, subtask: str) -> str:
        """
        Select the best sub-agent for a given subtask.

        Uses simple keyword matching:
        - Iterate through agent names
        - Return the first one found in the subtask text

        Example:
          subtask = "Research AI trends"
          agents = {"researcher": ..., "writer": ...}
          Returns: "researcher" (because "researcher" is in "Research AI trends")

        LIMITATIONS:
          - No ranking — first match wins, even if another would be better
          - Exact string matching — "research" matches "researcher" but not "researching"
          - Case sensitive (we lowercase, so this is mitigated)

        IMPROVEMENTS FOR PRODUCTION:
          - Use embeddings to find the most similar agent
          - Let the LLM decide based on agent descriptions
          - Add explicit capability declarations to each agent
        """
        subtask_lower = subtask.lower()

        for name in self.sub_agents:
            # Simple substring matching
            if name.lower() in subtask_lower:
                return name

        # Fallback: return the first agent (better than crashing)
        return list(self.sub_agents.keys())[0]
