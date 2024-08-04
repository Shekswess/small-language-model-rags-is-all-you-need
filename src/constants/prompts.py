"""Prompt template constants for different LLMs."""

CLAUDE_3_PROMPT_RAG_SIMPLE = """
System: {system_message}
Human: {user_message}
-----------
<context>
{{context}}
</context>
-----------
<question>
{{question}}
</question>
Assistant:
"""
