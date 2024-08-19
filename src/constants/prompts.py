"""Prompt template constants for different LLMs."""

CLAUDE_3_PROMPT_RAG_SIMPLE = """
System: {system_message}
Human: {user_message}

-----------
This is the context:

<context>
{{context}}
</context>
-----------
This is the question:

<question>
{{question}}
</question>

Assistant:
"""

GEMMA_PROMPT_RAG_SIMPLE = """
<start_of_turn>user
System: {system_message}
User: {user_message}

-----------
This is the context:
{{context}}

-----------
This is the question:
{{question}}
<end_of_turn>

<start_of_turn>model
"""

LLAMA_3_PROMPT_RAG_SIMPLE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> {system_message} <|eot_id|>
<|start_header_id|>user<|end_header_id|> {user_message}

-----------
This is the context:
{{context}}

-----------
This is the question:
{{question}}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

MISTRAL_7B_PROMPT_RAG_SIMPLE = """
<s>[INST]
System: {system_message}
User: {user_message}

-----------
This is the context:
{{context}}

-----------
This is the question:
{{question}}

[/INST]
"""

GPT_4_PROMPT_RAG_SIMPLE = """
System: {system_message}
User: {user_message}

-----------
This is the context:
{{context}}

-----------
This is the question:
{{question}}

Assistant:
"""

CLAUDE_3_MIXTURE_RAG = """
System: {system_message}
Human: {user_message}
Responses from the models:

From the first model:
{{output_1}}

From the second model:
{{output_2}}

From the third model:
{{output_3}}

Assistant:
"""

GEMMA_MIXTURE_RAG = """
<start_of_turn>user
System: {system_message}
User: {user_message}

Responses from the models:

From the first model:
{{output_1}}

From the second model:
{{output_2}}

From the third model:
{{output_3}}

<end_of_turn>

<start_of_turn>model
"""

LLAMA_3_MIXTURE_RAG = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|> {system_message} <|eot_id|>
<|start_header_id|>user<|end_header_id|> {user_message}

Responses from the models:

From the first model:
{{output_1}}

From the second model:
{{output_2}}

From the third model:
{{output_3}}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

MISTRAL_7B_MIXTURE_RAG = """
<s>[INST]
System: {system_message}
User: {user_message}

Responses from the models:

From the first model:
{{output_1}}

From the second model:
{{output_2}}

From the third model:
{{output_3}}

[/INST]
"""

GPT_4_MIXTURE_RAG = """
System: {system_message}
User: {user_message}

Responses from the models:

From the first model:
{{output_1}}

From the second model:
{{output_2}}

From the third model:
{{output_3}}

Assistant:
"""
