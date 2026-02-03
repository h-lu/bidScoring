"""Contextual Retrieval Generator

Implements Anthropic-style contextual retrieval by generating context prefixes
for document chunks using an LLM.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

import logging
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# Prompt template for context generation (based on Anthropic's approach)
CONTEXT_GENERATION_PROMPT = """You are analyzing a chunk of text from a document. Your task is to provide a brief context that explains what this chunk is about and how it relates to the broader document.

Document Information:
- Document Title: {document_title}
- Section Title: {section_title}

{surrounding_context}

Chunk to analyze:
"""

CHUNK_CONTENT = """{chunk_text}"""

CONTEXT_INSTRUCTION = """
Provide a concise context (1-2 sentences) that explains:
1. What this chunk contains
2. How it fits into the broader document context

The context should help someone understand this chunk even if they haven't read the rest of the document. Be specific but concise."""


def _build_prompt(
    chunk_text: str,
    document_title: str,
    section_title: Optional[str] = None,
    surrounding_chunks: Optional[List[str]] = None,
) -> str:
    """Build the prompt for context generation.

    Args:
        chunk_text: The text of the chunk to generate context for
        document_title: The title of the document
        section_title: The title of the section containing the chunk (optional)
        surrounding_chunks: List of surrounding chunk texts for context (optional)

    Returns:
        The complete prompt string
    """
    section_display = section_title if section_title else "Not specified"

    surrounding_context = ""
    if surrounding_chunks and len(surrounding_chunks) > 0:
        surrounding_text = "\n\n".join(
            f"Context chunk {i+1}:\n{chunk[:200]}..."
            for i, chunk in enumerate(surrounding_chunks[:2])  # Limit to 2 chunks
        )
        surrounding_context = f"\nSurrounding Context:\n{surrounding_text}"

    prompt = CONTEXT_GENERATION_PROMPT.format(
        document_title=document_title,
        section_title=section_display,
        surrounding_context=surrounding_context,
    )

    prompt += CHUNK_CONTENT.format(chunk_text=chunk_text)
    prompt += CONTEXT_INSTRUCTION

    return prompt


def _generate_rule_based_context(
    document_title: str,
    section_title: Optional[str] = None,
) -> str:
    """Generate a rule-based context when LLM fails.

    Args:
        document_title: The title of the document
        section_title: The title of the section (optional)

    Returns:
        A simple context string
    """
    if section_title:
        return f"This content is from the document '{document_title}', specifically from the section '{section_title}'."
    else:
        return f"This content is from the document '{document_title}'."


class ContextualRetrievalGenerator:
    """Generator for contextual retrieval prefixes.

    Uses an LLM to generate context prefixes for document chunks,
    with fallback to rule-based generation if the LLM fails.
    """

    def __init__(
        self,
        client: OpenAI,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ):
        """Initialize the contextual retrieval generator.

        Args:
            client: OpenAI client instance
            model: Model to use for generation (defaults to gpt-4)
            temperature: Sampling temperature (default 0.0 for consistent outputs)
            max_tokens: Maximum tokens for the generated context
        """
        self.client = client
        self.model = model or "gpt-4"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_context(
        self,
        chunk_text: str,
        document_title: str,
        section_title: Optional[str] = None,
        surrounding_chunks: Optional[List[str]] = None,
    ) -> str:
        """Generate a context prefix for a chunk.

        Uses an LLM to generate a context that explains what the chunk
        is about and how it relates to the broader document. If the LLM
        fails, falls back to rule-based context generation.

        Args:
            chunk_text: The text of the chunk to generate context for
            document_title: The title of the document containing the chunk
            section_title: The title of the section containing the chunk (optional)
            surrounding_chunks: List of surrounding chunk texts for context (optional)

        Returns:
            The generated context prefix string
        """
        # Build the prompt
        prompt = _build_prompt(
            chunk_text=chunk_text,
            document_title=document_title,
            section_title=section_title,
            surrounding_chunks=surrounding_chunks,
        )

        try:
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise, informative context for document chunks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            context = response.choices[0].message.content.strip()

            # Validate the response
            if not context or len(context) < 10:
                logger.warning("LLM returned empty or too short context, using fallback")
                return _generate_rule_based_context(document_title, section_title)

            return context

        except Exception as e:
            logger.warning(f"LLM context generation failed: {e}, using fallback")
            return _generate_rule_based_context(document_title, section_title)

    def generate_context_batch(
        self,
        chunks: List[dict],
    ) -> List[str]:
        """Generate context prefixes for multiple chunks.

        Args:
            chunks: List of dictionaries containing chunk information.
                Each dict should have keys: 'chunk_text', 'document_title',
                and optionally 'section_title' and 'surrounding_chunks'.

        Returns:
            List of generated context strings (one per chunk)
        """
        contexts = []
        for chunk in chunks:
            context = self.generate_context(
                chunk_text=chunk["chunk_text"],
                document_title=chunk["document_title"],
                section_title=chunk.get("section_title"),
                surrounding_chunks=chunk.get("surrounding_chunks"),
            )
            contexts.append(context)
        return contexts
