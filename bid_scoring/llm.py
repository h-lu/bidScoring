# bid_scoring/llm.py
"""LLM client and models for bid scoring."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


def get_model_for_task(settings: dict, task: str) -> str:
    """Get the appropriate model for a specific task.
    
    Args:
        settings: Configuration settings dictionary
        task: Task name to look up
        
    Returns:
        Model name to use for the task
    """
    task_models = settings.get("OPENAI_LLM_MODELS", {})
    return task_models.get(task, settings.get("OPENAI_LLM_MODEL_DEFAULT", "gpt-4"))


@dataclass
class Citation:
    """Represents a citation from source documents."""
    source_number: int
    cited_text: str
    supports_claim: str

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        return cls(
            source_number=data["source_number"],
            cited_text=data["cited_text"],
            supports_claim=data["supports_claim"]
        )


@dataclass
class ScoreResult:
    """Represents a scoring result for a dimension."""
    dimension: str
    score: float
    max_score: float
    reasoning: str
    citations: list[Citation] = field(default_factory=list)
    evidence_found: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreResult":
        citations = [Citation.from_dict(c) for c in data.get("citations", [])]
        return cls(
            dimension=data["dimension"],
            score=data["score"],
            max_score=data["max_score"],
            reasoning=data["reasoning"],
            citations=citations,
            evidence_found=data.get("evidence_found", False)
        )


class LLMClient:
    """Client for LLM API interactions."""

    def __init__(self, settings: dict):
        """Initialize LLM client with settings.
        
        Args:
            settings: Configuration dictionary with API keys and model settings
            
        Raises:
            ValueError: If OPENAI_API_KEY is not provided
        """
        self.settings = settings
        self.api_key = settings.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.base_url = settings.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.timeout = settings.get("OPENAI_TIMEOUT", 60.0)
        self.max_retries = settings.get("OPENAI_MAX_RETRIES", 3)
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            self._client = OpenAI(**client_kwargs)
        return self._client

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Get completion from LLM.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model to use (defaults to settings)
            temperature: Sampling temperature
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text content
        """
        client = self._get_client()
        model = model or self.settings.get("OPENAI_LLM_MODEL_DEFAULT", "gpt-4")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content

    def complete_with_schema(
        self,
        messages: list[dict],
        schema: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Any:
        """Get structured completion validated against JSON schema.
        
        Args:
            messages: List of message dictionaries
            schema: JSON schema for response validation
            model: Model to use
            temperature: Sampling temperature
            
        Returns:
            Parsed and validated response
            
        Raises:
            ValueError: If response cannot be parsed or validated
        """
        client = self._get_client()
        model = model or self.settings.get("OPENAI_LLM_MODEL_DEFAULT", "gpt-4")
        
        # Add schema instruction to system message or append
        schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"
        
        # Find system message or create one
        modified_messages = []
        system_found = False
        for msg in messages:
            if msg.get("role") == "system":
                modified_messages.append({
                    "role": "system",
                    "content": msg["content"] + schema_instruction
                })
                system_found = True
            else:
                modified_messages.append(msg)
        
        if not system_found:
            modified_messages.insert(0, {
                "role": "system",
                "content": f"You are a helpful assistant.{schema_instruction}"
            })
        
        response = client.chat.completions.create(
            model=model,
            messages=modified_messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        
        # Basic schema validation
        self._validate_against_schema(parsed, schema)
        
        return parsed

    def _validate_against_schema(self, data: Any, schema: dict, path: str = ""):
        """Basic JSON schema validation.
        
        Args:
            data: Data to validate
            schema: JSON schema
            path: Current path for error reporting
            
        Raises:
            ValueError: If validation fails
        """
        schema_type = schema.get("type")
        
        if schema_type == "object":
            if not isinstance(data, dict):
                raise ValueError(f"Expected object at {path}, got {type(data).__name__}")
            
            # Check required fields
            for required in schema.get("required", []):
                if required not in data:
                    raise ValueError(f"Missing required field: {path}.{required}")
            
            # Validate properties
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop in data:
                    self._validate_against_schema(data[prop], prop_schema, f"{path}.{prop}")
        
        elif schema_type == "array":
            if not isinstance(data, list):
                raise ValueError(f"Expected array at {path}, got {type(data).__name__}")
            
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    self._validate_against_schema(item, items_schema, f"{path}[{i}]")
            
            # Check minItems
            min_items = schema.get("minItems")
            if min_items and len(data) < min_items:
                raise ValueError(f"Array at {path} has fewer than {min_items} items")
        
        elif schema_type == "string":
            if not isinstance(data, str):
                raise ValueError(f"Expected string at {path}, got {type(data).__name__}")
        
        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                raise ValueError(f"Expected number at {path}, got {type(data).__name__}")
        
        elif schema_type == "boolean":
            if not isinstance(data, bool):
                raise ValueError(f"Expected boolean at {path}, got {type(data).__name__}")
        
        elif schema_type == "integer":
            if not isinstance(data, int):
                raise ValueError(f"Expected integer at {path}, got {type(data).__name__}")
