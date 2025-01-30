from typing import Dict, Any, Optional, Tuple
import json
import jsonschema
from jsonschema import validate, ValidationError
import numpy as np

class JSONValidator:
    """Validates and helps fix JSON responses from LLM"""

    @staticmethod 
    def validate_json(json_str: str, schema: Dict[str, Any], used_ids: Optional[set] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate JSON string against a schema
        Args:
            json_str: JSON string to validate
            schema: Schema to validate against
            used_ids: Optional set of already used IDs to check uniqueness
        Returns: (is_valid, parsed_json, error_message)
        """
        try:
            # First try to parse the JSON
            parsed = json.loads(json_str)
            
            # Then validate against schema
            validate(instance=parsed, schema=schema)
            
            # Check ID uniqueness if needed
            if used_ids is not None and 'id' in parsed:
                if parsed['id'] in used_ids:
                    return False, None, f"ID '{parsed['id']}' is already in use"
                
            return True, parsed, None
            
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON format: {str(e)}"
        except ValidationError as e:
            return False, None, f"Schema validation failed: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

    @staticmethod
    def generate_fix_prompt(original_prompt: str, error_message: str, schema: Dict[str, Any]) -> str:
        """Generate a prompt to fix invalid JSON"""
        return f"""Your previous response was invalid JSON. Please fix it.

Error: {error_message}

Required JSON Schema:
{json.dumps(schema, indent=2)}

Original Request:
{original_prompt}

Please provide a valid JSON response that matches the schema exactly."""

    @staticmethod
    def validate_example(example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate an example has the required input and output fields
        Args:
            example: Example dict to validate
        Returns: (is_valid, error_message)
        """
        if not isinstance(example, dict):
            return False, "Example must be a dictionary"
            
        if 'input' not in example:
            return False, "Example missing 'input' field"
            
        if 'output' not in example:
            return False, "Example missing 'output' field"
            
        if not isinstance(example['input'], (list, np.ndarray)):
            return False, "Example input must be a list or numpy array"
            
        if not isinstance(example['output'], (list, np.ndarray)):
            return False, "Example output must be a list or numpy array"
            
        return True, None

    # Common schemas
    CONCEPT_SCHEMA = {
        "type": "object",
        "required": ["concepts"],
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "name", "description", "confidence"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "rules": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
    }

    CONCEPT_VALIDATION_SCHEMA = {
        "type": "object",
        "properties": {
            "generalization_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "consistency_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "supports_examples": {
                "type": "array",
                "items": {"type": "boolean"}
            },
            "explanation": {"type": "string"}
        },
        "required": ["generalization_score", "consistency_score", "supports_examples", "explanation"]
    }

    PRIMITIVE_SCHEMA = {
        "type": "object",
        "required": ["primitive"],
        "properties": {
            "primitive": {
                "type": "object",
                "required": ["id", "name", "description", "parameters", "implementation_guide", "applicability", "examples"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    },
                    "implementation_guide": {"type": "string"},
                    "applicability": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }

    STRATEGY_SCHEMA = {
        "type": "object",
        "required": ["id", "name", "description", "steps", "applicability", "confidence"],
        "properties": {
            "id": {
                "type": "string",
                "pattern": "^strategy_[0-9]+$"  # Enforce format strategy_N
            },
            "name": {"type": "string"},
            "description": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["primitive", "params"],
                    "properties": {
                        "primitive": {"type": "string"},
                        "params": {"type": "object"}
                    }
                }
            },
            "applicability": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        }
    }

    COMBINED_PRIMITIVE_SCHEMA = {
        "type": "object",
        "required": ["combined_primitive"],
        "properties": {
            "combined_primitive": {
                "type": "object",
                "required": ["id", "name", "description", "parameters", "implementation_guide", "applicability", "examples", "components"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    },
                    "implementation_guide": {"type": "string"},
                    "applicability": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "components": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }

    ADAPTED_PRIMITIVE_SCHEMA = {
        "type": "object",
        "required": ["adapted_primitive"],
        "properties": {
            "adapted_primitive": {
                "type": "object",
                "required": ["id", "name", "description", "parameters", "implementation_guide", "applicability", "examples", "adaptations"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    },
                    "implementation_guide": {"type": "string"},
                    "applicability": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "adaptations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }

    PRIMITIVE_SUGGESTIONS_SCHEMA = {
        "type": "object",
        "required": ["suggestions"],
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["primitive_id", "confidence", "rationale"],
                    "properties": {
                        "primitive_id": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "rationale": {"type": "string"}
                    }
                }
            }
        }
    }

    PRIMITIVE_SUGGESTION_SCHEMA = {
        "type": "object",
        "required": ["suggestions"],
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["primitive_id", "confidence", "rationale"],
                    "properties": {
                        "primitive_id": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "rationale": {"type": "string"}
                    }
                }
            }
        }
    }

    TASK_SCHEMA = {
        "type": "object",
        "required": ["input", "output"],
        "properties": {
            "input": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "output": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            }
        }
    }

    LEARNING_RESULT_SCHEMA = {
        "type": "object",
        "required": ["learned_items"],
        "properties": {
            "learned_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "item"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["concept", "primitive"]
                        },
                        "item": {
                            "type": "object",
                            "required": ["id", "name", "description"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1
                                }
                            }
                        }
                    }
                }
            },
            "solution": {
                "type": "object",
                "properties": {
                    "success_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["primitive", "params"],
                            "properties": {
                                "primitive": {"type": "string"},
                                "params": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "feedback": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "content": {"type": "string"}
                }
            }
        }
    }

    PATTERN_SCHEMA = {
        "type": "object",
        "required": ["dimensions", "unique_values", "row_patterns", "col_patterns", "region_patterns"],
        "properties": {
            "dimensions": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2
            },
            "unique_values": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "row_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["values", "sum", "runs"],
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "sum": {"type": "integer"},
                        "runs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["value", "length"],
                                "properties": {
                                    "value": {"type": "integer"},
                                    "length": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
            },
            "col_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["values", "sum", "runs"],
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "sum": {"type": "integer"},
                        "runs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["value", "length"],
                                "properties": {
                                    "value": {"type": "integer"},
                                    "length": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
            },
            "region_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["position", "values", "sum"],
                    "properties": {
                        "position": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "sum": {"type": "integer"}
                    }
                }
            }
        }
    }

    CONTEXT_SCHEMA = {
        "type": "object",
        "required": ["input_shape", "input_values", "patterns", "constraints"],
        "properties": {
            "input_shape": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2
            },
            "input_values": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "patterns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "constraints": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
        }
    }

    PRIMITIVE_OUTPUT_SCHEMA = {
        "type": "object",
        "required": ["output"],
        "properties": {
            "output": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            }
        }
    }
