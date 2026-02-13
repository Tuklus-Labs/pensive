"""
Pattern registry for spreading activation entity extraction.

Provides categorized pattern sets for different data domains:
- BASE_PATTERNS: Universal patterns (dates, URLs, paths, metrics, versions)
- NL_PATTERNS: Natural language patterns (projects, tech, models, hardware, people)
- INFRA_PATTERNS: Synthetic benchmark patterns (rooms, servers, companies - backward compat)

Usage:
    from AEGIS.Pensive.spreading_activation.patterns import REAL_DATA_PATTERNS, SYNTHETIC_PATTERNS

    # For natural language / real-world data:
    sa = SpreadingActivation(patterns=REAL_DATA_PATTERNS)

    # For synthetic benchmarks (backward compatible with original PATTERNS):
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
"""

from typing import List, Tuple

# EntityPattern: (regex, entity_type, case_insensitive)
EntityPattern = Tuple[str, str, bool]

# ---------------------------------------------------------------------------
# BASE_PATTERNS - Universal patterns for any data source
# ---------------------------------------------------------------------------
BASE_PATTERNS: List[EntityPattern] = [
    # Dates
    (r'\b(\d{4}-\d{2}-\d{2})\b', 'date', True),
    (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4})\b', 'date', True),

    # URLs
    (r'(https?://[^\s<>"{}|\\^\`\[\]]+)', 'url', True),

    # File paths
    (r'(/(?:home|usr|var|etc|opt|tmp)/[^\s<>"]+)', 'filepath', True),
    (r'(~/[^\s<>"]+)', 'filepath', True),

    # Metrics with units
    (r'\b(\d+(?:\.\d+)?(?:ms|MB|GB|TB|KB|%|fps|Hz|GHz|tok/s))\b', 'metric', True),

    # Money
    (r'\$(\d+(?:,\d{3})*(?:\.\d+)?[MBK]?)', 'money', True),

    # Version numbers
    (r'\b(v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?)\b', 'version', True),
]

# ---------------------------------------------------------------------------
# NL_PATTERNS - Natural language / real-world data patterns
# ---------------------------------------------------------------------------
NL_PATTERNS: List[EntityPattern] = [
    # Gary's projects
    (r'\b(aegis|pensive|kairos|panoptes|lenora|hermes|kesagake|mud[\s-]?puppy|flywheel)\b', 'project', True),

    # Tech terms
    (r'\b(rocm|pytorch|triton|faiss|cuda|hip|vulkan|opencl|tensorflow|jax|numpy|pandas|sqlalchemy|flask|fastapi|django|redis|nginx|docker|kubernetes)\b', 'tech', True),

    # Concepts
    (r'\b(flash\s+attention|spreading\s+activation|kv[\s_-]?cache|vector[\s_-]?search|knowledge\s+graph|fine[\s-]?tun(?:e|ing)|quantiz(?:e|ation)|inference|embedding|tokenizer|transformer)\b', 'concept', True),

    # LLM tools
    (r'\b(llama\.?cpp|exllama(?:v2)?|vllm|ollama|lm[\s_-]?studio|stable[\s_-]?diffusion|midjourney|dall[\s-]?e|comfyui)\b', 'tool', True),

    # Hardware
    (r'\b(7900\s*xtx|mi300x|a100|h100|3090|4090|9900x|epyc|instinct)\b', 'hardware', True),

    # Models - GPT family
    (r'\b(gpt-?(?:oss-?)?\d+(?:\.\d+)?[a-z]*(?:-(?:turbo|mini|preview|o))?)\b', 'model', True),
    # Models - Claude family
    (r'\b(claude[\s-]?(?:opus|sonnet|haiku)?[\s-]?\d*(?:\.\d+)?)\b', 'model', True),
    # Models - Llama family
    (r'\b(llama[\s-]?\d+(?:\.\d+)?(?:[\s-]?\d+[bB])?)\b', 'model', True),
    # Models - Qwen
    (r'\b(qwen\d?[\s-]?\d+[bBmM]?)\b', 'model', True),
    # Models - Mistral
    (r'\b(mistral[\s-]?\d*(?:\.\d+)?(?:[\s-]?\d+[bBmM])?)\b', 'model', True),
    # Models - Gemini
    (r'\b(gemini[\s-]?(?:pro|ultra|nano|flash)?[\s-]?\d*(?:\.\d+)?)\b', 'model', True),
    # Models - Phi
    (r'\b(phi[\s-]?\d+)\b', 'model', True),

    # Model sizes
    (r'\b(\d+[bBmM])\b', 'model_size', True),

    # Performance changes (e.g. "45% -> 72%", "120 --> 212tok/s")
    (r'(\d+(?:\.\d+)?%?\s*(?:->|-->|\u2192|to)\s*\d+(?:\.\d+)?%?)', 'perf_change', True),

    # People - "Firstname Lastname" with at least one 4+ char component
    # Disabled: generates 111K noise entities from ChatGPT markdown formatting.
    # Use GENERAL_PATTERNS person pattern instead (requires both names 3+ chars,
    # negative lookahead for common non-name bigrams).
]

# ---------------------------------------------------------------------------
# GENERAL_PATTERNS - Domain-agnostic entity patterns
# Works on any English text without hardcoded vocabularies.
# Designed to supplement NL_PATTERNS or INFRA_PATTERNS.
# ---------------------------------------------------------------------------
GENERAL_PATTERNS: List[EntityPattern] = [
    # Locations: "City Office/HQ/Center/Lab/Hub" - match BEFORE person to avoid overlap
    (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:Office|HQ|Center|Lab|Hub|Campus|Annex|R&D))\b', 'location', False),

    # Client/Company names: "Word Industries/Dynamics/etc."
    (r'\b([A-Z][a-z]+\s+(?:Industries|Dynamics|Consulting|Systems|Solutions|Technologies|Analytics|Ventures|Partners|Group|Corp|Labs|Inc))\b', 'organization', False),

    # "Project Codename" - explicit prefix
    (r'Project\s+([A-Z][a-z]{2,})', 'project', False),

    # Person names: "Firstname Lastname" - requires both 3+ chars
    # First word: NOT a structural prefix (Project, The, New, etc.)
    # Second word: NOT a location/org/dept suffix
    (r'\b(?!(?:Project|The|New|Old|North|South|East|West|Upper|Lower|Greater|Central|Advanced|Global|Total|Chief|Senior|Junior|General)\s)([A-Z][a-z]{2,}\s+(?:O\')?(?!(?:Office|Center|Lab|Hub|Campus|Annex|Industries|Dynamics|Consulting|Systems|Solutions|Technologies|Analytics|Ventures|Partners|Group|Corp|Labs|Inc|Engineering|Operations|Services|Resources|Security|Marketing|Finance|Sales|Legal|Team|Initiative|Project)\b)[A-Z][a-z]{2,})\b', 'person', False),

    # Departments in parentheses or after pipe
    (r'\(([A-Z][A-Za-z\s]+?(?:Engineering|Operations|Analytics|Services|Resources|Security|Marketing|Finance|Sales|Legal|QA|DevOps|SRE))\)', 'department', False),
]

# ---------------------------------------------------------------------------
# INFRA_PATTERNS - Synthetic benchmark / infrastructure patterns
# (backward compatible with original PATTERNS from optimized_spread.py)
# ---------------------------------------------------------------------------
INFRA_PATTERNS: List[EntityPattern] = [
    # Room codes
    (r'\b([A-Z]-\d{2,4})\b', 'room', False),

    # Servers
    (r'\b(prod-[a-z0-9-]+|gateway-[a-z0-9-]+|staging-[a-z0-9-]+)\b', 'server', True),

    # Datacenters
    (r'\b(eu-[a-z]+-\d+|us-[a-z]+-\d+|ap-[a-z]+-\d+)\b', 'datacenter', True),

    # Codes (uppercase identifiers)
    (r'\b([A-Z]{2,}[A-Z0-9]{3,})\b', 'code', False),

    # Badges
    (r'badge\s+(?:number\s+)?(\d{4,6})', 'badge', True),

    # People - Employee/Dr prefixed
    (r'Employee\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'person', False),
    (r'Dr\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'person', False),

    # Projects
    (r'Project\s+([A-Z][a-z]+)', 'project', False),

    # Initiatives
    (r'the\s+([A-Z][a-z]+)\s+initiative', 'initiative', True),

    # Teams
    (r'the\s+([A-Z][a-z]+)\s+team', 'team', True),

    # Companies
    (r'\b([A-Z][a-z]+(?:Tech|Sys|Corp|Labs|AI))\b', 'company', False),
    (r'\b([A-Z]{2,}[a-z]+)\b', 'company', False),
    (r'\b([A-Z][a-z]+[A-Z][a-z]*)\b', 'company', False),

    # Generic proper nouns (catch-all, kept for backward compat with synthetic benchmarks)
    (r'\b([A-Z][a-z]{3,})\b', 'proper', False),
]


def build_pattern_set(*pattern_lists: List[EntityPattern]) -> List[EntityPattern]:
    """
    Merge multiple pattern lists, deduplicating by regex string.

    When the same regex appears in multiple lists, the first occurrence wins.
    This preserves ordering within each list and across lists.

    Args:
        *pattern_lists: Variable number of pattern lists to merge.

    Returns:
        A deduplicated list of EntityPattern tuples.
    """
    seen_regexes: set = set()
    merged: List[EntityPattern] = []

    for pattern_list in pattern_lists:
        for pattern in pattern_list:
            regex = pattern[0]
            if regex not in seen_regexes:
                seen_regexes.add(regex)
                merged.append(pattern)

    return merged


# ---------------------------------------------------------------------------
# Convenience pattern sets
# ---------------------------------------------------------------------------

# For natural language sources (real conversations, notes, docs)
REAL_DATA_PATTERNS: List[EntityPattern] = build_pattern_set(BASE_PATTERNS, NL_PATTERNS, GENERAL_PATTERNS)

# For synthetic benchmarks (backward compat - includes the old `proper` catch-all)
SYNTHETIC_PATTERNS: List[EntityPattern] = build_pattern_set(BASE_PATTERNS, INFRA_PATTERNS)

# Everything combined
ALL_PATTERNS: List[EntityPattern] = build_pattern_set(BASE_PATTERNS, NL_PATTERNS, GENERAL_PATTERNS, INFRA_PATTERNS)
