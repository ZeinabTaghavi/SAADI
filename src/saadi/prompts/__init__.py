"""Pluggable prompt profiles.

Select a profile wherever prompts are configured: ``--prompt-profile`` in the
example, ``ingest.prompt_profile`` in an experiment configuration, or
:func:`load_prompt_profile` from Python. A profile is a folder of guidance
files; the library adds the fixed output contract, so a custom profile changes
what counts as an entity without being able to break parsing. Validate one
with ``saadi check-prompts my_profile/``.
"""

from .check import PromptCheckReport, check_prompt_profile, format_report
from .contract import (
    CONTRACT_VERSION,
    compose_document_summary_prompt,
    compose_entity_system_prompt,
    compose_summary_system_prompt,
)
from .profiles import (
    BUILTIN_PROFILES,
    PROFILE_ALIASES,
    PromptProfile,
    apply_profile_to_strategy_kwargs,
    builtin_profile_names,
    builtin_profiles_root,
    load_prompt_profile,
    profile_cache_suffix,
    resolve_profile_directory,
)

__all__ = [
    "BUILTIN_PROFILES",
    "CONTRACT_VERSION",
    "PROFILE_ALIASES",
    "PromptCheckReport",
    "PromptProfile",
    "apply_profile_to_strategy_kwargs",
    "builtin_profile_names",
    "builtin_profiles_root",
    "check_prompt_profile",
    "compose_document_summary_prompt",
    "compose_entity_system_prompt",
    "compose_summary_system_prompt",
    "format_report",
    "load_prompt_profile",
    "profile_cache_suffix",
    "resolve_profile_directory",
]
