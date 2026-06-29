# Copyright 2026 Extended Systems Intelligence Corporation
# SPDX-License-Identifier: Apache-2.0
"""Register a 'nemotron' chat template in DeepSpec's TEMPLATE_REGISTRY.

REFERENCE SCAFFOLD — the special-token strings below are PLACEHOLDERS. Nemotron chat formats
vary by generation; set these to match YOUR checkpoint's actual template (inspect
``tokenizer.chat_template`` and the model card). The Llama-Nemotron line uses an
``<extra_id_*>`` role-marker convention and a ``detailed thinking on|off`` system directive to
toggle reasoning — adjust as needed.

Call ``register()`` once during setup, or paste an equivalent ``TEMPLATE_REGISTRY.register(...)``
block directly into ``deepspec/data/parser.py`` next to the built-in templates.
"""
import warnings

from deepspec.data.parser import ChatTemplate, TEMPLATE_REGISTRY

# PLACEHOLDER tokens — verify against your Nemotron checkpoint before any real run.
_PLACEHOLDER_MARKER = "<extra_id_1>"
NEMOTRON_TEMPLATE = ChatTemplate(
    assistant_header="<extra_id_1>Assistant\n",
    user_header="<extra_id_1>User\n",
    system_prompt="detailed thinking off",  # or None; match the mode you will serve
    end_of_turn_token="\n<extra_id_1>",
    assistant_loss_prefix=None,
)


def register(name: str = "nemotron", template: ChatTemplate = NEMOTRON_TEMPLATE) -> None:
    """Register the template, idempotently (safe to import more than once).

    Emits a warning if the still-placeholder template is being registered, so a real run can't
    silently train against fabricated chat tokens.
    """
    if _PLACEHOLDER_MARKER in (template.assistant_header or ""):
        warnings.warn(
            "Registering the PLACEHOLDER Nemotron chat template. Replace the token strings in "
            "scaffold/template.py with your checkpoint's real chat format before any real run.",
            stacklevel=2,
        )
    try:
        TEMPLATE_REGISTRY.get(name)
    except KeyError:
        TEMPLATE_REGISTRY.register(name, template)


register()
