# Integrating the Nemotron extension into DeepSpec

These files extend a [DeepSpec](https://github.com/deepseek-ai/DeepSpec) checkout with a
Nemotron target. They are written against DeepSpec's real extension points but are a **reference
implementation — validate before production use.**

Assume `$DEEPSPEC` is your DeepSpec checkout.

## 1. Register the chat template

Either import `scaffold/template.py` once during setup, or paste its `ChatTemplate(...)` /
`TEMPLATE_REGISTRY.register("nemotron", ...)` block directly into
`$DEEPSPEC/deepspec/data/parser.py` next to the built-in `qwen` / `gemma4` templates.

> The token strings in `template.py` are **placeholders** — set them to your checkpoint's real
> chat format before any run.

## 2. Add the Nemotron draft modeling

Copy the modeling package into DeepSpec's modeling tree:

```
scaffold/modeling/__init__.py  ->  $DEEPSPEC/deepspec/modeling/dspark/nemotron/__init__.py
scaffold/modeling/config.py    ->  $DEEPSPEC/deepspec/modeling/dspark/nemotron/config.py
```

This reuses DeepSpec's `Qwen3DSparkModel` as the draft and provides a Nemotron-specific
`build_draft_config` that maps the target's transformer dimensions into a clean draft config.

## 3. Add the trainer class

Append `NemotronDSparkTrainer` from `scaffold/trainer.py` to
`$DEEPSPEC/deepspec/trainer/dspark_trainer.py` (or import it there), then export it from
`$DEEPSPEC/deepspec/trainer/__init__.py`:

```python
from .dspark_trainer import Gemma4DSparkTrainer, Qwen3DSparkTrainer, NemotronDSparkTrainer
# ...
__all__ = [..., "NemotronDSparkTrainer"]
```

If you keep the trainer as a separate module instead, make sure its relative imports
(`from .modeling.config import ...`) resolve from wherever you place it.

## 4. Add the config

```
configs/dspark_nemotron_nano.py  ->  $DEEPSPEC/config/dspark/dspark_nemotron_nano.py
```

Then set the `SET-FOR-YOUR-CHECKPOINT` fields (target model id, `target_layer_ids`,
`mask_token_id`).

## 5. Run

Point DeepSpec's data / train / eval scripts at the Nemotron config and a Nemotron target
cache dir. The wrappers in [`../scripts/`](../scripts/) show the overrides. Read
[`../GUIDE.md`](../GUIDE.md) and [`../NEMOTRON-H-NOTES.md`](../NEMOTRON-H-NOTES.md) first.

## Validation checklist

- [ ] `python scaffold/selftest.py` passes (no-GPU import + registration + config-build check).
- [ ] `target_layer_ids` are valid for your checkpoint's `num_hidden_layers`.
- [ ] Draft config dimensions (heads, kv-heads, head_dim, rope) match your Nemotron target.
- [ ] Chat-template tokens match `tokenizer.chat_template`.
- [ ] Target-cache forward runs with `use_cache=False` (see Nemotron-H notes).
- [ ] A short smoke run (tiny dataset, few steps) trains and evaluates before the full job.
