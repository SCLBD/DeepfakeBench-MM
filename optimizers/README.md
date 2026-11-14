# Optimizers

[choose_optimizer](builder.py) builds an optimizer from a parameter iterable and
configuration. It filters out parameters whose `requires_grad` is false and
forwards `params` to the chosen constructor.

| Case-sensitive `type` | Constructor |
|---|---|
| `Adam` | `torch.optim.Adam` |
| `AdamW` | `torch.optim.AdamW` |
| `AdamP` | `adamp.AdamP` |
| `SGD` | `torch.optim.SGD` |
| `RMSprop` | `torch.optim.RMSprop` |

```python
optimizer = choose_optimizer(
    model.parameters(),
    {'type': 'AdamW', 'params': {'lr': 0.0001}},
)
```

[BasicTrainer](../trainers/basic_trainer.py) supplies all model parameters.
[AlternativeTrainer](../trainers/alternative_trainer.py) builds a separate
optimizer for each stage using the detector's `get_stage_parameters(stage_name)`.
The detector defines parameter freezing and stage membership.

To add an optimizer, import it and add a `type` branch in `choose_optimizer`.
Keep its parameter/configuration interface and the state-dictionary methods used
by the trainers for resume.
