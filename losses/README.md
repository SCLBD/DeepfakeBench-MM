# Loss functions

This package provides standard and method-specific loss modules. Importing
[__init__.py](__init__.py) registers them in [LOSSFUNC](../tools/registry.py).
Detectors select the inputs, combine loss terms, and return the optimization loss
as `losses['overall']`.

| Module | Contents |
|---|---|
| [abstract_loss.py](abstract_loss.py) | Shared loss interface |
| [standard_losses](standard_losses) | Cross-entropy, MSE, Wasserstein objectives, etc. |
| [customized_losses](customized_losses) | AVFF, FRADE, MCL objectives, etc. |

## Configuration

```yaml
model_config:
  losses:
    - name: CE
      func: cross_entropy
      weight: 1.0
      params: {}
```

`func` selects the registry key and `params` supplies constructor arguments.
`name` is the detector's lookup name, so use the names expected by that detector.
To reweight a loss term, set its `weight` in the detector configuration. The
detector's `get_losses` method decides how to apply the weight when combining
terms into `overall`. See each loss implementation for its forward signature
and reduction.

## Adding a loss

Implement an `nn.Module`, usually derived from `AbstractLoss`, decorate it with
`@LOSSFUNC.register_module(module_name='your_loss')`, and import it in this
package's initializer. Add the configuration entry and invoke it from the
relevant detector. Handle structured loss outputs explicitly.
