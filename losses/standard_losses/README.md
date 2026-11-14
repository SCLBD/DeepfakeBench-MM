# Standard losses

These wrappers expose common objectives through the loss registry. Their forward
signatures and reductions are documented in the source files.

| Class | Registry key | Purpose |
|---|---|---|
| [CrossEntropyLoss](cross_entropy.py) | `cross_entropy` | Classification loss with optional class-count weights |
| [MeanSquaredErrorLoss](mean_squared_error.py) | `mse` | MSE with configurable reduction |
| [WassersteinGeneratorLoss](wasserstein_gan.py) | `wasserstein_generator` | Negative mean fake score |
| [WassersteinDiscriminatorLoss](wasserstein_gan.py) | `wasserstein_discriminator` | Fake mean minus real mean |
| [WassersteinDiscriminatorLossGradientPenalty](wasserstein_gan.py) | `wasserstein_discriminator_gradient_penalty` | Discriminator objective with an interpolation gradient penalty |

Supplying both `real_num` and `fake_num` to cross-entropy enables class weights
and creates the weight tensor on CUDA. The gradient-penalty loss returns
`(total_loss, gradient_penalty)`, so its caller must handle both values.
