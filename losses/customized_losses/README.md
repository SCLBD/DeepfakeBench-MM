# Method-specific losses

These objectives are used by the corresponding audio/video detectors. The source
docstrings describe feature pairing, labels, reductions, and distributed behavior.

| Class | Registry key | Purpose |
|---|---|---|
| [AVFFContrastiveLoss](avff_losses.py) | `avff_contrastive` | Symmetric audio/video contrastive objective |
| [FRADECenterClusterLoss](frade_losses.py) | `frade_cc` | Real-feature attraction and bounded fake-feature repulsion |
| [FRADEContrastiveLoss](frade_losses.py) | `frade_cmc` | Label-dependent cross-modal cosine objective |
| [MCLJSDLoss](mcl_losses.py) | `mcl_jsd` | Sum of both directed KL-style terms |
| [MCLNCELoss](mcl_losses.py) | `mcl_nce` | Paired-feature objective using a batch-level softmax |

AVFF gathers features across ranks when distributed training is initialized.
`MCLJSDLoss` retains the method's name while computing the two directed terms
without a Jensen-Shannon mixture.

FRADE's center is an `nn.Parameter`. A loss held only in an ordinary Python
dictionary is not registered as a child module, so the detector controls whether
that parameter participates in optimization and checkpoints.
