# `models/pretrained/` — fine-tuned lightweight SOTA backbones

**Owner: Member 4.** Assignment §5 — **20 marks**.
**Notebook:** [`05_pretrained_finetuning`](../../../../notebooks/05_pretrained_finetuning.ipynb).

## Inputs

`num_classes` from the `DataBundle` — built from the **same split** as the custom models, which the
assignment requires. As the builder's `**overrides`, the matching entry of
`pretrained.backbones` in [`configs/stages/pretrained.yaml`](../../../../configs/stages/pretrained.yaml)
(`torchvision_name`, `weights`, `finetune_strategy`, `unfreeze_last_n`, `replace_classifier`,
`dropout`, `backbone_lr_scale`), plus `pretrained.input_resolution`. `build_model_from_config` finds
the entry and passes it in.

## Outputs

Two registry entries, `mobilenet_v2` and `squeezenet1_1`, in the **same registry** as Model A and
Model B. Member 3's `Trainer` and Member 1's benchmark therefore need no special case for them — which
is exactly what makes the §6 comparison fair.

`param_groups(model, base_lr, backbone_lr_scale)` splits the parameters so the new head trains at the
experiment's `optimizer.lr`, and the pretrained backbone at `lr × backbone_lr_scale`. The learning
rate keeps one home: the experiment file.

## Backbones

Both come from the list in the assignment itself. They save parameters by *different* mechanisms, and
§6 rewards comparing the mechanisms, not just the accuracy numbers:

- **MobileNetV2** — depthwise separable convolutions with inverted residuals.
- **SqueezeNet 1.1** — fire modules: a 1x1 "squeeze" that cuts the channel count, then a mixed
  1x1/3x3 "expand".

## Two traps

**Replacing the classifier.** The two backbones have different heads:

- MobileNetV2: `classifier[1]` is `Linear(1280, 1000)`.
- SqueezeNet: `classifier[1]` is `Conv2d(512, 1000, 1)`, **not** a Linear layer, followed by global
  average pooling.

If you forget to replace MobileNetV2's head, the model trains without error and reports nonsense: the
loss is computed over 1000 logits, and only 10 of them can ever be right. Substituting a `Linear`
into SqueezeNet fails on shape — that one, at least, fails loudly.

**Discriminative learning rates.** A head-sized learning rate applied to the backbone destroys the
pretrained features within a few hundred steps. The symptom is the classic fine-tuning failure:
validation accuracy peaks at epoch 1, then falls.

## The 64x64 decision

Fine-tuning happens at **64x64**, not at the backbones' native 224. This keeps the memory and compute
comparison honest: §6 is about cost at the resolution the sensor actually produces.

The handicap is real, and it is a **finding, not a flaw**. MobileNetV2 has a total stride of 32, so a
64x64 input reaches the classifier as a 2x2 feature map, where the architecture expects 7x7. A fair
report says so rather than claiming a clean win for the custom model. Set
`pretrained.input_resolution: 128` to run the ablation.

## For the report

Record **both** `trainable_params` and `total_params` — notebook `05` prints them. They differ
whenever layers are frozen, and "2.2M parameters" means little without knowing how many of them were
trained. Also state which freezing strategy you used, and that pretrained runs use ImageNet
normalisation statistics.
