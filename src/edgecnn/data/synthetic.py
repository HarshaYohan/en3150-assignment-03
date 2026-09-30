"""Synthetic fixture dataset - the Phase 0 unblocker.

Owner: Member 1.  **Build this before anything else in the data stage.**

Why it exists
-------------
The pipeline split puts Members 3 and 4 downstream of Members 1 and 2. Without
this file they wait. With it, they do not: it satisfies the full Seam 1
contract - same shapes, same dtypes, same manifest columns, same DataBundle -
using generated noise instead of satellite imagery.

So Member 3 can run a complete training loop in notebook 03 and debug their
checkpointing on day one, before EuroSAT has finished downloading anywhere.

It is deliberately tiny (a few hundred 64x64 images, 4 classes) so a full run
takes seconds on a CPU.

    Select it in any notebook with:

        MODE = "synthetic"

    which makes load_config(..., mode=MODE) set dataset.name = "synthetic".

The fixture is generated on demand into ``paths.SYNTHETIC_FIXTURE_DIR`` and is
git-ignored: it is deterministic, so regenerating is cheaper than committing.
Synthetic runs are never official, so nothing from them can reach ``results/``.
"""

from __future__ import annotations

from pathlib import Path


def make_synthetic_fixture(
    root: Path,
    *,
    num_classes: int = 4,
    images_per_class: int = 50,
    image_size: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> Path:
    """Generate a tiny, deterministic, contract-valid dataset on disk.

    Implementation notes:

    * Give each class a distinguishable statistical signature - a different
      per-channel mean plus noise is enough. A model that cannot fit this is
      genuinely broken, which makes the fixture a useful smoke test rather
      than just a shape check.
    * Write a real ``split_manifest.csv`` with the same four columns as the
      production one, and real ``split_meta.json`` / ``norm_stats.json``.
      Validate all three with ``edgecnn.contracts.schema`` before returning -
      if the fixture does not satisfy the contract, it is not a fixture.
    * Fully deterministic given ``seed``, so two members debugging the same
      failure see the same batches.

    Returns:
        The fixture root, containing ``images/`` and the three split files.
    """
    raise NotImplementedError(
        "Member 1: implement make_synthetic_fixture FIRST. "
        "Members 3 and 4 cannot start until this exists."
    )
