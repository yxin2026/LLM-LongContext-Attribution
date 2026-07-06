from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apbs_rope_patch import APBSConfig, apbs_position_weight, ntk_base


def main() -> None:
    positions = torch.tensor([[0, 4915, 6553, 8192, 9830, 16383]])
    weights = apbs_position_weight(positions, 16384).tolist()[0]
    print("weights:", [round(x, 3) for x in weights])
    print("ntk_base:", round(ntk_base(10000.0, 16384, 4096, 128), 3))
    cfg = APBSConfig(method="apbs", gamma=0.3, target_length=16384)
    print("config:", cfg)


if __name__ == "__main__":
    main()

