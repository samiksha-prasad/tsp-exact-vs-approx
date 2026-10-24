"""
Run a pretrained POMO model on our own TSP instances (Extension 4).

POMO: Kwon et al., "POMO: Policy Optimization with Multiple Optima for
Reinforcement Learning", NeurIPS 2020. https://github.com/yd-kwon/POMO

POMO's code and weights are not copied into this repository. They are
downloaded from the authors' GitHub repo at run time (see the notebook).
This file only wraps them so the model can be run on instances we choose,
instead of the random ones POMO's test script generates.

How the model is used:
  * POMO was trained on cities drawn uniformly from the unit square, so each
    instance is shifted and scaled into [0, 1]^2 (same scale on both axes,
    so the tour order is unaffected).
  * POMO decodes n tours in parallel, one starting from each city, and with
    8-fold augmentation (flips and rotations of the square) runs that 8
    times. The shortest of the 8n tours is returned.
  * The returned tour's length is computed by the caller with its own
    distance function (e.g. TSPLIB rounding), not by POMO.
"""

import os
import sys
import time

import torch

# Architecture of the published checkpoints (from POMO's test_n100.py).
MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}

# Published checkpoints: training size -> (folder, epoch)
CHECKPOINTS = {
    20: ("saved_tsp20_model", 510),
    50: ("saved_tsp50_model", 1000),
    100: ("saved_tsp100_model2_longTrain", 3100),
}


def _import_pomo(pomo_root):
    """Make POMO's TSP modules importable. pomo_root = path to the cloned repo."""
    tsp_dir = os.path.join(pomo_root, "NEW_py_ver", "TSP")
    for p in (os.path.join(tsp_dir, "POMO"), tsp_dir, os.path.join(pomo_root, "NEW_py_ver")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from TSPEnv import TSPEnv
    from TSPModel import TSPModel
    from TSProblemDef import augment_xy_data_by_8_fold
    return TSPEnv, TSPModel, augment_xy_data_by_8_fold


class PomoSolver:
    def __init__(self, pomo_root, train_size=100, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_default_device(self.device)  # POMO creates tensors without a device
        self.TSPEnv, TSPModel, self.augment = _import_pomo(pomo_root)

        folder, epoch = CHECKPOINTS[train_size]
        path = os.path.join(pomo_root, "NEW_py_ver", "TSP", "POMO", "result", folder,
                            f"checkpoint-{epoch}.pt")
        self.model = TSPModel(**MODEL_PARAMS)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.train_size = train_size

    @staticmethod
    def normalise(points):
        """Shift and scale points into the unit square, keeping the aspect ratio."""
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        scale = max(max(xs) - x0, max(ys) - y0) or 1.0
        return [((x - x0) / scale, (y - y0) / scale) for x, y in points]

    @torch.no_grad()
    def solve(self, points, augment=True):
        """Return (tour as a list of city indices, seconds taken)."""
        n = len(points)
        start = time.perf_counter()

        problems = torch.tensor([self.normalise(points)], dtype=torch.float32, device=self.device)
        aug = 8 if augment else 1
        if augment:
            problems = self.augment(problems)  # shape (8, n, 2)

        env = self.TSPEnv(problem_size=n, pomo_size=n)
        env.batch_size = aug
        env.problems = problems
        env.BATCH_IDX = torch.arange(aug)[:, None].expand(aug, n)
        env.POMO_IDX = torch.arange(n)[None, :].expand(aug, n)

        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = self.model(state)
            state, reward, done = env.step(selected)

        # reward = -(tour length in the normalised square), shape (aug, n)
        flat = reward.reshape(-1).argmax().item()
        a, p = divmod(flat, n)
        tour = env.selected_node_list[a, p].tolist()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return tour, time.perf_counter() - start
