"""Phase 2 correctness gates.

Run with:  python -m pytest tests/test_phase2.py -v

These are the checks that must pass BEFORE any GPU time is spent. The
causal-masking tests in particular are a hard spec requirement (2.1
stage 2) and catch the two bugs that silently produce a model which
appears to train while having leaked the future.
"""
from __future__ import annotations

import ast
import io
import sys
import tempfile
import tokenize
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.eval import verify_causal_masking, verify_mask_structure   # noqa: E402
from shared.model import DecoderLM, GPTConfig, TrainConfig             # noqa: E402
from shared.train import TokenDataset, Trainer                         # noqa: E402


def tiny_config(**kw) -> GPTConfig:
    base = dict(vocab_size=500, n_layer=2, n_head=4, d_model=64,
                d_ff=256, block_size=32)
    base.update(kw)
    return GPTConfig(**base)


# --------------------------------------------------------------- architecture

def test_parameter_count_is_on_target():
    """The Phase 2 configuration must hit ~25M params exactly."""
    model = DecoderLM(GPTConfig())
    assert model.count_parameters() == 25_321_472


def test_parameter_breakdown_sums_to_total():
    b = DecoderLM(GPTConfig()).parameter_breakdown()
    assert (b["token_embedding"] + b["positional_embedding"]
            + b["blocks_total"] + b["final_layernorm"]
            + b["lm_head"]) == b["total"]


def test_weight_tying_shares_one_matrix():
    m = DecoderLM(GPTConfig(tie_weights=True))
    assert m.lm_head.weight is m.token_embedding.weight
    untied = DecoderLM(GPTConfig(tie_weights=False))
    saved = untied.count_parameters() - m.count_parameters()
    assert saved == GPTConfig().vocab_size * GPTConfig().d_model == 6_144_000


def test_config_rejects_indivisible_head_dim():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(d_model=100, n_head=8)


def test_forward_shapes():
    cfg = tiny_config()
    out = DecoderLM(cfg)(torch.randint(0, cfg.vocab_size, (3, 16)))
    assert out["logits"].shape == (3, 16, cfg.vocab_size)


def test_loss_starts_near_uniform_entropy():
    """An untrained model should sit at ln(vocab_size) on random targets.

    With init_std=0.02 and a tied head the initial logits are near zero, so
    the predicted distribution is near-uniform and cross-entropy against
    UNRELATED targets is ~ln(V). Targets must be independent of the input:
    using the input as its own target asks the model to copy rather than
    predict, which is an easier problem and scores below ln(V).
    """
    cfg = tiny_config()
    m = DecoderLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    loss = m(x, y)["loss"].item()
    assert abs(loss - np.log(cfg.vocab_size)) < 0.3


def test_sequence_longer_than_block_size_is_rejected():
    cfg = tiny_config()
    with pytest.raises(ValueError, match="exceeds block_size"):
        DecoderLM(cfg)(torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1)))


# ------------------------------------------------------------- causal masking

def test_causal_masking_perturbation_sweep():
    """Changing token t+1 must not change any logit at position <= t.

    Spec 2.1 stage 2, verbatim requirement.
    """
    m = DecoderLM(tiny_config())
    r = verify_causal_masking(m, seq_len=32, device="cpu")
    assert r["passed"], r
    assert r["max_abs_delta_past"] == 0.0        # bit-identical, not merely close
    assert r["all_perturbed_positions_changed"]  # test is not vacuous


def test_attention_matrix_is_lower_triangular():
    r = verify_mask_structure(DecoderLM(tiny_config()), seq_len=16)
    assert r["passed"]
    assert r["max_future_attention_weight"] == 0.0


def test_attention_rows_sum_to_one():
    cfg = tiny_config()
    m = DecoderLM(cfg).eval()
    attns = m(torch.randint(0, cfg.vocab_size, (1, 16)),
              return_attn=True)["attentions"]
    assert len(attns) == cfg.n_layer
    for a in attns:
        assert a.shape == (1, cfg.n_head, 16, 16)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)


# ------------------------------------------------------------------ optimizer

def test_overfit_single_batch():
    """Loss must reach ~0 on one batch: proves loss/backward/optimizer work."""
    cfg = tiny_config(dropout=0.0, attn_dropout=0.0, embd_dropout=0.0)
    m = DecoderLM(cfg)
    opt = m.configure_optimizers(0.0, 3e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))
    for _ in range(300):
        loss = m(x, y)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.05


def test_weight_decay_excludes_norms_and_biases():
    m = DecoderLM(tiny_config())
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay, no_decay = opt.param_groups[0], opt.param_groups[1]
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


# ------------------------------------------------------------------ positional

def test_learned_positional_caps_sequence_length():
    """Learned absolute embeddings have no row past block_size."""
    from shared.model import LearnedPositional
    pe = LearnedPositional(16, 8)
    with pytest.raises(ValueError, match="block_size"):
        pe(torch.arange(17))


def test_no_positional_is_permutation_equivariant_without_the_mask():
    """Attention itself is permutation equivariant when positions are removed.

    This is the premise the ablation rests on, but it must be tested on
    UNMASKED attention. In the full model the causal mask breaks the
    symmetry -- position i attends only to positions <= i, so permuting a
    prefix genuinely changes what the intermediate positions see. The
    ablation's real effect is therefore not "the model becomes permutation
    invariant" but "the only order information left is whatever the causal
    mask implies", which is far weaker than explicit positions.
    """
    from shared.model import MultiHeadCausalSelfAttention
    torch.manual_seed(0)
    attn = MultiHeadCausalSelfAttention(d_model=32, n_head=4, block_size=8,
                                        dropout=0.0, attn_dropout=0.0).eval()
    # disable the mask to isolate the permutation property of attention
    attn.causal_mask.fill_(True)

    x = torch.randn(1, 6, 32)
    perm = torch.randperm(6)
    a = attn(x)[0][:, perm]        # attend, then permute
    b = attn(x[:, perm])[0]        # permute, then attend
    assert torch.allclose(a, b, atol=1e-5)


def test_no_positional_still_removes_the_positional_signal():
    """With positional='none' the position input contributes exactly zero."""
    cfg = tiny_config(positional="none")
    m = DecoderLM(cfg)
    pos = m.positional(torch.arange(cfg.block_size))
    assert torch.count_nonzero(pos) == 0


# --------------------------------------------------------------------- dataset

@pytest.fixture
def bin_file(tmp_path):
    p = tmp_path / "toy.bin"
    np.arange(10_000, dtype=np.uint32).tofile(p)
    return str(p)


def test_dataset_targets_are_inputs_shifted_by_one(bin_file):
    ds = TokenDataset(bin_file, block_size=16)
    x, y = ds.get_batch(4, "cpu")
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == torch.int64          # embedding index type
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_deterministic_batches_are_reproducible(bin_file):
    ds = TokenDataset(bin_file, block_size=16)
    a = list(ds.get_deterministic_batches(2, 3, "cpu", seed=7))
    b = list(ds.get_deterministic_batches(2, 3, "cpu", seed=7))
    for (x1, _), (x2, _) in zip(a, b):
        assert torch.equal(x1, x2)


# ------------------------------------------------------------------ scheduler

def test_lr_schedule_warms_up_then_decays():
    from shared.train import CosineWarmupScheduler
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    s = CosineWarmupScheduler(opt, warmup_steps=10, max_steps=100,
                              max_lr=1e-3, min_lr=1e-4)
    assert s.get_lr(0) < s.get_lr(5) < s.get_lr(9)      # warmup rises
    assert s.get_lr(9) <= s.get_lr(10)                  # peak at end of warmup
    assert s.get_lr(50) > s.get_lr(90)                  # then decays
    assert abs(s.get_lr(100) - 1e-4) < 1e-9             # floors at min_lr


def test_lr_is_a_pure_function_of_step():
    """Exact resume depends on LR(step) not being path-dependent."""
    from shared.train import CosineWarmupScheduler
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    s = CosineWarmupScheduler(opt, 10, 100, 1e-3, 1e-4)
    direct = s.get_lr(57)
    for i in range(58):
        s.step(i)
    assert s.get_lr(57) == direct


# --------------------------------------------------------- checkpoint / resume

def _toy_bins(d: Path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 200_000), ("validation", 20_000)):
        np.asarray(rng.integers(0, 500, n), dtype=np.uint32).tofile(d / f"{split}.bin")


def _mk_trainer(d: Path, ckpt: str, max_steps: int):
    torch.manual_seed(42)
    mc = tiny_config()
    tc = TrainConfig(max_steps=max_steps, micro_batch_size=4, grad_accum_steps=2,
                     eval_interval=10 ** 9, eval_batches=2,
                     checkpoint_interval=10 ** 9, log_interval=10 ** 9,
                     device="cpu", amp=False, learning_rate=1e-3)
    return Trainer(DecoderLM(mc),
                   TokenDataset(str(d / "train.bin"), mc.block_size),
                   TokenDataset(str(d / "validation.bin"), mc.block_size),
                   mc, tc, ckpt, str(d / "logs"), language="test")


def test_resume_is_bit_exact(tmp_path):
    """Uninterrupted training == train, kill, resume.

    Spec 2.2 makes checkpoint resume mandatory ("A run that cannot resume
    after interruption will lose marks"). max_steps is held constant across
    all three runs, because it defines the cosine schedule -- shortening it
    would legitimately change the LR at every step and is not what a real
    interruption does.
    """
    _toy_bins(tmp_path)
    MAX = 40

    torch.manual_seed(7)
    a = _mk_trainer(tmp_path, str(tmp_path / "ck_a"), MAX)
    a.train(resume=False)
    ref = {k: v.clone() for k, v in a.model.state_dict().items()}

    torch.manual_seed(7)
    b = _mk_trainer(tmp_path, str(tmp_path / "ck_b"), MAX)
    b.tcfg.max_steps = 20                 # simulate the session dying at step 20
    b.scheduler.max_steps = MAX           # the real schedule is unchanged
    b.train(resume=False)

    torch.manual_seed(7)
    c = _mk_trainer(tmp_path, str(tmp_path / "ck_b"), MAX)
    c.train(resume=True)

    assert c.step == a.step == MAX
    assert c.tokens_seen == a.tokens_seen
    delta = max(float((ref[k] - c.model.state_dict()[k]).abs().max()) for k in ref)
    assert delta == 0.0, f"resume diverged by {delta:.3e}"


def test_checkpoint_contains_every_required_field(tmp_path):
    """Spec 2.2 enumerates exactly what a checkpoint must carry."""
    _toy_bins(tmp_path)
    t = _mk_trainer(tmp_path, str(tmp_path / "ck"), 4)
    t.train(resume=False)
    ck = torch.load(tmp_path / "ck" / "latest.pt", map_location="cpu",
                    weights_only=False)
    for field in ("model_state_dict", "optimizer_state_dict",
                  "scheduler_state_dict", "scaler_state_dict",
                  "step", "config"):
        assert field in ck, f"checkpoint missing {field}"
    assert "model" in ck["config"] and "training" in ck["config"]
    assert (tmp_path / "ck" / "latest.config.json").exists()


# ------------------------------------------------- shared/ stays language-free

def _code_without_comments_or_docstrings(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type != tokenize.COMMENT]
    tree = ast.parse(tokenize.untokenize(toks))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree).lower()


def test_shared_code_never_names_a_language():
    """`shared/` must stay language-agnostic so H and L cannot couple.

    Comments and docstrings legitimately name the calling scripts, so only
    executable code is scanned.
    """
    root = Path(__file__).resolve().parents[1] / "shared"
    offenders = [
        (str(p.relative_to(root.parent)), lang)
        for p in sorted(root.rglob("*.py"))
        for lang in ("hindi", "nepali")
        if lang in _code_without_comments_or_docstrings(p)
    ]
    assert not offenders, f"language-specific code in shared/: {offenders}"
