"""A decoder-only Transformer, including the gradients used to train it."""


from dataclasses import dataclass
from math import sqrt, pi

import numpy as np


@dataclass(frozen=True)
class Config:
    vocab: int
    context: int = 48
    width: int = 48
    heads: int = 4
    layers: int = 2

    def __post_init__(self):
        if min(self.vocab, self.context, self.width, self.heads, self.layers) < 1 or self.width % self.heads:
            raise ValueError("Use positive dimensions and a width divisible by the number of heads.")


def softmax(x):
    exps = np.exp(x - x.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)


def layer_norm(x, scale, bias):
    centered = x - x.mean(axis=-1, keepdims=True)
    inv_std = 1 / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    normalized = centered * inv_std
    return normalized * scale + bias, (normalized, inv_std)


def norm_backward(upstream, cache, scale):
    normalized, inv_std = cache
    g = upstream * scale
    dx = inv_std * (g - g.mean(-1, keepdims=True) - normalized * (g * normalized).mean(-1, keepdims=True))
    return dx, (upstream * normalized).sum((0, 1)), upstream.sum((0, 1))


def gelu(x):
    t = np.tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t), t


def project_grad(inputs, upstream):
    return inputs.reshape(-1, inputs.shape[-1]).T @ upstream.reshape(-1, upstream.shape[-1])


class MiniGPT:
    def __init__(self, config, seed=17, dtype=np.float32):
        self.config = config
        rng = np.random.default_rng(seed)
        self.weights = w = {}
        width = config.width

        def matrix(a, b, std=0.02):
            return rng.normal(0, std, (a, b)).astype(dtype)

        def norm(name):
            w[name + ".scale"] = np.ones(width, dtype)
            w[name + ".bias"] = np.zeros(width, dtype)

        w["token"] = matrix(config.vocab, width)
        w["position"] = matrix(config.context, width)
        for layer in range(config.layers):
            k = f"block{layer}."
            norm(k + "attn_norm")
            norm(k + "ff_norm")
            w[k + "qkv"] = matrix(width, 3 * width)
            w[k + "out"] = matrix(width, width, 0.02 / np.sqrt(2 * config.layers))
            w[k + "expand"] = matrix(width, 4 * width)
            w[k + "contract"] = matrix(4 * width, width, 0.02 / np.sqrt(2 * config.layers))
        norm("final")

    @property
    def parameter_count(self):
        return sum(value.size for value in self.weights.values())

    def _split_heads(self, x):
        batch, length, width = x.shape
        return x.reshape(batch, length, self.config.heads, width // self.config.heads).transpose(0, 2, 1, 3)

    @staticmethod
    def _join_heads(x):
        batch, heads, length, depth = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * depth)

    def forward(self, ids, *, with_cache=False):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context:
            raise ValueError("Expected ids with shape [batch, 1..context].")
        if ids.shape[0] < 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Token IDs must be a nonempty integer array.")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab):
            raise ValueError("Token IDs are outside the vocabulary.")
        w = self.weights
        length = ids.shape[1]
        x = w["token"][ids] + w["position"][None, :length]
        future = np.triu(np.ones((length, length), dtype=bool), k=1)
        blocks = []
        for layer in range(self.config.layers):
            k = f"block{layer}."
            h, norm1 = layer_norm(x, w[k + "attn_norm.scale"], w[k + "attn_norm.bias"])
            q, key, v = (self._split_heads(a) for a in np.split(h @ w[k + "qkv"], 3, axis=-1))
            scores = (q @ key.swapaxes(-1, -2)) / sqrt(q.shape[-1])
            attention = softmax(np.where(future, -np.inf, scores))
            mixed = self._join_heads(attention @ v)
            residual = x + mixed @ w[k + "out"]
            f, norm2 = layer_norm(residual, w[k + "ff_norm.scale"], w[k + "ff_norm.bias"])
            z = f @ w[k + "expand"]
            activated, tanh_z = gelu(z)
            x = residual + activated @ w[k + "contract"]
            if with_cache:
                blocks.append((h, norm1, q, key, v, attention, mixed, f, norm2, z, tanh_z, activated))
        y, final_norm = layer_norm(x, w["final.scale"], w["final.bias"])
        logits = y @ w["token"].T  # input embedding and vocabulary projection share weights
        if with_cache:
            return logits, (ids, blocks, y, final_norm)
        return logits

    def backward(self, dlogits, cache):
        ids, blocks, y, final_norm = cache
        w, grads = self.weights, {}
        grads["token"] = project_grad(dlogits, y)
        dx, grads["final.scale"], grads["final.bias"] = norm_backward(dlogits @ w["token"], final_norm, w["final.scale"])
        for layer in reversed(range(self.config.layers)):
            k = f"block{layer}."
            h, norm1, q, key, v, attn, mixed, f, norm2, z, tanh_z, activated = blocks[layer]
            grads[k + "contract"] = project_grad(activated, dx)
            dact = dx @ w[k + "contract"].T
            dgelu = 0.5 * (1 + tanh_z) + 0.5 * z * (1 - tanh_z ** 2) * sqrt(2 / pi) * (1 + 3 * 0.044715 * z ** 2)
            dz = dact * dgelu
            grads[k + "expand"] = project_grad(f, dz)
            dnorm, grads[k + "ff_norm.scale"], grads[k + "ff_norm.bias"] = norm_backward(
                dz @ w[k + "expand"].T, norm2, w[k + "ff_norm.scale"]
            )
            dres = dx + dnorm
            grads[k + "out"] = project_grad(mixed, dres)
            dmix = self._split_heads(dres @ w[k + "out"].T)
            dv = attn.swapaxes(-1, -2) @ dmix
            datt = dmix @ v.swapaxes(-1, -2)
            ds = attn * (datt - (datt * attn).sum(axis=-1, keepdims=True))
            ds /= sqrt(q.shape[-1])
            dq, dk = ds @ key, ds.swapaxes(-1, -2) @ q
            dqkv = np.concatenate([self._join_heads(a) for a in (dq, dk, dv)], axis=-1)
            grads[k + "qkv"] = project_grad(h, dqkv)
            dnorm, grads[k + "attn_norm.scale"], grads[k + "attn_norm.bias"] = norm_backward(
                dqkv @ w[k + "qkv"].T, norm1, w[k + "attn_norm.scale"]
            )
            dx = dres + dnorm
        grads["position"] = np.zeros_like(w["position"])
        grads["position"][:ids.shape[1]] = dx.sum(axis=0)
        np.add.at(grads["token"], ids, dx)
        return grads


def cross_entropy(logits, targets, *, gradient=False):
    """Mean negative log probability over labeled tokens. A target of -1 is padding."""
    flat = logits.reshape(-1, logits.shape[-1])
    labels = targets.reshape(-1)
    valid = labels >= 0
    if not np.any(valid):
        raise ValueError("At least one next token is required.")
    shifted = flat - flat.max(-1, keepdims=True)
    exp = np.exp(shifted)
    total = exp.sum(-1, keepdims=True)
    rows = np.flatnonzero(valid)
    loss = float((np.log(total[rows, 0]) - shifted[rows, labels[rows]]).mean())
    if not gradient:
        return loss
    grad = exp / total
    grad[rows, labels[rows]] -= 1
    grad[~valid] = 0
    grad /= len(rows)
    return loss, grad.reshape(logits.shape)
