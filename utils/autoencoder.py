import random
from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import trange


def top_k_activation(z, k: int):
    values, indices = jax.lax.top_k(z, k=k, axis=-1)
    threshold = jnp.min(values, axis=-1)
    condition = threshold[:, None] <= z
    return jnp.where(condition, z, 0)


def encode(params, x, k: int):
    z = x @ params["W_e"] + params["b_e"]
    return top_k_activation(z, k)


def decode(params, z):
    pred_x = z @ params["W_d"] + params["b_d"]
    return pred_x


def loss(params, x, k: int, sparsity_weight: float):
    # Inspired by https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py
    z = encode(params, x, k)
    pred_x = decode(params, z)
    # Normalized mean squared error
    reconstruction = jnp.mean(
        jnp.mean(jnp.square(x - pred_x), axis=-1) / jnp.mean(jnp.square(x), axis=-1)
    )
    # Normalized L1 norm
    sparsity = jnp.mean(
        jnp.sum(jnp.abs(z), axis=-1) / jnp.mean(jnp.linalg.norm(x, axis=-1))
    )
    return reconstruction + sparsity_weight * sparsity


def init_params(rng_key, n_latent: int, n_input: int):
    # I'm not tying parameters due to this paper: https://transformer-circuits.pub/2023/monosemantic-features/index.html
    # Kaiming Initialization for parameters.
    rng_key, subkey = jax.random.split(rng_key)
    W_e = jax.random.normal(subkey, (n_input, n_latent)) * jnp.sqrt(2 / n_input)
    rng_key, subkey = jax.random.split(rng_key)
    b_e = jax.random.normal(subkey, (n_latent)) * jnp.sqrt(2 / n_latent)
    rng_key, subkey = jax.random.split(rng_key)
    b_d = jax.random.normal(subkey, (n_input)) * jnp.sqrt(2 / n_latent)
    return dict(W_e=W_e, b_e=b_e, b_d=b_d, W_d=W_e.T)


def train_autoencoder(
    rng_key,
    params,
    optimizer,
    opt_state,
    x,
    n_latent: int,
    k: int,
    lr: float,
    sparsity_weight: float,
    batch_size: int,
    n_epochs: int,
    show_progress_bar=True,
):
    loss_history = []
    for i_epoch in trange(
        n_epochs, desc="Going through all epochs.", disable=not show_progress_bar
    ):
        epoch_loss = 0
        indices = jnp.arange(x.shape[0])
        rng_key, subkey = jax.random.split(rng_key)
        indices = jax.random.permutation(subkey, indices)
        for batch_start in trange(
            0,
            x.shape[0],
            batch_size,
            leave=False,
            desc="Going through all batches",
            disable=not show_progress_bar,
        ):
            batch_end = batch_start + batch_size
            batch_idx = indices[batch_start:batch_end]
            batch_x = x[batch_idx]
            loss_value, grads = jax.value_and_grad(loss)(
                params, x=batch_x, k=k, sparsity_weight=sparsity_weight
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss_value
        loss_history.append(float(epoch_loss))
        print(f"Epoch {i_epoch}: loss={epoch_loss:.2e}")
    return params, loss_history


def to_jax(params: dict):
    return {key: jnp.array(val) for key, val in params.items()}


class TopKAutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_latent: int = 32768,
        top_k: int = 16,
        lr: float = 1e-3,
        batch_size: int = 4096,
        n_epochs: int = 10,
        alpha: float = 0.03,
        show_progress_bar: bool = True,
        random_state: Optional[int] = None,
    ):
        self.random_state = random_state
        if random_state is None:
            self.rng_key = jax.random.key(random.randint(0, 10_000))
        else:
            self.rng_key = jax.random.key(random_state)
        self.n_latent = n_latent
        self.lr = lr
        self.alpha = alpha
        self.top_k = top_k
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.show_progress_bar = show_progress_bar

    def partial_fit(self, X, y=None, n_epochs: Optional[int] = 1):
        optimizer = optax.adamw(self.lr)
        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]
        if hasattr(self, "coef_"):
            params = to_jax(self._params)
        else:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            params = init_params(subkey, self.n_latent, self.n_features_in_)
        if not hasattr(self, "opt_state"):
            self.opt_state = optimizer.init(params)
        if not hasattr(self, "loss_curve_"):
            self.loss_curve_ = []
        self.rng_key, subkey = jax.random.split(self.rng_key)
        params, loss_curve = train_autoencoder(
            rng_key=subkey,
            params=params,
            optimizer=optimizer,
            opt_state=self.opt_state,
            x=X,
            n_latent=self.n_latent,
            k=self.top_k,
            lr=self.lr,
            sparsity_weight=self.alpha,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            show_progress_bar=self.show_progress_bar,
        )
        self.loss_curve_.extend(loss_curve)
        self.coef_ = np.array(params["W_e"])
        self.coef_d_ = np.array(params["W_d"])
        self.intercept_ = np.array(params["b_e"])
        self.intercept_d_ = np.array(params["b_d"])
        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def to_dict(self) -> dict:
        return dict(
            attr=self.get_params(), params=self._params, loss_curve=self.loss_curve_
        )

    @classmethod
    def from_dict(cls, data):
        obj = cls(**data["attr"])
        params = data["params"]
        obj.coef_ = np.array(params["W_e"])
        obj.coef_d_ = np.array(params["W_d"])
        obj.intercept_ = np.array(params["b_e"])
        obj.intercept_d_ = np.array(params["b_d"])
        obj.loss_curve_ = data["loss_curve"]
        return obj

    def to_disk(
        self,
        path,
    ):
        joblib.dump(self.to_dict(), path)

    @classmethod
    def from_disk(cls, path):
        data = joblib.load(path)
        return cls.from_dict(data)

    @property
    def _params(self):
        return {
            "W_e": self.coef_,
            "b_e": self.intercept_,
            "W_d": self.coef_d_,
            "b_d": self.intercept_d_,
        }

    def transform(self, X):
        if spr.issparse(X):
            X = X.todense()
        Z = []
        _encode = partial(encode, params=self._params, k=self.top_k)
        for batch_start in trange(
            0,
            X.shape[0],
            self.batch_size,
            leave=False,
            desc="Going through all batches",
            disable=not self.show_progress_bar,
        ):
            batch_end = batch_start + self.batch_size
            batch_x = X[batch_start:batch_end]
            batch_z = _encode(x=batch_x)
            Z.append(spr.csr_array(batch_z))
        return spr.vstack(Z, format="csr")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
