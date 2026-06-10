import joblib
import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin
from turftopic.late import LateSentenceTransformer, flatten_repr

from .autoencoder import TopKAutoEncoder


def sparse_sumpool(flat_repr, lengths):
    """Sum pooling for sparse ragged arrays. Equivalent of Turftopic's pool_flat for sparse arrays using the sum"""
    pooled = []
    start_index = 0
    for length in lengths:
        pooled.append(
            spr.csr_matrix(flat_repr[start_index : start_index + length].sum(axis=0))
        )
        start_index += length
    return spr.vstack(pooled, format="csr")


class LatentTermsVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        encoder: str | LateSentenceTransformer,
        autoencoder: TopKAutoEncoder,
        show_progress_bar: bool = True,
    ):
        self.encoder = encoder
        if isinstance(self.encoder, str):
            self._encoder = LateSentenceTransformer(self.encoder)
        else:
            self._encoder = self.encoder
        self.autoencoder = autoencoder
        self.show_progress_bar = show_progress_bar
        self.autoencoder.show_progress_bar = show_progress_bar

    def fit(self, raw_documents, y=None):
        # Does nothing, for compatibility
        return self

    def transform(self, raw_documents):
        token_embeddings, offsets = self._encoder.encode_tokens(
            list(raw_documents), show_progress_bar=self.show_progress_bar
        )
        flat_token_embeddings, lengths = flatten_repr(token_embeddings)
        flat_z = self.autoencoder.transform(flat_token_embeddings)
        # Pooling procedure from section 3.2
        pooled_z = sparse_sumpool(flat_z, lengths=lengths)
        return np.sqrt(pooled_z)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)

    def to_dict(self):
        return dict(
            encoder=self.encoder,
            autoencoder=self.autoencoder.to_dict(),
            show_progress_bar=self.show_progress_bar,
        )

    @classmethod
    def from_dict(cls, data):
        autoencoder = TopKAutoEncoder.from_dict(data["autoencoder"])
        return cls(
            encoder=data["encoder"],
            autoencoder=autoencoder,
            show_progress_bar=data["show_progress_bar"],
        )

    def to_disk(
        self,
        path,
    ):
        joblib.dump(self.to_dict(), path)

    @classmethod
    def from_disk(cls, path):
        data = joblib.load(path)
        return cls.from_dict(data)
