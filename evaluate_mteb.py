import mteb
import numpy as np
from mteb.models import SearchProtocol
from turftopic import load_model
from turftopic.retrieval import BM25Transformer


class LatentTermsEvaluator(SearchProtocol):
    def __init__(self, hf_name: str):
        self.hf_name = hf_name
        self.model = load_model(hf_name)
        self.bm25 = BM25Transformer()
        super().__init__(model_name=hf_name, revision=None)

    def index(
        self,
        corpus,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        encode_kwargs,
        num_proc=None,
    ):
        texts = list(corpus["text"])
        self.X_orig = self.model.transform(texts)
        self.X_transformed = self.bm25.fit_transform(self.X_orig)
        self.ids = np.array(list(corpus["id"]))

    def search(
        self,
        queries,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        top_k,
        encode_kwargs,
        top_ranked=None,
        num_proc=None,
    ):
        q_X = self.model.transform(list(queries["text"]))
        sim = np.asarray((q_X @ self.X_transformed.T).todense())
        res = {}
        for q_id, scores in zip(queries["id"], sim):
            top_idx = np.argsort(-scores)[:top_k]
            res[q_id] = dict(zip(self.ids[top_idx], scores))
        return res


model = LatentTermsEvaluator("kardosdrur/latent-terms_paraphrase-MiniLM-L6-v2")
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
# Filter to only retrieval tasks
retrieval_tasks = mteb.filter_tasks(benchmark, task_types=["Retrieval"])
results = mteb.evaluate(model, tasks=retrieval_tasks)

results.to_disk("mteb_results.json")
