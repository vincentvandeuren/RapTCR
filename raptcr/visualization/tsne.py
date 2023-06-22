from ..hashing import Cdr3Embedder
from ..analysis import Repertoire
from sklearn.base import TransformerMixin
from sklearn.manifold import TSNE

class TsneTransformer(TransformerMixin):
    def __init__(self, embedder:Cdr3Embedder, tsne=None, **kwargs) -> None:

        if not tsne:
            perplexity = kwargs.pop("perplexity", 200)
            n_jobs = kwargs.pop("n_jobs", -1)

            tsne = TSNE(
                perplexity=perplexity,
                n_jobs=n_jobs,
                **kwargs
                )
        self.embedder = embedder
        self.tsne = tsne

        super().__init__() 

    def __repr__(self) -> str:
        return "tSNE repertoire transformer"
    
    def fit_transform(self, data:Repertoire):
        hashes = self.embedder.transform(data)
        embedding = self.tsne.fit_transform(hashes)
        data["x"], data["y"] = embedding.T
        return data