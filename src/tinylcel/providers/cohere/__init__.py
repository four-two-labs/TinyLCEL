from tinylcel.providers.cohere.embeddings import CohereEmbeddings

try:
    import cohere  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    raise ImportError('Cohere is not installed. Please install it with `pip install cohere`.') from None

__all__ = ['CohereEmbeddings']
