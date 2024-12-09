from sentence_transformers import SentenceTransformer


class NLI:
    def __init__(self, model_name, generation_settings):
        self.model = SentenceTransformer(model_name)
        self.generation_settings = generation_settings
        self.device = "cpu"

    def check_entailment(self, premise, hypothesis, threshold = 0.6):
        # Compute embeddings for both lists
        embeddings1 = self.model.encode(premise)
        embeddings2 = self.model.encode(hypothesis)

        # Compute cosine similarities
        similarities = self.model.similarity(embeddings1, embeddings2)

        return similarities[0, 0].item() > threshold