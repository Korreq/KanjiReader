from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SemanticComparisonModels:
    def __init__(self):
        # Initialize semantic comparison models and tokenizers
        self.models = {
            'minilm': (None, None)
        }

    def similarity_minilm(self):
        # Load Minilm semantic comparison model
        if self.models['minilm'][0] is None:
            model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.models['minilm'] = (model, tokenizer)
        return self.models['minilm']

    def compare_semantics(self, sentences):
        # Compute sentence embeddings using a semantic similarity model
        model, tokenizer = self.similarity_minilm()
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

    def mean_pooling(self, model_output, attention_mask):
        # Mean pool the model output to get the sentence embeddings
        token_embeddings = model_output[0]  # Token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compare_embeddings(self, embedding1, embedding2):
        # Compute cosine similarity between two sentence embeddings
        cos_sim = cosine_similarity(embedding1.unsqueeze(0).cpu().numpy(), embedding2.unsqueeze(0).cpu().numpy())
        return cos_sim[0][0]

    def compare_with_original(self, sentences):
        """Compare the embeddings of all translations with the original text."""
        embeddings = self.compare_semantics(sentences)
        similarity_scores = {
            f"Sentence {i+1}": self.compare_embeddings(embeddings[0], emb)
            for i, emb in enumerate(embeddings[1:])
        }
        return similarity_scores
