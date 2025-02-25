import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    
    @staticmethod
    def clean_text(text: str, remove_punctuation: bool = True) -> str:
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")
        
        text = text.lower().strip()
        
        if remove_punctuation:
            text = ''.join([c for c in text if c.isalnum() or c.isspace()])
            
        return ' '.join(text.split())  # Normalize whitespace

class TextEmbedder:
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
        self.preprocessor = TextPreprocessor()
        
    def text_to_embedding(
        self, 
        texts: Union[str, List[str]], 
        show_progress: bool = False
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        try:
            all_embeddings = []
            for i in tqdm(range(0, len(cleaned_texts), self.batch_size),
                        disable=not show_progress):
                batch = cleaned_texts[i:i + self.batch_size]
                embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    normalize_embeddings=False,
                    show_progress_bar=False
                )
                all_embeddings.append(embeddings.cpu().numpy())
                
            # Concatenate all batches
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            return final_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def save_embeddings(self, embeddings: np.ndarray, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        try:
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise

    @staticmethod
    def load_embeddings(filepath: Union[str, Path]) -> np.ndarray:
        filepath = Path(filepath)
        try:
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise

def main():
    embedder = TextEmbedder(batch_size=16)
    
    texts = [
        "I LOVE YOU",
        "I HATE YOU",
        "I LIKE YOU"
    ]
    
    try:
        embeddings = embedder.text_to_embedding(texts, show_progress=True)
        
        # PyTorch tensor
        embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        print(f"Embedding shape: {embedding_tensor.shape}")
        
        # Save 
        embedder.save_embeddings(embeddings, "embeddings.npy")
        
        # Load 
        loaded_embeddings = embedder.load_embeddings("embeddings.npy")
        print(f"Loaded embedding shape: {loaded_embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    
    for i, text in enumerate(texts):
        print(f"Text: {text}\nEmbedding: {embeddings[i][:5]}...\n")  # Print first 5 values


if __name__ == "__main__":
    main()