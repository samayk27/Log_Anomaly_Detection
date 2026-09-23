
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedTextVectorizer:
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 use_svd: bool = True,
                 svd_components: int = 300):
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_svd = use_svd
        self.svd_components = svd_components
        
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            sublinear_tf=True,
            norm='l2'
        )
        
        
        if use_svd:
            self.svd = TruncatedSVD(
                n_components=svd_components,
                random_state=42,
                algorithm='randomized'
            )
            self.scaler = StandardScaler()
        
        self.is_fitted = False
        self.feature_names = []
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        processed_texts = []
        
        for text in texts:
            
            processed = text.lower()
            
            
            processed = self._remove_timestamps(processed)
            
            
            processed = self._remove_ips(processed)
            
            
            processed = self._clean_numbers(processed)
            
            
            processed = ' '.join(processed.split())
            
            processed_texts.append(processed)
        
        return processed_texts
    
    def _remove_timestamps(self, text: str) -> str:
        import re
        
        patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}:\d{2}:\d{2}',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text
    
    def _remove_ips(self, text: str) -> str:
        import re
        return re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)
    
    def _clean_numbers(self, text: str) -> str:
        import re
        
        text = re.sub(r'\b\d{8,}\b', '<LONGNUM>', text)
        
        text = re.sub(r'\b[1-5]\d{2}\b', '<STATUS>', text)
        
        text = re.sub(r'\bport\s+\d+\b', '<PORT>', text)
        return text
    
    def fit(self, texts: List[str]) -> 'AdvancedTextVectorizer':
        print("Preprocessing texts...")
        processed_texts = self.preprocess_texts(texts)
        
        print("Fitting TF-IDF vectorizer...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        if self.use_svd:
            print(f"Applying SVD dimensionality reduction to {self.svd_components} components...")
            self.svd.fit(tfidf_matrix)
            
            
            tfidf_reduced = self.svd.transform(tfidf_matrix)
            self.scaler.fit(tfidf_reduced)
        
        self.is_fitted = True
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"TF-IDF vocabulary size: {len(self.feature_names)}")
        print(f"Final feature dimension: {self.svd_components if self.use_svd else len(self.feature_names)}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        processed_texts = self.preprocess_texts(texts)
        tfidf_matrix = self.tfidf_vectorizer.transform(processed_texts)
        
        if self.use_svd:
            tfidf_reduced = self.svd.transform(tfidf_matrix)
            tfidf_scaled = self.scaler.transform(tfidf_reduced)
            return tfidf_scaled
        
        return tfidf_matrix.toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
    
    def get_feature_importance(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        tfidf_matrix = self.tfidf_vectorizer.transform(self.preprocess_texts(texts))
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        
        top_indices = np.argsort(mean_scores)[-top_k:][::-1]
        top_features = [(self.feature_names[i], mean_scores[i]) for i in top_indices]
        
        return top_features
    
    def save(self, filepath: str):
        save_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'svd': self.svd if self.use_svd else None,
            'scaler': self.scaler if self.use_svd else None,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'params': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'use_svd': self.use_svd,
                'svd_components': self.svd_components
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Vectorizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedTextVectorizer':
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        instance = cls(**save_data['params'])
        instance.tfidf_vectorizer = save_data['tfidf_vectorizer']
        if instance.use_svd:
            instance.svd = save_data['svd']
            instance.scaler = save_data['scaler']
        instance.is_fitted = save_data['is_fitted']
        instance.feature_names = save_data['feature_names']
        
        print(f"Vectorizer loaded from {filepath}")
        return instance


class SimpleSentenceEmbedder:
    
    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.word_to_vec = {}
        self.tfidf_vectorizer = None
        self.is_fitted = False
    
    def _create_simple_embeddings(self, vocabulary: List[str]) -> dict:
        embeddings = {}
        
        for word in vocabulary:
            
            vec = np.zeros(self.embedding_dim)
            
            
            for i, char in enumerate(word[:min(len(word), self.embedding_dim)]):
                vec[i] = ord(char) / 255.0
            
            
            if len(word) > 0:
                vec[self.embedding_dim - 3] = len(word) / 20.0  
                vec[self.embedding_dim - 2] = word.count('@') > 0  
                vec[self.embedding_dim - 1] = word.isdigit()  
            
            embeddings[word] = vec
        
        return embeddings
    
    def fit(self, texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            min_df=2,
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(texts)
        
        
        vocabulary = self.tfidf_vectorizer.get_feature_names_out()
        print(f"Creating embeddings for {len(vocabulary)} words...")
        self.word_to_vec = self._create_simple_embeddings(vocabulary)
        
        self.is_fitted = True
        print(f"Sentence embedder fitted with {self.embedding_dim}D embeddings")
    
    def _get_sentence_embedding(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted first")
        
        
        try:
            tfidf_vec = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            word_scores = dict(zip(feature_names, tfidf_vec.toarray()[0]))
        except:
            return np.zeros(self.embedding_dim)
        
        
        embedding = np.zeros(self.embedding_dim)
        total_weight = 0
        
        for word, weight in word_scores.items():
            if weight > 0 and word in self.word_to_vec:
                embedding += weight * self.word_to_vec[word]
                total_weight += weight
        
        if total_weight > 0:
            embedding /= total_weight
        
        return embedding
    
    def transform(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            embedding = self._get_sentence_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
