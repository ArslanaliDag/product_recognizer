# embedder.py
from PIL import Image
import numpy as np

class Embedder:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self._cache = {}

    def embed_image_path(self, path, use_cache=True):
        """Создает эмбеддинг для одного изображения"""
        # Кэширование по пути файла
        if use_cache and path in self._cache:
            return self._cache[path].copy()
        
        try:
            img = Image.open(path).convert("RGB")
            vec = self.model_loader.encode_image(img)
            
            # Normalize (на всякий случай, хотя модель уже нормализует)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            if use_cache:
                self._cache[path] = vec.copy()
            
            return vec
        except Exception as e:
            print(f"Error embedding image {path}: {e}")
            raise
    
    def embed_images_batch(self, paths, use_cache=True):
        """Батч-обработка нескольких изображений"""
        if not paths:
            return np.array([]).reshape(0, self.model_loader.model_dim).astype('float32')
        
        vectors = []
        for path in paths:
            try:
                vec = self.embed_image_path(path, use_cache=use_cache)
                vectors.append(vec)
            except Exception as e:
                print(f"Skipping image {path} due to error: {e}")
                continue
        
        if not vectors:
            raise ValueError("No valid images to embed")
        
        return np.stack(vectors, axis=0).astype('float32')
    
    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()