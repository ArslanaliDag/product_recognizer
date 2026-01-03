# faiss_db.py
import os
import json
import numpy as np
import faiss

class FaissDB:
    def __init__(self, dim, path_index="embeddings/index.faiss", path_meta="embeddings/meta.json"):
        self.dim = dim
        self.path_index = path_index
        self.path_meta = path_meta
        self.index = None
        self.meta = {}
        self.use_ivf = False
        os.makedirs(os.path.dirname(self.path_index), exist_ok=True)
        self._init_index()

    def _init_index(self):
        if os.path.exists(self.path_index) and os.path.exists(self.path_meta):
            print("Loading FAISS index from disk...")
            try:
                self.index = faiss.read_index(self.path_index)
                self._load_meta()
                print(f"FAISS index loaded, size: {self.index.ntotal}")
                self.use_ivf = isinstance(self.index, faiss.IndexIVFFlat)
            except Exception as e:
                print(f"Error loading index: {e}, creating new one")
                self.index = faiss.IndexFlatIP(self.dim)
                self.meta = {}
        else:
            print("Creating new FAISS index")
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = {}

    def _load_meta(self):
        if os.path.exists(self.path_meta):
            try:
                with open(self.path_meta, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.meta = {}
        else:
            self.meta = {}

    def _save_meta(self):
        try:
            with open(self.path_meta, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def add_embeddings(self, vectors, metas):
        """
        vectors: numpy array shape (N, dim) float32
        metas: list of dict metadata
        """
        if len(vectors) == 0:
            print("Warning: No vectors to add")
            return
            
        assert len(vectors) == len(metas), f"Vectors ({len(vectors)}) and metas ({len(metas)}) must have same length"
        
        # Приводим к нужному типу и проверяем размерность
        vectors = np.asarray(vectors).astype('float32')
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dim}")
        
        # Получаем текущий размер индекса
        start_id = self.index.ntotal
        
        # Добавляем векторы
        self.index.add(vectors)
        
        # Обновляем метаданные
        for i, m in enumerate(metas):
            self.meta[str(start_id + i)] = m
        
        # Сохраняем
        self.save()
        print(f"Added {len(vectors)} embeddings. Total: {self.index.ntotal}")
        
        # Проверяем, нужно ли обновить до IVF
        if not self.use_ivf and self.index.ntotal >= 1000:
            self._upgrade_to_ivf()

    def _upgrade_to_ivf(self):
        """Обновляет индекс до IVF для ускорения поиска при большом размере БД"""
        if self.use_ivf or self.index.ntotal < 1000:
            return
        
        print("Upgrading to IVF index for better performance...")
        try:
            # Извлекаем все векторы
            all_vectors = np.zeros((self.index.ntotal, self.dim), dtype='float32')
            for i in range(self.index.ntotal):
                all_vectors[i] = self.index.reconstruct(int(i))
            
            # Создаем IVF индекс
            nlist = min(100, max(10, self.index.ntotal // 10))
            quantizer = faiss.IndexFlatIP(self.dim)
            new_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Обучаем и добавляем векторы
            new_index.train(all_vectors)
            new_index.add(all_vectors)
            new_index.nprobe = 10
            
            self.index = new_index
            self.use_ivf = True
            self.save()
            print("Successfully upgraded to IVF index")
        except Exception as e:
            print(f"Failed to upgrade to IVF: {e}, keeping flat index")

    def search(self, qvec, topk=5):
        """
        qvec: numpy array shape (dim,), normalized
        returns list of list of (meta, score)
        """
        # Проверяем, есть ли данные в индексе
        if self.index is None or self.index.ntotal == 0:
            print("Warning: Index is empty, cannot search")
            return [[]]
        
        # Приводим к нужной форме
        

        q = np.asarray(qvec, dtype='float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # Проверяем размерность
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dimension {q.shape[1]} does not match index dimension {self.dim}")
        
        # Ограничиваем topk размером индекса
        actual_topk = min(topk, self.index.ntotal)
        
        try:
            D, I = self.index.search(q, actual_topk)
        except Exception as e:
            print(f"Search error: {e}")
            return [[]]
        
        results = []
        for dist_row, idx_row in zip(D, I):
            row = []
            for score, idx in zip(dist_row, idx_row):
                if idx < 0 or idx >= self.index.ntotal:
                    continue
                meta = self.meta.get(str(int(idx)), None)
                if meta:
                    row.append((meta, float(score)))
            results.append(row)
        
        return results

    def save(self):
        try:
            faiss.write_index(self.index, self.path_index)
            self._save_meta()
            print(f"FAISS index saved to {self.path_index}")
        except Exception as e:
            print(f"Error saving index: {e}")
            raise

    def rebuild(self, vectors, metas):
        """
        Rebuild index from scratch with given vectors & metas.
        vectors: np array (N, dim)
        metas: list of dicts
        """
        self.index = faiss.IndexFlatIP(self.dim)
        if len(vectors) > 0:
            self.index.add(np.asarray(vectors).astype('float32'))
        self.meta = {}
        for i, m in enumerate(metas):
            self.meta[str(i)] = m
        self.use_ivf = False
        self.save()