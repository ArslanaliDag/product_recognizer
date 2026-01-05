# model_loader.py
import torch
import open_clip
import os
from PIL import Image

class ModelLoader:
    def __init__(self, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", device=None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.model_dim = None
        self._load()

    def _load(self):
        print(f"Loading OpenCLIP model {self.model_name}/{self.pretrained} on {self.device} ...")

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained
        )

        model = model.to(self.device)
        model.eval()

        # Оптимизация для CPU
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        print(f"Using {num_threads} CPU threads")

        self.model = model
        self.preprocess = preprocess

        # Определяем размерность эмбеддинга
        with torch.no_grad():
            try:
                if hasattr(model.visual, 'proj'):
                    self.model_dim = model.visual.proj.shape[1]
                else:
                    # Fallback: создаем тестовое изображение
                    test_img = Image.new('RGB', (224, 224))
                    test_input = preprocess(test_img).unsqueeze(0).to(self.device)
                    test_features = model.encode_image(test_input)
                    self.model_dim = test_features.shape[-1]
            except Exception as e:
                print(f"Could not determine model dimension automatically: {e}")
                self.model_dim = 1024  # Для ViT-H-14 по умолчанию

        print("Model loaded. Embedding dim:", self.model_dim)

    @torch.no_grad()
    def encode_image(self, pil_image):
        # Препроцессинг изображения
        img = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Получаем эмбеддинг
        img_features = self.model.encode_image(img)

        # Нормализация
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        return img_features.squeeze(0).cpu().numpy().astype('float32')
