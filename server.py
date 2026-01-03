import io
import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <--- NEW: Для раздачи файлов
from PIL import Image
import numpy as np
import easyocr
from thefuzz import fuzz

# Импорт твоих модулей
from model_loader import ModelLoader
from embedder import Embedder
from faiss_db import FaissDB
from product_manager import ProductManager

# Настройки путей
EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.faiss")
META_PATH = os.path.join(EMBEDDINGS_DIR, "meta.json")
PRODUCTS_DIR = "products" # Папка, куда ProductManager сохраняет фото

app = FastAPI(title="AI Product Scanner API")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: Открываем доступ к папке с картинками по ссылке /images ---
os.makedirs(PRODUCTS_DIR, exist_ok=True) # Создаем папку, если нет
app.mount("/images", StaticFiles(directory=PRODUCTS_DIR), name="images")

# Глобальные переменные
model_loader = None
embedder = None
faiss_db = None
product_manager = None
ocr_reader = None

@app.on_event("startup")
async def startup_event():
    global model_loader, embedder, faiss_db, product_manager, ocr_reader
    print("🚀 Запуск сервера. Загрузка моделей...")
    
    # 1. AI Модель
    # ViT-L-14 оптимален. Если будет тормозить - ставь ViT-B-16. Если много памяти - ViT-H-14.
    model_loader = ModelLoader(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k") 
    embedder = Embedder(model_loader)
    
    # 2. База данных
    faiss_db = FaissDB(dim=model_loader.model_dim, path_index=INDEX_PATH, path_meta=META_PATH)
    product_manager = ProductManager(products_dir=PRODUCTS_DIR)
    
    # 3. OCR
    print("Загрузка OCR...")
    try:
        # gpu=False для надежности при демо. Поставь True, если настроил драйвера.
        ocr_reader = easyocr.Reader(['en', 'ru'], gpu=False)
    except Exception as e:
        print(f"Ошибка загрузки OCR: {e}")
        ocr_reader = None
        
    print(f"✅ Сервер готов! Товаров в базе: {faiss_db.index.ntotal}")

@app.post("/search")
async def search_product(
    request: Request, # <--- NEW: Получаем объект запроса, чтобы узнать текущий URL (ngrok или localhost)
    file: UploadFile = File(...)
):
    """
    Поиск товара по фото (Гибридный: CLIP + OCR) + Возврат ссылки на изображение
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    if faiss_db.index.ntotal == 0:
        return {"results": [], "message": "База пуста"}

    # --- 1. Визуальный поиск (CLIP) ---
    query_vec = model_loader.encode_image(image)
    query_vec = query_vec.reshape(1, -1).astype('float32')
    
    D, I = faiss_db.index.search(query_vec, k=min(15, faiss_db.index.ntotal))
    
    # --- 2. Текстовый поиск (OCR) ---
    ocr_words = []
    if ocr_reader:
        try:
            img_np = np.array(image)
            ocr_results = ocr_reader.readtext(img_np, detail=0)
            ocr_words = [w.upper() for w in ocr_results if len(w) > 2]
            print(f"DEBUG OCR: {ocr_words}")
        except Exception as e:
            print(f"OCR Error: {e}")

    # --- 3. Гибридная логика ---
    meta_map = faiss_db.meta
    candidates = {}
    
    VISUAL_THRESHOLD = 0.45
    TEXT_BONUS_THRESHOLD = 0.50

    for idx, visual_score in zip(I[0], D[0]):
        if visual_score < VISUAL_THRESHOLD:
            continue
            
        meta = meta_map.get(str(int(idx)))
        if not meta: continue
        
        pid = meta['product_id']
        title_upper = meta['title'].upper()
        
        text_bonus = 0.0
        matches = []
        
        if ocr_words and visual_score > TEXT_BONUS_THRESHOLD:
            for word in ocr_words:
                if len(word) < 4: continue
                ratio = fuzz.partial_ratio(word, title_upper)
                if ratio >= 85:
                    bonus = 0.25
                    if bonus > text_bonus: text_bonus = bonus
                    matches.append(f"{word} ({ratio}%)")

        total_score = min(visual_score + text_bonus, 1.0)
        
        if pid not in candidates or total_score > candidates[pid]['total_score']:
            
            # --- NEW: Генерация ссылки на картинку ---
            # Путь в метаданных: products/uuid/1.jpg
            # Нам нужно: http://domain.com/images/uuid/1.jpg
            
            local_path = meta.get('example_image', '')
            image_url = None
            
            if local_path:
                # Извлекаем часть пути после 'products'
                # Например, если путь "products/123/1.jpg", берем "123/1.jpg"
                try:
                    # Нормализуем слеши для Windows/Linux
                    norm_path = os.path.normpath(local_path)
                    parts = norm_path.split(os.sep)
                    
                    # Ищем индекс папки products и берем всё, что после
                    if PRODUCTS_DIR in parts:
                        idx = parts.index(PRODUCTS_DIR)
                        rel_path = "/".join(parts[idx+1:])
                    else:
                        # Если путь абсолютный или странный, пробуем взять последние 2 части (id/file.jpg)
                        rel_path = "/".join(parts[-2:])
                        
                    # Собираем полный URL
                    # request.base_url вернет https://xxxx.ngrok-free.app/ или http://localhost:8000/
                    image_url = f"images/{rel_path}"
                except Exception as e:
                    print(f"Error generating URL: {e}")

            candidates[pid] = {
                "id": pid,
                "title": meta['title'],
                "price": meta['price'],
                "total_score": float(total_score),
                "visual_score": float(visual_score),
                "text_bonus": float(text_bonus),
                "matches": matches,
                "image_url": image_url  # <--- Добавили ссылку
            }

    sorted_results = sorted(candidates.values(), key=lambda x: x['total_score'], reverse=True)
    
    return {"results": sorted_results[:5]}

@app.post("/add_product")
async def add_product(
    title: str = Form(...),
    price: float = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No images provided")

    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
        
        print(f"Добавляем товар: {title}, фото: {len(saved_paths)}")

        pid = product_manager.add_product(title, price, saved_paths)
        final_img_paths = product_manager.get_product_images(pid)
        
        vectors = embedder.embed_images_batch(final_img_paths)
        
        metas = [{
            "product_id": pid,
            "title": title,
            "price": price,
            "example_image": img_path
        } for img_path in final_img_paths]
        
        faiss_db.add_embeddings(vectors, metas)
        
        return {
            "status": "success",
            "product_id": pid,
            "title": title,
            "message": f"Товар добавлен. Всего: {faiss_db.index.ntotal}"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 позволяет доступ внутри локальной сети
    uvicorn.run(app, host="0.0.0.0", port=8000)