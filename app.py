import sys
import os
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QListWidget, QLineEdit, QFormLayout, QMessageBox, QListWidgetItem, QDialog
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from model_loader import ModelLoader
from embedder import Embedder
from faiss_db import FaissDB
from product_manager import ProductManager
import numpy as np
from PIL.ImageQt import ImageQt
from PIL import Image

# --- NEW: Библиотеки для OCR ---
import easyocr
from thefuzz import fuzz

EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.faiss")
META_PATH = os.path.join(EMBEDDINGS_DIR, "meta.json")

class RecognitionWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str) # Для отображения статуса (OCR занимает время)
    
    def __init__(self, embedder, faiss_db, ocr_reader, image_path, threshold=0.5):
        super().__init__()
        self.embedder = embedder
        self.faiss_db = faiss_db
        self.ocr_reader = ocr_reader # Передаем загруженный OCR
        self.image_path = image_path
        self.threshold = threshold
    
    def run(self):
        try:
            if self.faiss_db.index.ntotal == 0:
                self.error.emit("База данных пуста.")
                return
            
            # 1. Поиск по картинке (CLIP)
            self.progress.emit("Анализ изображения...")
            query_vec = self.embedder.embed_image_path(self.image_path)
            query_vec = query_vec.reshape(1, -1).astype('float32')
            
            # Берем топ-15 кандидатов
            D, I = self.faiss_db.index.search(query_vec, k=min(15, self.faiss_db.index.ntotal))
            
            if I is None or len(I[0]) == 0:
                self.finished.emit([])
                return

            # 2. Поиск текста (OCR)
            self.progress.emit("Чтение текста (OCR)...")
            ocr_words = []
            try:
                # detail=0 дает просто список строк, нам этого хватит для скорости
                ocr_results = self.ocr_reader.readtext(self.image_path, detail=0)
                # Фильтруем короткий мусор (менее 3 букв)
                ocr_words = [w.upper() for w in ocr_results if len(w) > 2]
                print(f"DEBUG: Найден текст на фото: {ocr_words}")
            except Exception as e:
                print(f"OCR Error: {e}")

            # 3. Объединение результатов (HYBRID SEARCH)
            self.progress.emit("Сравнение результатов...")
            
            candidates = {} # pid -> {meta, visual_score, text_bonus, total_score}
            meta_map = self.faiss_db.meta
            
            for idx, visual_score in zip(I[0], D[0]):
                if visual_score < self.threshold:
                    continue

                meta = meta_map.get(str(int(idx)))
                if meta is None:
                    continue
                
                pid = meta['product_id']
                title_upper = meta['title'].upper()
                
                # --- ЛОГИКА ТЕКСТОВОГО БОНУСА ---
                text_bonus = 0.0
                matched_keywords = []

                if ocr_words and visual_score > 0.5:
                    for word in ocr_words:

                        if len(word) < 4: 
                            continue

                        # Используем partial_ratio (поиск подстроки с нечеткостью)
                        ratio = fuzz.partial_ratio(word, title_upper)
                        
                        # Если совпадение сильное (>75%), даем большой бонус
                        if ratio >= 85:
                            bonus = 0.25 
                            if bonus > text_bonus: text_bonus = bonus
                            matched_keywords.append(f"{word}({ratio}%)")

                # Расчет финального балла
                total_score = visual_score + text_bonus
                
                # Сохраняем лучшую версию этого товара
                if pid not in candidates or total_score > candidates[pid]['total_score']:
                    candidates[pid] = {
                        'meta': meta,
                        'visual_score': visual_score,
                        'text_bonus': text_bonus,
                        'total_score': min(total_score, 1.0), # Не больше 1.0
                        'matches': matched_keywords
                    }

            # Превращаем в список и сортируем
            final_results = []
            for item in candidates.values():
                final_results.append((item['meta'], item['total_score'], item['text_bonus'], item['matches']))
            
            # Сортировка по финальному баллу
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            self.finished.emit(final_results)
            
        except Exception as e:
            import traceback
            error_text = f"{str(e)}\n\nПодробности:\n{traceback.format_exc()}"
            self.error.emit(error_text)

class AddProductWorker(QThread):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, pm, embedder, faiss_db, title, price, image_paths):
        super().__init__()
        self.pm = pm
        self.embedder = embedder
        self.faiss_db = faiss_db
        self.title = title
        self.price = price
        self.image_paths = image_paths
    
    def run(self):
        try:
            self.progress.emit("Создание товара...")
            pid = self.pm.add_product(self.title, self.price, self.image_paths)
            img_paths = self.pm.get_product_images(pid)
            
            self.progress.emit("Создание векторов...")
            vectors = self.embedder.embed_images_batch(img_paths)
            
            metas = [{
                    "product_id": pid,
                    "title": self.title,
                    "price": self.price,
                    "example_image": img_path
                } for img_path in img_paths]
            
            self.faiss_db.add_embeddings(vectors, metas)
            self.finished.emit(pid, self.title)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Product Scanner (MVP)")
        self.resize(1100, 700)

        # 1. Инициализация моделей
        print("Инициализация AI моделей...")
        self.model = ModelLoader()
        self.embedder = Embedder(self.model)
        self.pm = ProductManager()
        self.faiss = FaissDB(dim=self.model.model_dim, path_index=INDEX_PATH, path_meta=META_PATH)
        
        # --- NEW: Инициализация OCR (один раз при запуске) ---
        print("Загрузка OCR модуля (может занять время)...")
        # gpu=False для стабильности на MVP, если драйверы шалят. 
        # Если починишь драйверы, поставь gpu=True
        try:
            self.ocr_reader = easyocr.Reader(['en', 'ru'], gpu=False) 
            print("OCR модуль готов.")
        except Exception as e:
            print(f"Ошибка загрузки OCR: {e}")
            self.ocr_reader = None

        # UI
        self.left_image_label = QLabel("Загрузите фото")
        self.left_image_label.setFixedSize(450, 450)
        self.left_image_label.setStyleSheet("border: 2px dashed #555;")
        self.left_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.load_btn = QPushButton("📷 Выбрать фото")
        self.load_btn.clicked.connect(self.load_query_image)
        self.load_btn.setHeight = 50

        self.recognize_btn = QPushButton("🔍 РАСПОЗНАТЬ ТОВАР")
        self.recognize_btn.clicked.connect(self.run_recognition)
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.top_candidates_list = QListWidget()
        self.top_candidates_list.itemClicked.connect(self.show_candidate_image)

        # Result Details
        self.result_title = QLabel("Название: -")
        self.result_title.setWordWrap(True)
        self.result_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.result_price = QLabel("Цена: -")
        self.result_conf = QLabel("Уверенность: -")

        # Add Product Section
        self.add_title = QLineEdit()
        self.add_price = QLineEdit()
        self.add_images_btn = QPushButton("Выбрать фото для базы")
        self.add_images_btn.clicked.connect(self.select_add_images)
        self.add_images_list = QListWidget()
        self.add_product_btn = QPushButton("➕ Добавить в базу")
        self.add_product_btn.clicked.connect(self.add_product)

        # Layouts
        main_layout = QHBoxLayout()
        
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.left_image_label)
        left_panel.addWidget(self.load_btn)
        left_panel.addWidget(self.recognize_btn)
        left_panel.addWidget(self.status_label)
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("РЕЗУЛЬТАТ ПОИСКА:"))
        right_panel.addWidget(self.result_title)
        right_panel.addWidget(self.result_price)
        right_panel.addWidget(self.result_conf)
        right_panel.addWidget(self.top_candidates_list)
        
        right_panel.addSpacing(30)
        right_panel.addWidget(QLabel("--- ПАНЕЛЬ АДМИНИСТРАТОРА (ДОБАВЛЕНИЕ) ---"))
        form = QFormLayout()
        form.addRow("Название:", self.add_title)
        form.addRow("Цена:", self.add_price)
        right_panel.addLayout(form)
        right_panel.addWidget(self.add_images_btn)
        right_panel.addWidget(self.add_images_list)
        right_panel.addWidget(self.add_product_btn)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

        self.query_image_path = None
        self.to_add_images = []
        self.current_top_candidates = []

    def load_query_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите фото", "", "Images (*.png *.jpg *.jpeg)")
        if not path: return
        self.query_image_path = path
        pixmap = QPixmap(path).scaled(450, 450, Qt.AspectRatioMode.KeepAspectRatio)
        self.left_image_label.setPixmap(pixmap)
        self.recognize_btn.setEnabled(True)
        self.status_label.setText("Готово к поиску")

    def run_recognition(self):
        if not self.query_image_path: return
        
        self.recognize_btn.setEnabled(False)
        self.top_candidates_list.clear()
        
        # Передаем self.ocr_reader в воркер
        self.worker = RecognitionWorker(
            self.embedder, 
            self.faiss, 
            self.ocr_reader, # <-- Передаем OCR
            self.query_image_path
        )
        self.worker.finished.connect(self.on_recognition_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def on_progress(self, msg):
        self.status_label.setText(msg)

    def on_recognition_finished(self, results):
        self.recognize_btn.setEnabled(True)
        self.status_label.setText(f"Найдено: {len(results)}")
        
        if not results:
            QMessageBox.information(self, "Упс", "Ничего похожего не найдено.")
            return

        # Лучший результат
        best_meta, best_score, best_bonus, best_matches = results[0]
        
        self.result_title.setText(f"{best_meta.get('title')}")
        self.result_price.setText(f"Цена: {best_meta.get('price')} руб.")
        
        # Красивый вывод уверенности
        conf_text = f"{(best_score*100):.1f}%"
        if best_bonus > 0:
            conf_text += f" (Visual: {(best_score-best_bonus)*100:.1f}% + Text Bonus)"
        self.result_conf.setText(f"Уверенность: {conf_text}")

        # Заполнение списка
        self.current_top_candidates = results
        for meta, score, bonus, matches in results[:5]:
            match_str = f" [OCR: {','.join(matches)}]" if matches else ""
            item_text = f"{score:.2f} | {meta.get('title')}{match_str}"
            
            item = QListWidgetItem(item_text)
            # Подсветка, если OCR помог
            if bonus > 0:
                item.setBackground(Qt.GlobalColor.green) 
                item.setForeground(Qt.GlobalColor.black)
            
            self.top_candidates_list.addItem(item)

    def on_error(self, msg):
        self.recognize_btn.setEnabled(True)
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Error", msg)

    # --- Методы добавления (без изменений) ---
    def select_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Фото", "", "Images (*.png *.jpg *.jpeg)")
        if paths:
            self.to_add_images = paths[:4]
            self.add_images_list.clear()
            for p in paths: self.add_images_list.addItem(p)

    def add_product(self):
        title = self.add_title.text().strip()
        price = self.add_price.text().strip()
        if not title or not price or not self.to_add_images:
            QMessageBox.warning(self, "Error", "Заполните данные")
            return
        
        self.add_product_btn.setEnabled(False)
        self.add_worker = AddProductWorker(self.pm, self.embedder, self.faiss, title, price, self.to_add_images)
        self.add_worker.finished.connect(lambda: [self.add_product_btn.setEnabled(True), QMessageBox.information(self, "OK", "Добавлено")])
        self.add_worker.start()

    def show_candidate_image(self, item):
        idx = self.top_candidates_list.row(item)
        meta = self.current_top_candidates[idx][0]
        path = meta.get("example_image")
        if path and os.path.exists(path):
            d = QDialog(self)
            l = QVBoxLayout()
            lbl = QLabel()
            lbl.setPixmap(QPixmap(path).scaled(500,500, Qt.AspectRatioMode.KeepAspectRatio))
            l.addWidget(lbl)
            d.setLayout(l)
            d.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())