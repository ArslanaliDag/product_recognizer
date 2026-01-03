# product_manager.py
import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict

DB_FILE = "db.sqlite"
PRODUCTS_DIR = "products"

class ProductManager:
    def __init__(self, db_file=DB_FILE, products_dir=PRODUCTS_DIR):
        self.db_file = db_file
        self.products_dir = products_dir
        os.makedirs(self.products_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products(
            id TEXT PRIMARY KEY,
            title TEXT,
            price REAL,
            created_at TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS product_images(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            path TEXT,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
        """)
        conn.commit()
        conn.close()

    def add_product(self, title: str, price: float, image_paths: List[str]) -> str:
        """
        image_paths: list of local image file paths to copy into products/<id>/
        returns product id
        """
        pid = str(uuid.uuid4())
        pdir = os.path.join(self.products_dir, pid)
        os.makedirs(pdir, exist_ok=True)
        # copy images
        stored_paths = []
        for i, p in enumerate(image_paths):
            ext = Path(p).suffix or ".jpg"
            dst = os.path.join(pdir, f"{i+1}{ext}")
            shutil.copyfile(p, dst)
            stored_paths.append(dst)
        # insert into sqlite
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("INSERT INTO products(id, title, price, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (pid, title, float(price)))
        for sp in stored_paths:
            cur.execute("INSERT INTO product_images(product_id, path) VALUES (?, ?)", (pid, sp))
        conn.commit()
        conn.close()
        return pid

    def list_products(self):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("SELECT id, title, price FROM products")
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_product_images(self, product_id: str):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("SELECT path FROM product_images WHERE product_id = ?", (product_id,))
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_product(self, product_id: str):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("SELECT id, title, price FROM products WHERE id = ?", (product_id,))
        row = cur.fetchone()
        conn.close()
        return row
