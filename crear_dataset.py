from icrawler.builtin import BingImageCrawler
import os

# Clases de animales
clases = ["perro", "gato", "caballo", "elefante"]

# Carpeta base
ruta_base = "dataset_animales"

if not os.path.exists(ruta_base):
    os.makedirs(ruta_base)

for clase in clases:
    print(f"Descargando imágenes de {clase}...")
    
    ruta_clase = os.path.join(ruta_base, clase)
    os.makedirs(ruta_clase, exist_ok=True)

    crawler = BingImageCrawler(storage={'root_dir': ruta_clase})
    crawler.crawl(keyword=clase, max_num=50)  # puedes subir a 100+

print("✅ Dataset creado correctamente")