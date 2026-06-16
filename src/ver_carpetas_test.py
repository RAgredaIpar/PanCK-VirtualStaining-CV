import os

# Apuntamos a tu carpeta de pruebas raw
TEST_DIR = r"D:\job\TESIS\data\raw\DeepLIIF_Testing_Set"

if not os.path.exists(TEST_DIR):
    print(f"[-] ERROR: No se encontró la ruta: {TEST_DIR}")
else:
    print("[+] Carpeta de pruebas localizada con éxito.")

    # Listar los primeros 10 elementos que encuentre (archivos o subcarpetas)
    elementos = os.listdir(TEST_DIR)
    print(f"[+] Total de elementos encontrados en el set de prueba: {len(elementos)}")
    print("\n--- Muestra de los primeros 10 archivos/carpetas ---")
    for elem in elementos[:10]:
        print(f" - {elem}")