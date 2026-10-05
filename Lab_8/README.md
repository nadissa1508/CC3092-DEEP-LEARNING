# Laboratorio 8 — Mini-GPT, muestreo y embeddings contextuales

Notebook principal: `Lab8_MiniGPT_Sampling.ipynb` (Tasks 1 a 4).

## Dependencias
Las del entorno del curso (`requirements.txt` en la raíz) más:
```bash
pip install transformers gensim
```

## Modelo Word2Vec (no está en el repositorio)
El Task 3.2 usa un Word2Vec preentrenado en español. Pesa unos 3.2 GB descomprimido (el `.zip` pesa 2.9 GB), así que no se versiona.

- Modelo: *Spanish 3B Words Word2Vec Embeddings* (vectores de 400 dim., ~1.9 M palabras).
- Página: https://zenodo.org/record/1155473 (registro Zenodo 1410403)
- Archivo: `keyed_vectors.zip` (2.9 GB)
- Descarga directa: https://zenodo.org/api/records/1410403/files/keyed_vectors.zip/content

Descargar y descomprimir en `~/gensim-data/es/` (PowerShell):
```powershell
New-Item -ItemType Directory -Force $HOME\gensim-data\es
cd $HOME\gensim-data\es
curl.exe -L -C - -o keyed_vectors.zip "https://zenodo.org/api/records/1410403/files/keyed_vectors.zip/content"
tar -xf keyed_vectors.zip
```
Si la descarga se corta, vuelva a correr el comando `curl.exe`: el `-C -` la continúa donde quedó.

El notebook espera encontrar `~/gensim-data/es/complete.kv` y `complete.kv.vectors.npy`. El `.kv` está en un formato antiguo de gensim 3; la celda del notebook lo carga con un lector propio, por lo que funciona con gensim 4.

## BERT
`bert-base-multilingual-cased` se descarga automáticamente de HuggingFace la primera vez (~700 MB).

## Datos
`input.txt` es `tiny_shakespeare` (1 MB); el notebook lo descarga si no existe.
