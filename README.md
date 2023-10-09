# Lectura de PDF Chatbot con Langchain y Streamlit
Este Chatbot es una aplicación interactiva desarrollada para interactuar con su PDF. Está construido utilizando Open Source Stack. 

<p align=center>
<img src="src\banner.png" height = 420 weight=500>
<p>


## Arquitectura de la aplicación

<p align=center>
<img src="src\arq.png" height = 450 weight=250>
<p>

## Ejecución Local 💻

Siga estos pasos para configurar y ejecutar el proyecto en su máquina local.

### Instalación:

```bash
# Clonar el repositorio:
git clone <repository_url>
```

# Crear el entorno virtual :
```bash
python -m virtualenv venv
source venv/Scripts/activate
```


## Instalar las dependencias en el ambiente virtual :

```bash
pip install -r requirements.txt`
```

## Ejecutar la ingestion para la data:

```bash
python ingest.py
```
## Ejecutar chatbot application en Streamlit:

```bash
streamlit run chatbot_app.py
```

### Si tienes inconvenientes en la ingestion puedes ejecutar en PowerShell

## Crear el entorno virtual:
```sh
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force 
./venv/Scripts/activate.ps1
```

## Ejecutar la ingestion para la data:

```sh
py ingest.py
```

## Para usar el modelo se necesita setear una variable y luego clonar:
```sh

wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin

git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

## Ejecutar chatbot application en Streamlit:
```bash
streamlit run app.py
```