from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPVisionModel
from io import BytesIO
from datetime import datetime
#from chromadb.config import Settings
#from pymilvus import MilvusClient
import torch
import pyautogui
import base64
import requests
import time
#import chromadb
import threading
import msvcrt
import sys

HF_HOME = "D:\\Downloads\\clip-vit-large-patch14"
LLM_URL = "http://localhost:5001/v1"
#LLM_URL = "http://192.168.0.63:5001/v1"
SYSTEM_PROMPT = '''You are a helpful assistant that can see images.

Please respond to all user requests.
'''
SAVE_PATH = "desktop_log.pt"
#DB_PATH = "database"

'''
# Milvus
def setup_milvus():
    client = MilvusClient(DB_PATH)
    if client.has_collection("logs"):
        client.drop_collection("logs")
    else:
        client.create_collection(
            collection_name="logs",
            dimension=512,
            auto_id=True,
            enable_dynamic_field=True
        )
    return client

def save_entries(client, log):
    for entry in log['entries']:
        data = {"vector": entry['embedding'], "caption": ['caption'], "timestamp": ['timestamp']}
        client.insert(collection_name="logs", data=data)
    print(f"Log saved to {DB_PATH}")
'''

'''
#ChromaDB
def setup_chromadb():
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    client.reset()
    logs_db = client.create_collection("logs")
    return logs_db

def save_logs_db(logs_db, log):
    for i in range(len(log)):
        logs_db.add(
        embeddings=[log['entries'][i]['embedding']],
        metadatas=[{"caption": log['entries'][i]['caption'], "timestamp": log['entries'][i]['timestamp']}],
        ids=[str(i)]
    )
    print(f"Log saved to {DB_PATH}")
'''

# obtain vector embedding for an image
def get_image_embedding(image):
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=HF_HOME)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=HF_HOME)

    #image = Image.open("C:\\Users\\Lenovo\\Documents\\Lightshot\\VLM Testing\\desktop.png")
    #image2 = Image.open("C:\\Users\\Lenovo\\Documents\\Lightshot\\VLM Testing\\chrome.png")

    inputs = processor(images=image, return_tensors="pt")
    #print(inputs)

    with torch.no_grad():
        outputs = model(**inputs)

    # image only
    embedding = outputs.last_hidden_state[0]
    #print(len(embedding))
    return embedding

# encode screenshot into base64
def encode_screenshot(screenshot):
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    screenshot64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return screenshot64

# use llm to caption a single image
def get_llm_caption(screenshot64):
    messages = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what is happening in this image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    response = requests.post(LLM_URL + "/chat/completions", json=messages)
    caption = response.json()['choices'][0]['message']['content']
    print(caption)
    return caption

# add a screenshot record to the log
def add_to_log(log, last_id, caption, embedding):
    timestamp = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    record = {'id': last_id, 'timestamp': timestamp, 'caption': caption, 'embedding': embedding}
    log['entries'].append(record)
    return last_id + 1

# save log as pytorch dict
def save_log(log):
    torch.save(log, SAVE_PATH)
    print(f"Log saved to {SAVE_PATH}")

# unused test operation
def async_operation():
    """Simulates an asynchronous operation."""
    print(f"Async operation started, running for 3 seconds...")
    time.sleep(3)
    print("Async operation completed.")
    return True  # Indicate completion
          
def main(interval=60):
    #client = setup_milvus()
    #logs_db = setup_chromadb()
    log = {'entries': [], 'version': "0.1"}
    last_id = 0
    print("Started Recording. press Q to exit or S to save anytime")
    while True:
        screenshot = pyautogui.screenshot()
        screenshot64 = encode_screenshot(screenshot)
        caption = get_llm_caption(screenshot64)
        embedding = get_image_embedding(screenshot)
        last_id = add_to_log(log, last_id, caption, embedding)
        time.sleep(interval)

        # check keyboard asynchronously to halt program
        if msvcrt.kbhit():
            if msvcrt.getch() == b's':
                save_log(log)
                #save_entries(milvus, log)
                #save_logs_db(logs_db, log)
            if msvcrt.getch() == b'q':
                exit(0)
     
if len(sys.argv) > 1:
    main(int(sys.argv[1]))
else:
    main()
print("Exiting...")
