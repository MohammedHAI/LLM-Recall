from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPVisionModel
from io import BytesIO
from datetime import datetime
import torch
import pyautogui
import base64
import requests
import time
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
    embedding = outputs.last_hidden_state[0][0]
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

# save log as pytorch dict, adding metadata
def save_log(log):
    timestamp = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    log['version'] =  "0.2"
    log['date_saved'] = timestamp
    log['entry_count'] = str(len(log['entries']))
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
    log = {'entries': []}
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
            if msvcrt.getch() == b'q':
                exit(0)
     
if len(sys.argv) > 1:
    main(int(sys.argv[1]))
else:
    main()
print("Exiting...")
