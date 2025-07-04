from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPVisionModel
import os
import torch

HF_HOME = os.environ['HF_HOME']
SAVE_PATH = "desktop_log.pt"
SIMILARITY_MINIMUM = 0.75

clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=HF_HOME)
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=HF_HOME)

# obtain vector embedding for an image
def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = clip_model(**inputs)

    embedding = outputs.last_hidden_state[0][0]
    return embedding

# filter entries by text match
def search_log_text(log, query):
    filtered_log = []
    for entry in log['entries']:
        if query in entry['caption']:
            filtered_log.append(entry)
    return filtered_log

# handle converting the embedding depending on log version
def handle_embedding(embedding, log_version):
    if float(log_version) < 0.2:
        return embedding[0]
    else:
        return embedding

# filter entries by similarity score
def search_log_image(log, query):
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)
    
    try:
        image = Image.open(query)
    except:
        print("Error: Failed to open image")
        return [] # empty log
    
    q_embedding = handle_embedding(get_image_embedding(image), log['version'])
    filtered_log = []
    for entry in log['entries']:
        # compute similarity between query embedding and entry embedding
        similarity = cosine_similarity(q_embedding, handle_embedding(entry['embedding'], log['version'])).item()
        if similarity >= SIMILARITY_MINIMUM:
            filtered_log.append(entry)
    return filtered_log

# display entries or a subset of entries
def view_entries(entries):
    if len(entries) < 1:
        print("No entries")
        return
    
    position = 0
    while True:
        print("Entry " + str(entries[position]['id'] + 1))
        print(entries[position]['timestamp'])
        print(entries[position]['caption'])
        print()
        print("P - Previous entry")
        print("N - Next entry")
        print("Q - Quit")
        choice = input("Choose an option >")
        if choice.lower() == "p":
            position -= 1
            if position < 0:
                position = 0
        elif choice.lower() == "n":
            position += 1
            if position >= len(entries):
                position = len(entries) - 1
        elif choice.lower() == "q":
            return
        else:
            print("Invalid option")

# subject to version number
def view_metadata(log):
    print("Log version: " + log['version'])
    if float(log['version']) >= 0.2:
        print("Date saved: " + log['date_saved'])
        print("Entry count: " + log['entry_count'])
    print("End of metadata")

def main():
    try:
        log = torch.load(SAVE_PATH)
        print(f"loaded log from {SAVE_PATH}")
    except:
        print("Error: couldn't load log")
        
    while True:
        print()
        print("Options:")
        print("A - View all entries")
        print("T - Text search")
        print("I - Image search")
        print("M - View metadata")
        print("Q - Quit")
        choice = input("Choose an option >")
        if choice.lower() == "a":
            view_entries(log['entries'])
        elif choice.lower() == "t":
            query = input("Enter text query >")
            filtered_log = search_log_text(log, query)
            view_entries(filtered_log)
        elif choice.lower() == "i":
            query = input("Enter file path for image search >")
            filtered_log = search_log_image(log, query)
            view_entries(filtered_log)
        elif choice.lower() == "m":
            view_metadata(log)
        elif choice.lower() == "q":
            return
        else:
            print("Invalid option")

# for testing
def test_image_search():
    try:
        log = torch.load(SAVE_PATH)
        print(f"loaded log from {SAVE_PATH}")
    except:
        print("Error: couldn't load log")
    
    query = "C:\\Users\\Lenovo\\Documents\\Lightshot\\VLM Testing\\chrome.png"
    results = search_log_image(log, query)
    print(len(results))

main()
#test_image_search()
print("Exiting...")
