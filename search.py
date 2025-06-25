from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPVisionModel
import torch

HF_HOME = "D:\\Downloads\\clip-vit-large-patch14"
SAVE_PATH = "desktop_log.pt"
SIMILARITY_MINIMUM = 0.75

# obtain vector embedding for an image
def get_image_embedding(image):
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=HF_HOME)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=HF_HOME)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[0]
    return embedding

# filter entries by text match
def search_log_text(log, query):
    filtered_log = []
    for entry in log['entries']:
        if query in entry['caption']:
            filtered_log.append(entry)
    return filtered_log

# filter entries by similarity score
def search_log_image(log, query):
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)
    image = Image.open(query)
    q_embedding = get_image_embedding(image)
    #print(len(q_embedding[0]))
    filtered_log = []
    for entry in log['entries']:
        # compute similarity between query embedding and entry embedding
        similarity = cosine_similarity(q_embedding[0], entry['embedding'][0]).item()
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

def main():
    try:
        log = torch.load(SAVE_PATH)
        print(f"loaded log from {SAVE_PATH}")
    except:
        print("Error: couldn't load log")
        
    while True:
        print("Options:")
        print("A - View all entries")
        print("T - Text search")
        print("I - Image search")
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
        elif choice.lower() == "q":
            return
        else:
            print("Invalid option")

'''
def test_image_search():
    log = torch.load(SAVE_PATH)
    query = "C:\\Users\\Lenovo\\Documents\\Lightshot\\VLM Testing\\chrome.png"
    results = search_log_image(log, query)
    print(len(results))
'''

main()
#test_image_search()
print("Exiting...")
