

def log(*texts, ext = "", delimiter = " "):
    with open(f"log{ext}.txt",'a') as file:
        file.write(delimiter.join([f"{t}" for t in texts])+"\n")