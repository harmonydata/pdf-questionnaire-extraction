import os, re

MODEL_NO = "00"

output_folder = "output/" + MODEL_NO

try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

for file_name in os.listdir("../data/annotations/"):
    orig_file = "../data/preprocessed_text/" + file_name
    with open(orig_file, "r", encoding="utf-8") as f:
        with open(f"output/{MODEL_NO}/" + file_name, "w", encoding="utf-8") as fo:
            for l in f:
                text = re.sub(r'\s+', ' ', l).strip()
                text = re.sub(r'\n+', '', text)
                fo.write(f"\t{text}\t\n")

