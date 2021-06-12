import bz2
from keybert import KeyBERT
import numpy as np
import pandas as pd
import transformers as ppb


def create_file_from_bz2(infile, outfile, folderpath="../../data/amazon/"):
    print("Generating BZ2File")
    new_file = bz2.BZ2File(folderpath + infile)

    print("Obtaining File Lines")
    new_file_lines = new_file.readlines()

    del new_file

    print("Decoding File Lines")
    new_file_lines = [x.decode('utf-8') for x in new_file_lines]

    with open(folderpath + outfile, "w", encoding="utf-8") as f:
        f.writelines(new_file_lines)


def format_dataset(infile, filepath="../../data/amazon/"):
    # Data Headers
    data = []

    print("Opening File")
    with open(filepath + infile, "r", encoding="utf-8") as f:
        print("Formatting Label and Comments")
        for line in f:
            new_data = []

            label, line = line.split(" ", 1)
            new_data.append(0 if label == "__label__1" else 1)

            # Format Comment Line
            line = line.replace("&", "and").lower()

            new_data.append(line)
            data.append(new_data)

    print("Returning DataFrame")
    return pd.DataFrame(data, columns=["Label", "Comments"])


# create_file_from_bz2("train.ft.txt.bz2", "train.txt")
# create_file_from_bz2("test.ft.txt.bz2", "test.txt")

data = format_dataset("train.txt", "../../data/amazon/")
print(data["Comments"][0])

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                    ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = data["Comments"][:10].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
max_token_len = 0
arr = []
for tokens in tokenized:
    arr.append(list(tokens))
    if max_token_len < len(tokens):
        max_token_len = len(tokens)

for i in range(len(arr)):
    arr[i] += [0 for _ in range(max_token_len - len(arr[i]))]
arr = np.array(arr)
