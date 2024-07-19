'''
MIT License

Copyright (c) 2023 Ulster University (https://www.ulster.ac.uk).
Project: Harmony (https://harmonydata.ac.uk)
Maintainer: Thomas Wood (https://fastdatascience.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os
import pickle as pkl
import re

import sklearn_crfsuite

from feature_extraction import convert_text_to_features

text_folder = "pdf-questionnaire-extraction-private/annotated-files"

print("Loading annotated training files")
file_to_text = {}
for file in os.listdir(text_folder):
    with open(text_folder + "/" + file, "r", encoding="utf-8") as f:
        content = f.read()
    file_to_text[file] = content

X_text = []
X = []
y = []

BIG_NUMBER = 9999999

for file, text_with_markup in file_to_text.items():

    text = text_with_markup

    text_no_tags = re.sub(r'</?q>', '', text)
    orig_char_are_questions = [0] * len(text_no_tags)

    index_to_tag_type = {}

    for j in range(1000):
        next_opening_tag_char_idx = BIG_NUMBER
        next_closing_tag_char_idx = BIG_NUMBER
        if "<q>" in text:
            next_opening_tag_char_idx = text.index("<q>")
        if "</q>" in text:
            next_closing_tag_char_idx = text.index("</q>")

        next_tag_char_idx = BIG_NUMBER
        next_tag_type = None
        if next_opening_tag_char_idx < next_closing_tag_char_idx:
            next_tag_type = "open"
            next_tag_char_idx = next_opening_tag_char_idx
        elif next_closing_tag_char_idx < next_opening_tag_char_idx:
            next_tag_type = "close"
            next_tag_char_idx = next_closing_tag_char_idx

        if next_tag_char_idx < BIG_NUMBER:
            index_to_tag_type[next_tag_char_idx] = next_tag_type
            if next_tag_type == "open":
                text = text[:next_tag_char_idx] + text[next_tag_char_idx + 3:]
            else:
                text = text[:next_tag_char_idx] + text[next_tag_char_idx + 4:]
        else:
            break

    for char_index, tag_type in list(index_to_tag_type.items()):
        if tag_type == "open":
            for i in range(char_index, len(orig_char_are_questions)):
                if i in index_to_tag_type and index_to_tag_type[i] == "close":
                    break
                orig_char_are_questions[i] = 1

    token_texts, token_start_char_indices, token_end_char_indices, token_properties = convert_text_to_features(
        text_no_tags)

    ground_truths = ["O"] * len(token_texts)
    last_token_category = "O"
    for token_idx in range(len(token_texts)):
        token_category = "O"
        token_start_char_idx = token_start_char_indices[token_idx]
        token_end_char_idx = token_end_char_indices[token_idx]
        if orig_char_are_questions[token_start_char_idx] or orig_char_are_questions[token_end_char_idx]:
            token_category = "I"

        for j in range(token_start_char_idx - 1, token_end_char_idx):
            if j >= 0 and j < len(index_to_tag_type) and index_to_tag_type.get(j) == "open":
                token_category = "B"

        if token_category == "I" and last_token_category == "O":
            token_category = "B"

        last_token_category = token_category

        ground_truths[token_idx] = token_category

    X_text.append(token_texts)
    X.append(token_properties)
    y.append(ground_truths)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=1000,
    all_possible_transitions=True
)
crf.fit(X, y)

with open("20240719_pdf_question_extraction_sklearn_crf_model.pkl", "wb") as f:
    pkl.dump(crf, f)

y_pred = crf.predict(X)

for doc_no in range(len(y_pred)):
    for idx in range(len(X[doc_no])):

        if y_pred[doc_no][idx] != "O":
            print(y_pred[doc_no][idx], X_text[doc_no][idx])
