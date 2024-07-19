import pickle as pkl
import re
import sys

from feature_extraction import convert_text_to_features

with open("20240719_pdf_question_extraction_sklearn_crf_model.pkl", "rb") as f:
    crf_text_model = pkl.load(f)


def predict(test_text):
    token_texts, token_start_char_indices, token_end_char_indices, token_properties = convert_text_to_features(
        test_text)

    X = []
    X.append(token_properties)

    y_pred = crf_text_model.predict(X)

    questions_from_text = []

    tokens_already_used = set()

    last_token_category = "O"

    for idx in range(len(X[0])):

        if y_pred[0][idx] != "O" and idx not in tokens_already_used:
            if last_token_category == "O" or y_pred[0][idx] == "B":
                start_idx = token_start_char_indices[idx]
                end_idx = len(test_text)
                for j in range(idx + 1, len(X[0])):
                    if y_pred[0][j] == "O" or y_pred[0][j] == "B":
                        end_idx = token_end_char_indices[j - 1]
                        break
                    tokens_already_used.add(j)

                question_text = test_text[start_idx:end_idx]
                question_text = re.sub(r'\s+', ' ', question_text)
                question_text = question_text.strip()
                questions_from_text.append(question_text)

        last_token_category = y_pred[0][idx]

    return questions_from_text


if __name__ == "__main__":
    input_file_name = sys.argv[1]

    print(input_file_name)

    with open(input_file_name, "r", encoding="utf-8") as f:
        test_text = f.read()

    questions_from_text = predict(test_text)

    for q in questions_from_text:
        print(q)
