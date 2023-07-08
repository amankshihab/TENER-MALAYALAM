import streamlit as st
import streamlit.components.v1 as components

from malayalam_ner import MalayalamNER

from collections import Counter


st.title("Named Entity Recognition In Malayalam")

components.html(
    """<iframe src="https://ghbtns.com/github-btn.html?user=amankshihab&repo=TENER-MALAYALAM&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>"""
)

model_name = st.selectbox(
    'Which model would you like to use?',
    ('TENER', 'BiLSTM')
)

model = MalayalamNER(
        model_name=model_name,
        path_to_weights='./pretrained_weights/'
    )

ids_to_tags = {
            0 : 'O',
            1 : 'B-PER',
            2 : 'I-PER',
            3 : 'B-ORG',
            4 : 'I-ORG',
            5 : 'B-LOC',
            6 : 'I-LOC'
        }

def _parse_predictions(output):
    meaningful_words = []
    current_word = ""
    current_tag = None

    for word, tag_id, _ in output:
        if word.startswith("▁"):
            if current_word:
                meaningful_words.append((current_word, current_tag))
            current_word = word[1:]
            current_tag = tag_id
        else:
            current_word += word

    if current_word:
        meaningful_words.append((current_word, current_tag))

    results = []
    for word, tag_id in meaningful_words:
        tags = [tag_id]
        for _, next_tag_id, _ in output:
            if next_tag_id == tag_id:
                tags.append(tag_id)
        most_common_tag = Counter(tags).most_common(1)[0][0]
        if most_common_tag and most_common_tag != 0:
            results.append((word, ids_to_tags[most_common_tag.item()]))

    return results


def parse_predictions(text: str):
    named_entities = []
    for t in text.split('.'):
        output = model.predict(t)
        named_entities.extend(_parse_predictions(output))
    return named_entities


input_string = st.text_area("Enter your sentence", placeholder="Text in Malayalam goes here")

if len(input_string) > 0:
    # outputs = model.predict(input_string=input_string)
    outputs = parse_predictions(input_string)
    output_dict = {
        "Tokens" : [],
        "Tags" : []
    }

    for op in outputs[1:-1]:
        output_dict["Tokens"].append(op[0])
        output_dict["Tags"].append(op[1])
        # output_dict["Confidence"].append(op[2])
    
    st.table(output_dict)