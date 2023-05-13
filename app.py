import streamlit as st
import streamlit.components.v1 as components

from malayalam_ner import MalayalamNER

st.title("Named Entity Recognition In Malayalam")

components.html(
    """<iframe src="https://ghbtns.com/github-btn.html?user=amankshihab&repo=TENER-MALAYALAM&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>"""
)


model_name = st.selectbox(
    'Which model would you like to use?',
    ('TENER', 'BiLSTM')
)

input_string = st.text_input("Enter your sentence", placeholder="Text in Malayalam goes here")

if len(input_string) > 0:
    model = MalayalamNER(
        model_name=model_name,
        path_to_weights='./pretrained_weights/'
    )

    outputs = model.predict(input_string=input_string)
    output_dict = {
        "Tokens" : [],
        "Tags" : [],
        "Confidence" : []
    }

    for op in outputs[1:-1]:
        output_dict["Tokens"].append(op[0])
        output_dict["Tags"].append(op[1])
        output_dict["Confidence"].append(op[2])
    
    st.table(output_dict)