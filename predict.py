from malayalam_ner import MalayalamNER


model = MalayalamNER(
    model_name='tener',
    path_to_weights='path-to-weights', # Check README.MD
)

print(
    model.predict(
        "വര്‍ഷങ്ങൾക്ക് മുമ്പ് മണിയൻ പിള്ള രാജുവും കൊച്ചിൻ ഹനീഫയും സിനിമയിൽ അവസരം തേടി നടക്കുന്ന സമയം."
    )
)