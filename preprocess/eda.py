import os
import json
import plotly.graph_objects as go

from utils import vac_tags, vac_tags_dict

data_folder = 'data/vac-nl-no-norm'
tag_list = [k for k in vac_tags_dict.keys()]

ctrs = []

for json_file in ['train.jsonl', 'train-m.jsonl']:
    json_path = os.path.join(data_folder, json_file)
    with open(json_path, 'r') as json_fr:
        json_list = json_fr.readlines()
        counter = {tag:0 for tag in vac_tags}
        for json_str in json_list:
            json_obj = json.loads(json_str)
            for tag in json_obj['ner_tags']:
                if tag != 0:
                    counter[tag_list[tag].replace('open', '')] += 1

        ctrs.append(counter)

# Create a bar chart
fig = go.Figure([
    go.Bar(x=list(ctrs[0].keys()), y=list(ctrs[0].values()), name='dutch'),
    go.Bar(x=list(ctrs[1].keys()), y=list(ctrs[1].values()), name='dutch + german'),
])

# Update chart layout
fig.update_layout(
    title='Dutch data Tag Distribution',
    xaxis_title='Tag Type',
    yaxis_title='Number'
)

# Show chart
fig.show()