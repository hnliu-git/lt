"""

From span tagging to list tagging

Span:
{'text': 'I love Amsterdam', 'spans': [{'start': 7, 'end': 16, 'tag': 'loc'}]}

List:
{'text': ['I', 'love', 'Amsterdam'], 'tags': ['O', 'O', 'LOC']}

"""
import json

from utils import vac_tags_dict

def list_all_overlap_span(spans):
    overlapping_spans = []
    for i, span1 in enumerate(spans):
        for span2 in spans[i+1:]:
            if span1['start'] < span2['end'] and span1['end'] > span2['start']:
                overlapping_spans.append((span1, span2))
    return overlapping_spans

def span2list(doc):

    text = doc['text']
    spans = doc['spans']

    words = text.split(' ')
    tags = ['-' for _ in range(len(words))]

    start = 0

    for i, word in enumerate(words):
        end = start + len(word)

        for span in spans:
            if span['start'] < end and span['end'] > start:
                tags[i] = span['label']
                validate = False
                for tk in span['text']:
                    if tk in word:
                        validate = True
                assert validate
        start = end + 1
    
    if words[0] == "":
        words.pop(0)
        tags.pop(0)

    prev_tag = '-'
    for i, tag in enumerate(tags):
        if tag != prev_tag and tag != '-':
            tags[i] = 'open' + tag
        prev_tag = tag

    for i, tag in enumerate(tags):
        tags[i] = vac_tags_dict[tag]

    output = {
        'tokens': words,
        'ner_tags': tags,

    }

    return output

if __name__ == '__main__':
    
    jsonl_path = 'vac-nl-xmls/devel.jsonl'
    output = 'data/vac-nl-orig/devel.jsonl'
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _id, line in enumerate(f.readlines()):
            doc = json.loads(line)
            print(doc['fname'])
            # if doc['fname'] == '00119.xml':
                # print('Stop')
            record = span2list(doc)
            record['id'] = _id
            records.append(record)
    with open (output, 'w', encoding='utf8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")