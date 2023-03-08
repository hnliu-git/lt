import os
import re
import json
import xml.etree.ElementTree as ET

from argparse import ArgumentParser

"""
Issues observed in the data

1. Fully overlapping annotation, only one tag can survice
2. Empty or annotation with only whitespaces (resolved by both processer)
3. Xmls are hard to process
4. Documents without any annotations

"""

vac_tags = [
    "vac_job_title","vac_num_positions","vac_location","org_name","org_num_employees","org_industry","prof_education",
    "prof_experience","prof_languages","prof_drivers_license","prof_computer_skills","prof_competence","cond_working_hours",
    "cond_hours_per_week","cond_employment_type","cond_contract_type","salary","org_contact_person","org_contact_person_function",
    "org_address","org_phone","org_fax","org_email","org_website","vac_ref_no","vac_posted_date","vac_apply_date","vac_start_date"
]


def parse_args():
    parser = ArgumentParser(description="Create jsonl data from annotated XML documents")
    parser.add_argument("xml_dir", help="Directory of input XMLs")
    parser.add_argument("output", help="Output jsonl file")
    return parser.parse_args()

def clean_text(text):
    # remove multiple space
    text = re.sub(r'[ \t]+', ' ', text)
    # remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    return text

def element2cleantext(element):
    return clean_text("".join(element.itertext()))

def element2text(element):
    return ("".join(element.itertext()))

def text_before_target(root, target):
    """
    Find text before the target in the root xml tree
    TODO Optimize speed and memory
    """
    global acc_text
    
    if root == target:
        return True

    if root.text:
        acc_text += root.text

    for child in root:
        if text_before_target(child, target):
            return True
        if child.tail:
            acc_text += child.tail

    return False

def validate_spans_in_order(spans):
    """
    'start' should be in ascending order
    """
    start = -1
    for span in spans:
        assert span['start'] >= start
        start = span['start']


def get_spans(vac):
    """
    Get span of all the entities in the vacancy
    """
    global acc_text

    spans = []
    # The cleaned vac text
    vac_text = element2cleantext(vac)
    # the tag is iterated in order
    for tag in vac.iter():
        if tag.tag not in vac_tags:
            continue

        tag_text = element2cleantext(tag)
        # Empty or space annotations
        if tag_text.isspace() or tag_text == "":
            continue

        # Construct hook to reduce the chances of multiple match
        if tag.tail:
            hook = clean_text(element2text(tag) + tag.tail)
        else:
            hook = tag_text
        
        matches = list(re.finditer(re.escape(hook), vac_text))

        if len(matches) > 1:
            # Get pre_text
            acc_text = ""
            text_before_target(vac, tag)
            acc_text = clean_text(acc_text)
            # Validate pre_text
            assert acc_text == vac_text[:len(acc_text)]
            for match in matches:
                # The match should has the start lq to the pre_text
                if match.start() >= len(clean_text(acc_text)):
                    break
        else:
            match = matches[0]
        start, end = match.start(), match.start() + len(tag_text)

        spans.append({
            "start": start,
            "end": end,
            "label": tag.tag,
            "text": tag_text          
        })
    
    # spans should be in order
    validate_spans_in_order(spans)

    return spans

if __name__ == '__main__':
    # args = parse_args()
    xml_dir = 'vac-nl-xmls/test'
    output = 'vac-nl-xmls/test.jsonl'

    records = []
    for f in os.listdir(xml_dir):
        # print(f)
        
        tree = ET.parse(os.path.join(xml_dir, f))
        root = tree.getroot()
        vacancy = root.find('sec_vacancy')

        spans = get_spans(vacancy)
        # Skip document with no annotation
        if len(spans) == 0:
            continue

        records.append({
            'text': element2cleantext(vacancy),
            'spans': get_spans(vacancy),
            'fname': f
        })
    
    with open(output, 'w', encoding='utf8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")

