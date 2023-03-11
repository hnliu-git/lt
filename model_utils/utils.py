
def build_tags(fields, non_fields):
    tags = list()
    tags.append(non_fields)
    for field in fields:
        tags.append('open' + field)
        tags.append(field)
    return {tag: tag_id for tag_id, tag in enumerate(tags)}


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels_and_chunk(example, tokenizer, stride=0):

    tokenized_input_chunk = tokenizer(
        example['tokens'],
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        stride=stride,
        is_split_into_words=True
    )

    chunk_num = len(tokenized_input_chunk['overflow_to_sample_mapping'])

    label = example['ner_tags']
    labels = []

    for i in range(chunk_num):
        word_ids = tokenized_input_chunk.word_ids(i)
        previous_word_idx = None
        label_ids = []
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append(-100)
            # if still in the overlapping scope, set to invalid labels
            elif i > 0 and j <= stride:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
        
        
        labels.append(label_ids)

    tokenized_input_chunk['labels'] = labels
    tokenized_input_chunk.pop('overflow_to_sample_mapping')
    tokenized_input_chunk['chunk_num'] = chunk_num
    
    return tokenized_input_chunk


vac_tags = [
    "vac_job_title","vac_num_positions","vac_location","org_name","org_num_employees","org_industry","prof_education",
    "prof_experience","prof_languages","prof_drivers_license","prof_computer_skills","prof_competence","cond_working_hours",
    "cond_hours_per_week","cond_employment_type","cond_contract_type","salary","org_contact_person","org_contact_person_function",
    "org_address","org_phone","org_fax","org_email","org_website","vac_ref_no","vac_posted_date","vac_apply_date","vac_start_date"
]

main_vac_tags = [
    "vac_job_title", "vac_location", "org_name", "prof_education", "prof_experience","prof_languages", "cond_working_hours",
    "cond_hours_per_week", "cond_employment_type", "cond_contract_type", "salary", "org_contact_person", "org_address",
    "org_phone", "org_email", "org_website", "vac_ref_no"
]

non_tag = "-"

vac_tags_dict = build_tags(vac_tags, non_tag)
vac_main_dict = build_tags(main_vac_tags, non_tag)