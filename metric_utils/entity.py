# *************************************************************************
# *
# * Textkernel CONFIDENTIAL
# * __________________
# *
# *  Textkernel BV
# *  All Rights Reserved.
# *
# * NOTICE:  All information contained herein is, and remains
# * the property of Textkernel BV and its suppliers,
# * if any.  The intellectual and technical concepts contained
# * herein are proprietary to Textkernel BV
# * and its suppliers and may be covered by U.S. and Foreign Patents,
# * patents in process, and are protected by trade secret or copyright law.
# * Dissemination of this information or reproduction of this material
# * is strictly forbidden unless prior written permission is obtained
# * from Textkernel BV.
# *

from collections import defaultdict, namedtuple


class Span(namedtuple('Span', ['start', 'stop'])):

    def __len__(self):
        return self[1] - self[0]


def overlaps(span_A, span_B):
    overlaps = span_A[0] < span_B[1] and span_B[0] < span_A[1]
    return overlaps


def get_overlap_pairs(spans_A, spans_B):
    overlap_pairs = []
    for span_A in spans_A:
        for span_B in spans_B:
            if overlaps(span_A, span_B):
                pair = (span_A, span_B)
                overlap_pairs.append(pair)
    return overlap_pairs


def intersection(span_A, span_B):
    intersection = None
    if overlaps(span_A, span_B):
        start = max(span_A[0], span_B[0])
        stop = min(span_A[1], span_B[1])
        intersection = Span(start, stop)
    return intersection


def union(span_A, span_B):
    union = None
    if overlaps(span_A, span_B):
        start = min(span_A[0], span_B[0])
        stop = max(span_A[1], span_B[1])
        union = Span(start, stop)
    return union


def intersection_over_union(span_A, span_B):
    i = intersection(span_A, span_B)
    u = union(span_A, span_B)
    if i and u:
        iou = len(i) / len(u)
    else:
        iou = 0.0
    return iou


def align_spans(spans_A, spans_B):
    pairs = get_overlap_pairs(spans_A, spans_B)
    # If we have overlapping pairs with same I/U we want the one with the
    # smallest union
    pairs.sort(key=lambda pair: (intersection_over_union(
        *pair), -len(union(*pair))), reverse=True)

    assigned_spans_A, assigned_spans_B = set(), set()
    aligned_spans = []
    for pair in pairs:
        if pair[0] not in assigned_spans_A and pair[1] not in assigned_spans_B:
            aligned_spans.append(pair)
            assigned_spans_A.add(pair[0])
            assigned_spans_B.add(pair[1])
    return aligned_spans


class EntitySetCreator:

    def __init__(self, tagdict):
        self.tagdict = tagdict

    def transform_labels(self, labels):
        '''Transforms sequence of labels to dictionary mapping entity names to
        list of (start_idx,stop_idx) tuples'''

        entities = []
        current_entity = {}
        for i, tag_id in enumerate(labels):
            tag = self.tagdict.id_to_tag[tag_id]
            ent_name = self.tagdict.tag_to_entity.get(tag, '')

            # extend current_entity
            if tag not in self.tagdict.b_tags and \
               ent_name == current_entity.get('name'):
                current_entity['stop'] = i + 1
            else:
                # finalize current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = {}
                # start new entity
                if tag != self.tagdict.NONE_LABEL:
                    current_entity['name'] = ent_name
                    current_entity['start'] = i
                    current_entity['stop'] = i + 1
        # add last entity
        if current_entity:
            entities.append(current_entity)

        # write results to entity dictionary
        entities_dict = defaultdict(list)
        for ent in entities:
            name, coords = ent['name'], Span(ent['start'], ent['stop'])
            entities_dict[name].append(coords)

        return entities_dict
