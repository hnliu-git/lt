from metric_utils.decorators import load_resource_file_once

class TagDict:
    NONE_LABEL = "-"
    B_PREFIX = 'open'

    def __init__(self, tag_to_id):
        self.tag_to_id = tag_to_id
        self.id_to_tag = dict((v, k) for k, v in self.tag_to_id.items())
        self.tags = self.tag_to_id.keys()
        self.n_tags = len(self.tag_to_id)
        self.b_tags = set(
            tag for tag in self.tags if tag.startswith(
                self.B_PREFIX))
        self.i_tags = self.tags - self.b_tags - \
            {self.NONE_LABEL}
        self.tag_to_entity = self._tag_to_entity()
        self.entity_names = set(self.tag_to_entity.values())

    @staticmethod
    @load_resource_file_once
    def from_file(filename):
        with open(filename, "r", encoding="utf-8") as f:
            tags = f.read().split("\n")
        tag_to_id = {}
        for i, tag in enumerate(tags):
            if tag:
                tag_to_id[tag] = i
        return TagDict(tag_to_id)

    def _tag_to_entity(self):
        tag_to_entity = {}
        for tag in self.b_tags:
            tag_to_entity[tag] = tag[len(self.B_PREFIX):]
        for tag in self.i_tags:
            tag_to_entity[tag] = tag
        return tag_to_entity
