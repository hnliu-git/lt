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

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from enum import Enum
from math import isclose

import numpy as np

from metric_utils.data_utils import TagDict
from metric_utils.entity import EntitySetCreator, align_spans, intersection


class Metrics(Enum):
    accuracy = 'accuracy'
    mean_accuracy = 'mean_accuracy'
    entity_score = 'entity_score'
    mean_entity_score = 'mean_entity_score'
    entity_overlap = 'entity_overlap'
    mean_entity_overlap = 'mean_entity_overlap'


def initialize_metrics(metric_names, tags=None, main_entities=None):
    """_summary_

    Args:
        metric_names (str): default
        tags (TagDict, optional): entity id mapping. Defaults to None.
        main_entities (list, optional): list of main entity. Defaults to None.

    Returns:
        MetricRunner: A runner that contains multiple tasks
    """
    if main_entities:
        """
            Setup a entity weight dict and set main entities to 1
        """
        if tags is None:
            raise ValueError(
                "A TagDict has to be supplied when using main_entities option")
        main_entities = tags.entity_names & set(main_entities)
        if main_entities == {}:
            raise ValueError("Main entities don't overlap with tags")
        entity_weights = dict((tag, 0) for tag in tags.entity_names)
        for ent in main_entities:
            entity_weights[ent] = 1
    else:
        entity_weights = None

    if metric_names == "default":
        metrics = [Metrics.accuracy, Metrics.entity_overlap]
    elif metric_names == "all":
        metrics = [m for m in Metrics]
    else:
        metrics = [Metrics(n) for n in metric_names]

    # always add accuracy as it's used in train loop
    if Metrics.accuracy not in metrics:
        metrics.append(Metrics.accuracy)

    metric_runner = MetricRunner()
    for metric in metrics:
        if metric == Metrics.accuracy:
            metric_func = accuracy
            mean_over_docs = False
        if metric == Metrics.mean_accuracy:
            metric_func = accuracy
            mean_over_docs = True
        if metric == Metrics.entity_score:
            metric_func = EntityScore(tags, entity_weights=entity_weights)
            mean_over_docs = False
        if metric == Metrics.mean_entity_score:
            metric_func = EntityScore(tags, entity_weights=entity_weights)
            mean_over_docs = True
        if metric == Metrics.entity_overlap:
            metric_func = EntityOverlapScore(
                tags, entity_weights=entity_weights)
            mean_over_docs = False
        if metric == Metrics.mean_entity_overlap:
            metric_func = EntityOverlapScore(
                tags, entity_weights=entity_weights)
            mean_over_docs = True
        metric_runner.add(metric.value, metric_func, mean_over_docs)

    return metric_runner


def accuracy(predictions, gold):
    n_total, n_correct = 0, 0
    for predicted_labels, gold_labels in zip(predictions, gold):
        n_total += len(gold_labels)
        n_correct += np.sum(predicted_labels == gold_labels)
    if n_total == 0:
        acc = None
    else:
        acc = n_correct / n_total
    return acc


def p_r_f1(n_correct, n_predicted, n_gold):
    if n_gold == 0 and n_predicted == 0:
        precision = recall = f1 = None
    elif n_correct == 0:
        precision = recall = f1 = 0
    else:
        precision = n_correct / n_predicted
        recall = n_correct / n_gold
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision,
            'recall': recall,
            'f1': f1}


class EntityScoreBase:

    def __init__(self, tagdict, entity_weights=None):
        if not isinstance(tagdict, TagDict):
            raise ValueError(
                "Entity metrics require a TagDict to be supplied.")
        self.tagdict = tagdict
        self.entityset_creator = EntitySetCreator(tagdict)
        self.entity_names = self.tagdict.entity_names
        if entity_weights is not None and \
           entity_weights.keys() != self.entity_names:
            raise ValueError(
                "Keys for entity_weights don't match entity_names: {} != {}".format(  # noqa
                    entity_weights.keys(), self.entity_names))
        self.entity_weights = entity_weights

    def finalize(self, result):
        if self.entity_weights is None:
            weights = np.ones(len(self.entity_names))
        else:
            weights = [self.entity_weights[ent_name]
                       for ent_name in self.entity_names]
        for metric in 'precision', 'recall', 'f1':
            values = np.array([result[ent_name + '/' + metric]
                               for ent_name in self.entity_names], dtype=float)
            values = np.ma.array(values, mask=np.isnan(values), dtype=float)
            result['avg/' + metric] = np.ma.average(values, weights=weights)
        return result


class EntityScore(EntityScoreBase):

    def __call__(self, predictions, gold):
        result = {
            k: defaultdict(int) for k in [
                'n_correct',
                'n_gold',
                'n_predicted']}
        for predicted_labels, gold_labels in zip(predictions, gold):
            gold_entities = self.entityset_creator.transform_labels(
                gold_labels)
            predicted_entities = self.entityset_creator.transform_labels(
                predicted_labels)
            for ent_name in self.entity_names:
                for ent in gold_entities[ent_name]:
                    if ent in predicted_entities[ent_name]:
                        result['n_correct'][ent_name] += 1
                result['n_gold'][ent_name] += len(gold_entities[ent_name])
                result['n_predicted'][ent_name] += len(
                    predicted_entities[ent_name])

        scores = {}
        for ent_name in self.entity_names:
            n_correct = result['n_correct'][ent_name]
            n_gold = result['n_gold'][ent_name]
            n_predicted = result['n_predicted'][ent_name]
            ent_scores = p_r_f1(n_correct, n_predicted, n_gold)
            scores[ent_name + '/precision'] = ent_scores['precision']
            scores[ent_name + '/recall'] = ent_scores['recall']
            scores[ent_name + '/f1'] = ent_scores['f1']
        return scores


class EntityOverlapScore(EntityScoreBase):

    def __call__(self, predictions, gold):
        result = {
            k: defaultdict(int) for k in [
                'partial_precision',
                'partial_recall',
                'n_gold',
                'n_predicted']}
        for predicted_labels, gold_labels in zip(predictions, gold):
            gold_entities = self.entityset_creator.transform_labels(
                gold_labels)
            predicted_entities = self.entityset_creator.transform_labels(
                predicted_labels)
            for ent_name in self.entity_names:
                entity_pairs = align_spans(
                    predicted_entities[ent_name],
                    gold_entities[ent_name])
                intersections = np.array(
                    [len(intersection(*pair)) for pair in entity_pairs],
                    dtype=np.float)
                gold_lengths = np.array([len(gold)
                                         for _, gold in entity_pairs])
                pred_lengths = np.array([len(pred)
                                         for pred, _ in entity_pairs])
                result['partial_recall'][ent_name] += sum(
                    intersections / gold_lengths)
                result['partial_precision'][ent_name] += sum(
                    intersections / pred_lengths)
                result['n_gold'][ent_name] += len(gold_entities[ent_name])
                result['n_predicted'][ent_name] += len(
                    predicted_entities[ent_name])

        scores = {}
        for ent_name in self.entity_names:
            partial_precision = result['partial_precision'][ent_name]
            partial_recall = result['partial_recall'][ent_name]
            n_gold = result['n_gold'][ent_name]
            n_predicted = result['n_predicted'][ent_name]

            if n_gold == 0 and n_predicted == 0:
                precision = recall = f1 = None
            elif n_gold == 0 or n_predicted == 0:
                precision = recall = f1 = 0
            else:
                precision = partial_precision / n_predicted
                recall = partial_recall / n_gold
                if isclose(precision + recall, 0):
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

            scores[ent_name + '/precision'] = precision
            scores[ent_name + '/recall'] = recall
            scores[ent_name + '/f1'] = f1
        return scores


MetricDefinition = namedtuple(
    'MetricDefinition', [
        'name', 'function', 'mean_over_docs'])


class MetricRunner:
    """
    Metric Wrapper to compute metrics

    
    """
    def __init__(self):
        self.metrics = []

    def add(self, name, function, mean_over_docs=False):
        metric = MetricDefinition(name, function, mean_over_docs)
        self.metrics.append(metric)

    def compute(self, predictions, gold):
        assert len(predictions) == len(gold), \
            "Error in metric computation: Predicted and gold differ in number of docs"  # noqa

        metric_values = {}
        for metric in self.metrics:
            if metric.mean_over_docs:
                # calculate metric seperately for each doc
                results = defaultdict(list)
                for predicted_labels, gold_labels in zip(predictions, gold):
                    doc_result = metric.function(
                        [predicted_labels], [gold_labels])
                    if isinstance(doc_result, dict):
                        for key in doc_result:
                            results[key].append(doc_result[key])
                    else:
                        results['_'].append(doc_result)
                # take mean over docs
                if set(results.keys()) == {'_'}:
                    result = np.nanmean(np.array(results['_'], dtype=float))
                else:
                    result = {}
                    for key in results:
                        if all(v is None for v in results[key]):
                            result[key] = None
                        else:
                            result[key] = np.nanmean(
                                np.array(results[key], dtype=float))
            else:
                result = metric.function(predictions, gold)

            if hasattr(metric.function, 'finalize'):
                result = metric.function.finalize(result)

            if isinstance(result, dict):
                for k, v in result.items():
                    metric_values["{}/{}".format(metric.name, k)] = v
            else:
                metric_values[metric.name] = result
        return metric_values

if __name__ == '__main__':

    metrics = initialize_metrics(
        metric_names=['accuracy', 'entity_score', 'entity_overlap'],
        tags=TagDict.from_file('vac-phrases-full-tags.txt'),
        main_entities=open('main_ents.txt').read().splitlines()
    )

    results = metrics.compute([[0, 1]], [[0, 0]])