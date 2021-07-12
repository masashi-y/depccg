#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pascual Martinez-Gomez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
from lxml import etree
from multiprocessing import Pool
from multiprocessing import Lock
import os
import sys

from nltk.sem.logic import LogicalExpressionException
from .ccg2lambda_tools import assign_semantics_to_ccg
from .semantic_index import SemanticIndex
from .logic_parser import lexpr
from .nltk2normal import remove_true

logger = logging.getLogger(__name__)

SEMANTIC_INDEX = None
GOLD_TREES = True
NBEST = 0
SENTENCES = None
kMaxTasksPerChild = None
lock = Lock()


def parse(ccg, templates, nbest=0, ncores=3):
    global SEMANTIC_INDEX
    global SENTENCES
    global NBEST
    NBEST = nbest

    if not os.path.exists(templates):
        print('File does not exist: {0}'.format(templates))
        sys.exit(1)

    logger.info(templates)
    SEMANTIC_INDEX = SemanticIndex(templates)
    SENTENCES = ccg.findall('.//sentence')
    print(SENTENCES)
    sentence_inds = range(len(SENTENCES))
    sem_nodes_lists = semantic_parse_sentences(sentence_inds, ncores)
    assert len(sem_nodes_lists) == len(SENTENCES), \
        'Element mismatch: {0} vs {1}'.format(
            len(sem_nodes_lists), len(SENTENCES))
    logging.info('Adding XML semantic nodes to sentences...')
    formulas_list = []
    for sentence, (sem_nodes, orig_formulas) in zip(SENTENCES, sem_nodes_lists):
        formulas = []
        for formula in orig_formulas:
            try:
                formulas.append(str(remove_true(lexpr(formula))))
            except LogicalExpressionException:
                formulas.append(formula)
        formulas_list.append(formulas)
        sentence.extend(sem_nodes)
    logging.info('Finished adding XML semantic nodes to sentences.')

    root_xml_str = serialize_tree(ccg)
    return root_xml_str, formulas_list


def semantic_parse_sentences(sentence_inds, ncores=1):
    if ncores <= 1:
        sem_nodes_lists = semantic_parse_sentences_seq(sentence_inds)
    else:
        sem_nodes_lists = semantic_parse_sentences_par(sentence_inds, ncores)
    results = [([etree.fromstring(s) for s in sem_nodes], formulas)
               for sem_nodes, formulas in sem_nodes_lists]
    return results


def semantic_parse_sentences_par(sentence_inds, ncores=3):
    pool = Pool(processes=ncores, maxtasksperchild=kMaxTasksPerChild)
    results = pool.map(semantic_parse_sentence, sentence_inds)
    pool.close()
    pool.join()
    return results


def semantic_parse_sentences_seq(sentence_inds):
    results = []
    for sentence_ind in sentence_inds:
        result = semantic_parse_sentence(sentence_ind)
        results.append(result)
    return results


def semantic_parse_sentence(sentence_ind):
    """
    `sentence` is an lxml tree with tokens and ccg nodes.
    It returns an lxml semantics node.
    """
    global lock
    sentence = SENTENCES[sentence_ind]
    sem_nodes = []
    formulas = []

    tree_indices = [int(sentence.get('gold_tree', '0')) + 1]
    if NBEST != 1:
        tree_indices = get_tree_indices(sentence, NBEST)
    for tree_index in tree_indices:
        sem_node = etree.Element('semantics')
        try:
            sem_tree = assign_semantics_to_ccg(
                sentence, SEMANTIC_INDEX, tree_index)
            filter_attributes(sem_tree)
            sem_node.extend(sem_tree.xpath('.//descendant-or-self::span'))
            sem_node.set('status', 'success')
            sem_node.set('ccg_id',
                         sentence.xpath('./ccg[{0}]/@id'.format(tree_index))[0])
            sem_node.set('root',
                         sentence.xpath('./ccg[{0}]/@root'.format(tree_index))[0])
            formulas.append(sem_tree.attrib['sem'])
        except Exception as e:
            sem_node.set('status', 'failed')
            # from pudb import set_trace; set_trace()
            sentence_surf = ' '.join(sentence.xpath('tokens/token/@surf'))
            lock.acquire()
            logging.error('An error occurred: {0}\nSentence: {1}\nTree XML:\n{2}'.format(
                e, sentence_surf,
                etree.tostring(sentence, encoding='utf-8', pretty_print=True).decode('utf-8')))
            lock.release()
            # print('x', end='', file=sys.stdout)
            formulas.append('FAILED!')
        sem_nodes.append(sem_node)
    sem_nodes = [etree.tostring(sem_node) for sem_node in sem_nodes]
    return sem_nodes, formulas


def get_tree_indices(sentence, nbest):
    num_ccg_trees = int(sentence.xpath('count(./ccg)'))
    if nbest < 1:
        nbest = num_ccg_trees
    return list(range(1, min(nbest, num_ccg_trees) + 1))


keep_attributes = set(['id', 'child', 'sem', 'type'])


def filter_attributes(tree):
    if 'coq_type' in tree.attrib and 'child' not in tree.attrib:
        sem_type = \
            tree.attrib['coq_type'].lstrip('["Parameter ').rstrip('."]')
        if sem_type:
            tree.attrib['type'] = sem_type
    attrib_to_delete = [
        a for a in tree.attrib.keys() if a not in keep_attributes]
    for a in attrib_to_delete:
        del tree.attrib[a]
    for child in tree:
        filter_attributes(child)
    return


def serialize_tree(tree):
    tree_str = etree.tostring(
        tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
    return tree_str
