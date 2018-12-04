
from __future__ import print_function

from libc.stdlib cimport malloc, free

cimport numpy as np
import numpy as np
import sys
import re
import tarfile
import os
import json
import chainer
import logging

from .combinator cimport en_headfirst_binary_rules, ja_headfinal_binary_rules
from .utils cimport *
from .utils import maybe_split_and_join
from .cat cimport Category


logger = logging.getLogger(__name__)


cdef class EnglishCCGParser:
    cdef unordered_map[string, vector[bool]] category_dict_
    cdef object py_category_dict
    cdef vector[Cat] tag_list_
    cdef float beta_
    cdef bint use_beta_
    cdef unsigned pruning_size_
    cdef unsigned nbest_
    cdef unordered_set[Cat] possible_root_cats_
    cdef unordered_map[Cat, vector[Cat]] unary_rules_
    cdef vector[Op] binary_rules_
    cdef unordered_map[CatPair, vector[RuleCache]] cache_
    cdef unordered_set[CatPair] seen_rules_
    cdef ApplyBinaryRules apply_binary_rules_
    cdef ApplyUnaryRules apply_unary_rules_
    cdef unsigned max_length_
    cdef unsigned loglevel
    cdef object tagger
    cdef bytes lang

    def __init__(self,
                 category_dict,
                 tag_list,
                 unary_rules,
                 # binary_rules,
                 seen_rules,
                 beta=0.00001,
                 use_beta=True,
                 pruning_size=50,
                 nbest=1,
                 possible_root_cats=None,
                 max_length=250,
                 loglevel=3):

        if possible_root_cats is None:
            possible_root_cats = ['S[dcl]', 'S[wq]', 'S[q]', 'S[qem]', 'NP']
        possible_root_cats = [Category.parse(cat) if not isinstance(cat, Category) else cat
                              for cat in possible_root_cats]

        logger.info(f'beta value = {beta} (use beta = {use_beta})')
        logger.info(f'pruning size = {pruning_size}')
        logger.info(f'N best = {nbest}')
        logger.info(f'allow at the root of a tree only categories in {possible_root_cats}'),
        logger.info(f'give up sentences that contain > {max_length} words')

        self.py_category_dict = category_dict
        self.category_dict_ = convert_cat_dict(category_dict, tag_list)
        self.tag_list_ = cat_list_to_vector(tag_list)
        self.beta_ = beta
        self.use_beta_ = use_beta
        self.pruning_size_ = pruning_size
        self.nbest_ = nbest
        self.possible_root_cats_ = cat_list_to_unordered_set(possible_root_cats)
        self.unary_rules_ = convert_unary_rules(unary_rules)
        self.binary_rules_ = en_headfirst_binary_rules
        self.seen_rules_ = convert_seen_rules(seen_rules)
        self.apply_binary_rules_ = EnGetRules
        self.apply_unary_rules_ = EnApplyUnaryRules
        self.max_length_ = max_length
        self.loglevel = loglevel
        self.tagger = None
        self.lang = b'en'

    def load_default_tagger(self, dirname):
        logger.info(f'loading default supertagger at {dirname}')
        from depccg.lstm_parser_bi_fast import FastBiaffineLSTMParser
        from depccg.ja_lstm_parser_bi import BiaffineJaLSTMParser
        model_file = os.path.join(dirname, 'tagger_model')
        def_file = os.path.join(dirname, 'tagger_defs.txt')
        assert os.path.exists(model_file) and os.path.exists(def_file), \
            (f'Failed in initialization. Directory "{dirname}" must contain both'
             '"tagger_model" and "tagger_defs.txt" files')
        with open(def_file) as f:
            self.tagger = eval(json.load(f)['model'])(dirname)
        logger.info(f'initializing supertagger with parameters at {model_file}')
        chainer.serializers.load_npz(model_file, self.tagger)

    @classmethod
    def from_dir(cls, dirname, load_tagger=False, **kwargs):
        logger.info(f'loading parser from {dirname}')
        args = [os.path.join(dirname, file)
                for file in ['unary_rules.txt', 'cat_dict.txt', 'target.txt', 'seen_rules.txt']]
        if load_tagger:
            kwargs['tagger_model_dir'] = dirname
        return cls.from_files(*args, **kwargs)

    @classmethod
    def from_files(cls, unary_rules, category_dict, categories, seen_rules, tagger_model_dir=None, **kwargs):
        files = [file for file in [unary_rules, category_dict, categories, seen_rules, tagger_model_dir]
                 if file is not None]
        logger.info(f'loading parser from files: {files}')
        unary_rules = read_unary_rules(unary_rules)
        category_dict = read_cat_dict(category_dict)
        tag_list = read_cat_list(categories)
        seen_rules = read_seen_rules(seen_rules, lambda cat: cat.strip_feat('[X]').strip_feat('[nb]'))
        parser = cls(category_dict, tag_list, unary_rules, seen_rules, **kwargs)
        if tagger_model_dir:
            parser.load_default_tagger(tagger_model_dir)
        return parser

    @classmethod
    def from_gzip(cls, filename):
        tf = tarfile.open(filename, 'r')

    def parse_doc(self, sents, probs=None, batchsize=16):
        splitted, sents = zip(*map(maybe_split_and_join, sents))
        sents = [sent.encode('utf-8') for sent in sents]
        logger.info('start tagging sentences')
        if probs is None:
            assert self.tagger is not None, 'default supertagger is not loaded.'
            probs = self.tagger.predict_doc(splitted, batchsize=batchsize)
        logger.info('done tagging sentences')
        res = self._parse_doc_tag_and_dep(list(sents), list(probs))
        return res

    def parse_json(self, json_input):
        if isinstance(json_input, str):
            json_input = [json.loads(line.strip()) for line in open(json_input)]
        categories = None
        sents = []
        probs = []
        for json_dict in json_input:
            if categories is None:
                categories = [Category.parse(cat) for cat in json_dict['categories']]

            sent = json_dict['words'].encode('utf-8')
            dep = np.array(json_dict['heads']).reshape(json_dict['heads_shape']).astype(np.float32)
            tag = np.array(json_dict['head_tags']).reshape(json_dict['head_tags_shape']).astype(np.float32)
            sents.append(sent)
            probs.append((tag, dep))
        res = self._parse_doc_tag_and_dep(sents, probs, tag_list=categories)
        return res

    cdef list _parse_doc_tag_and_dep(self, list sents, list probs, tag_list=None):
        cdef int doc_size = len(sents), i, j
        cdef np.ndarray[float, ndim=2, mode='c'] cat_scores, dep_scores
        cdef vector[string] csents = sents
        cdef float **tags = <float**>malloc(doc_size * sizeof(float*))
        cdef float **deps = <float**>malloc(doc_size * sizeof(float*))

        cdef vector[Cat] c_tag_list = cat_list_to_vector(tag_list) if tag_list else self.tag_list_
        cdef unordered_map[string, vector[bool]] c_category_dict
        if tag_list:
            c_category_dict = convert_cat_dict(self.py_category_dict, tag_list)
        else:
            c_category_dict = self.category_dict_

        for i, (cat_scores, dep_scores) in enumerate(probs):
            tags[i] = &cat_scores[0, 0]
            deps[i] = &dep_scores[0, 0]

        logger.info('start A* parsing')
        cdef vector[vector[ScoredNode]] cres = ParseSentences(
                    csents,
                    tags,
                    deps,
                    c_category_dict,
                    c_tag_list,
                    self.beta_,
                    self.use_beta_,
                    self.pruning_size_,
                    self.nbest_,
                    self.possible_root_cats_,
                    self.unary_rules_,
                    self.binary_rules_,
                    self.cache_,
                    self.seen_rules_,
                    self.apply_binary_rules_,
                    self.apply_unary_rules_,
                    self.max_length_)
        cdef int num_sents = len(sents)
        # for i in prange(num_sents, nogil=True, schedule='dynamic'):
        #     cres[i] = ParseSentence(
        #                 i,
        #                 csents[i],
        #                 tags[i],
        #                 deps[i],
        #                 self.category_dict_,
        #                 c_tag_list,
        #                 self.beta_,
        #                 self.use_beta_,
        #                 self.pruning_size_,
        #                 self.nbest_,
        #                 self.possible_root_cats_,
        #                 self.unary_rules_,
        #                 self.binary_rules_,
        #                 self.cache_,
        #                 self.seen_rules_,
        #                 self.apply_binary_rules_,
        #                 self.apply_unary_rules_,
        #                 self.max_length_)
        logger.info('done A* parsing')
        cdef list tmp, res = []
        cdef Tree parse
        for i in range(len(sents)):
            tmp = []
            for j in range(min(self.nbest_, cres[i].size())):
                tmp.append((Tree.from_ptr(cres[i][j].first, self.lang), cres[i][j].second))
            res.append(tmp)
        free(tags)
        free(deps)
        return res


cdef class JapaneseCCGParser(EnglishCCGParser):
    def __init__(self,
                 category_dict,
                 tag_list,
                 unary_rules,
                 seen_rules,
                 beta=0.00001,
                 use_beta=True,
                 pruning_size=50,
                 nbest=1,
                 possible_root_cats=None,
                 max_length=250,
                 loglevel=3):

        if possible_root_cats is None:
            possible_root_cats = [
                'NP[case=nc,mod=nm,fin=f]',
                'NP[case=nc,mod=nm,fin=t]',
                'S[mod=nm,form=attr,fin=t]',
                'S[mod=nm,form=base,fin=f]',
                'S[mod=nm,form=base,fin=t]',
                'S[mod=nm,form=cont,fin=f]',
                'S[mod=nm,form=cont,fin=t]',
                'S[mod=nm,form=da,fin=f]',
                'S[mod=nm,form=da,fin=t]',
                'S[mod=nm,form=hyp,fin=t]',
                'S[mod=nm,form=imp,fin=f]',
                'S[mod=nm,form=imp,fin=t]',
                'S[mod=nm,form=r,fin=t]',
                'S[mod=nm,form=s,fin=t]',
                'S[mod=nm,form=stem,fin=f]',
                'S[mod=nm,form=stem,fin=t]'
            ]
        possible_root_cats = [Category.parse(cat) if not isinstance(cat, Category) else cat
                              for cat in possible_root_cats]

        self.category_dict_ = convert_cat_dict(category_dict, tag_list)
        self.tag_list_ = cat_list_to_vector(tag_list)
        self.beta_ = beta
        self.use_beta_ = use_beta
        self.pruning_size_ = pruning_size
        self.nbest_ = nbest
        self.possible_root_cats_ = cat_list_to_unordered_set(possible_root_cats)
        self.unary_rules_ = convert_unary_rules(unary_rules)
        self.binary_rules_ = ja_headfinal_binary_rules
        self.seen_rules_ = convert_seen_rules(seen_rules)
        self.apply_binary_rules_ = JaGetRules
        self.apply_unary_rules_ = JaApplyUnaryRules
        self.max_length_ = max_length
        self.loglevel = loglevel
        self.tagger = None
        self.lang = b'ja'
