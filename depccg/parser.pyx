
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
from cython.parallel cimport prange
from libc.stdio cimport fprintf, stderr
from .combinator cimport combinator_list_to_vector, en_binary_rules
from .combinator import en_default_binary_rules, ja_default_binary_rules
from .utils cimport *
from .utils import maybe_split_and_join, denormalize
from .cat cimport Category


logger = logging.getLogger(__name__)


cdef PartialConstraints build_nonterminal_constraints(
        list py_constraints, const unordered_map[Cat, unordered_set[Cat]]& unary_rules):
    cdef PartialConstraints c_constraints = PartialConstraints(unary_rules)
    cdef Category cat
    cdef int start_of_span, span_length
    for cat, start_of_span, span_length in py_constraints:
        if cat is not None:
            c_constraints.Add(cat.cat_, start_of_span, span_length)
        else:
            c_constraints.Add(start_of_span, span_length)
    return c_constraints


def build_terminal_constraints(list constraints, tag_probs, tag_list):
    pseudo_neginf = -10e10
    tag_dict = {tag: i for i, tag in enumerate(tag_list)}
    for cat, i in constraints:
        cat_index = tag_dict[cat]
        tag_probs[i, :] = pseudo_neginf
        tag_probs[i, cat_index] = 0
    return tag_probs


cdef class EnglishCCGParser:
    cdef object py_category_dict
    cdef object py_tag_list
    cdef object use_beta
    cdef object use_category_dict
    cdef object use_seen_rules
    cdef unordered_map[string, unordered_set[Cat]] category_dict_
    cdef vector[Cat] tag_list_
    cdef float beta_
    cdef unsigned pruning_size_
    cdef unsigned nbest_
    cdef list py_binary_rules
    cdef vector[Op] binary_rules_
    cdef unordered_set[Cat] possible_root_cats_
    cdef unordered_map[Cat, unordered_set[Cat]] unary_rules_
    cdef unordered_set[CatPair] seen_rules_
    cdef ApplyBinaryRules apply_binary_rules_
    cdef ApplyUnaryRules apply_unary_rules_
    cdef unsigned max_length_
    cdef object tagger
    cdef bytes lang

    def __init__(self,
                 category_dict,
                 tag_list,
                 unary_rules,
                 seen_rules,
                 binary_rules=None,
                 beta=0.00001,
                 use_beta=True,
                 use_category_dict=True,
                 use_seen_rules=True,
                 pruning_size=50,
                 nbest=1,
                 possible_root_cats=None,
                 max_length=250):

        if binary_rules is None:
            binary_rules = en_default_binary_rules
        if possible_root_cats is None:
            possible_root_cats = ['S[dcl]', 'S[wq]', 'S[q]', 'S[qem]', 'NP']
        possible_root_cats = [Category.parse(cat) if not isinstance(cat, Category) else cat
                              for cat in possible_root_cats]

        logger.info(f'beta value = {beta} (use beta = {use_beta})')
        logger.info(f'pruning size = {pruning_size}')
        logger.info(f'N best = {nbest}')
        logger.info(f'use category dictionary = {use_category_dict}')
        logger.info(f'use seen rules = {use_seen_rules}')
        logger.info(f'allow at the root of a tree only categories in {possible_root_cats}'),
        logger.info(f'give up sentences that contain > {max_length} words')
        logger.info(f'combinators: {binary_rules}')

        self.py_tag_list = tag_list
        self.py_category_dict = category_dict
        self.category_dict_ = convert_cat_dict(category_dict)
        self.tag_list_ = cat_list_to_vector(tag_list)
        self.beta_ = beta
        self.use_beta = use_beta
        self.use_category_dict = use_category_dict
        self.use_seen_rules = use_seen_rules
        self.pruning_size_ = pruning_size
        self.nbest_ = nbest
        self.py_binary_rules = binary_rules
        self.binary_rules_ = combinator_list_to_vector(self.py_binary_rules)
        self.possible_root_cats_ = cat_list_to_unordered_set(possible_root_cats)
        self.unary_rules_ = convert_unary_rules(unary_rules)
        self.seen_rules_ = convert_seen_rules(seen_rules)
        self.apply_binary_rules_ = MakeEnApplyBinaryRules(self.binary_rules_)
        self.apply_unary_rules_ = EnApplyUnaryRules
        self.max_length_ = max_length
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

    def parse_doc(self, sents, probs=None, tag_list=None, constraints=None, batchsize=16):
        splitted, sents = zip(*map(maybe_split_and_join, sents))
        logger.info('start tagging sentences')
        if probs is None:
            assert self.tagger is not None, 'default supertagger is not loaded.'
            probs = self.tagger.predict_doc(splitted, batchsize=batchsize)
        logger.info('done tagging sentences')
        res = self._parse_doc_tag_and_dep(list(sents),
                                          list(probs),
                                          tag_list=tag_list,
                                          constraints=constraints)
        return res

    def parse_json(self, json_input, batchsize=16):
        if isinstance(json_input, str):
            json_input = [json.loads(line.strip()) for line in open(json_input)]
        categories = None
        sents = []
        probs = []
        constraints = []
        unprocessed = {}
        for i, json_dict in enumerate(json_input):
            if categories is None:
                categories = [Category.parse(cat) for cat in json_dict['categories']]

            words = [denormalize(word) for word in json_dict['words'].split(' ')]
            sent = ' '.join(words)
            dep = np.array(json_dict.get('heads', None)).reshape(json_dict['heads_shape']).astype(np.float32)
            tag = np.array(json_dict.get('head_tags', None)).reshape(json_dict['head_tags_shape']).astype(np.float32)

            if dep is None and tag is None:
                def process_fun(_, new_tag_and_dep):
                    return new_tag_and_dep
                tag_and_dep = None
            elif dep is None:
                def process_fun(tag_and_dep, new_tag_and_dep):
                    tag, _ = tag_and_dep
                    _, new_dep = new_tag_and_dep
                    return tag, new_dep
                tag_and_dep = (tag, None)
            elif tag is None:
                def process_fun(tag_and_dep, new_tag_and_dep):
                    _, dep = tag_and_dep
                    new_tag, _ = new_tag_and_dep
                    return new_tag, dep
                tag_and_dep = (None, dep)
            else:
                process_fun = None
                tag_and_dep = (tag, dep)

            if process_fun is not None:
                unprocessed[i] = (process_fun, words)

            constraints.append(json_dict.get('constraints', []))
            sents.append(sent)
            probs.append(tag_and_dep)

        if len(unprocessed) > 0:
            assert self.tagger is not None, 'default supertagger is not loaded.'
            _, splitted = zip(*unprocessed.values())
            unprocessed_probs = self.tagger.predict_doc(splitted, batchsize=batchsize)
            for (i, (process_fun, _)), new_tag_and_dep in zip(unprocessed.items(), unprocessed_probs):
                probs[i] = process_fun(probs[i], new_tag_and_dep)
            assert all(tag is not None and dep is not None for tag, dep in probs)

        if all(len(constraint) == 0 for constraint in constraints):
            constraints = None
        else:
            constraints = [[(Category.parse(cat), i, j) for cat, i, j in cxs]
                           for cxs in constraints]
        res = self._parse_doc_tag_and_dep(sents,
                                          probs,
                                          tag_list=categories,
                                          constraints=constraints)
        return res

    cdef list _parse_doc_tag_and_dep(self,
                                     list sents,
                                     list probs,
                                     tag_list=None,
                                     constraints=None):
        cdef int doc_size = len(sents), i, j
        cdef np.ndarray[float, ndim=2, mode='c'] cat_scores, dep_scores
        cdef vector[string] csents = vector[string](doc_size)
        cdef float **tags = <float**>malloc(doc_size * sizeof(float*))
        cdef float **deps = <float**>malloc(doc_size * sizeof(float*))

        cdef vector[Cat] c_tag_list = cat_list_to_vector(tag_list) if tag_list else self.tag_list_
        cdef unordered_map[string, unordered_set[Cat]] category_dict = \
            self.category_dict_ if self.use_category_dict else unordered_map[string, unordered_set[Cat]]()
        cdef unordered_set[CatPair] seen_rules = \
            self.seen_rules_ if self.use_seen_rules else unordered_set[CatPair]()

        cdef vector[ApplyBinaryRules] apply_binary_rules
        cdef ApplyBinaryRules constrained_binary_rules

        if constraints is not None:
            assert len(constraints) == doc_size
            logger.info('loading partial constraints')
            new_probs = []
            for constraint, (py_cat_scores, py_dep_scores) in zip(constraints, probs):
                nonterminal_constraints = [cx for cx in constraint if len(cx) == 3]
                terminal_constraints = [cx for cx in constraint if len(cx) == 2]
                logging.debug(f'non-terminal constraints: {nonterminal_constraints}')
                logging.debug(f'terminal constraints: {terminal_constraints}')
                constrained_binary_rule = MakeConstrainedBinaryRules(self.binary_rules_,
                    build_nonterminal_constraints(nonterminal_constraints, self.unary_rules_))
                apply_binary_rules.push_back(constrained_binary_rule)
                py_cat_scores = build_terminal_constraints(terminal_constraints, py_cat_scores, self.py_tag_list)
                new_probs.append((py_cat_scores, py_dep_scores))
            probs = new_probs
        else:
            apply_binary_rules = vector[ApplyBinaryRules](doc_size, self.apply_binary_rules_)

        tag_size = len(self.py_tag_list)
        for i, (py_cat_scores, py_dep_scores) in enumerate(probs):
            sent_size = len(sents[i].split(' '))
            if (sent_size, tag_size) != py_cat_scores.shape or \
                (sent_size, sent_size + 1) != py_dep_scores.shape:
                raise RuntimeError(
                    'invalid shape of input matrices:\n'
                    f'Expected P_tag: ({sent_size}, {tag_size}), P_dep: ({sent_size}, {sent_size + 1})\n'
                    f'Actual P_tag: {py_cat_scores.shape}, P_dep: {py_dep_scores.shape}')
            cat_scores = py_cat_scores
            dep_scores = py_dep_scores
            csents[i] = sents[i].encode('utf-8')
            tags[i] = &cat_scores[0, 0]
            deps[i] = &dep_scores[0, 0]

        logger.info('start A* parsing')
        cdef vector[vector[ScoredNode]] cres = ParseSentences(
                   csents,
                   tags,
                   deps,
                   category_dict,
                   c_tag_list,
                   self.beta_,
                   self.use_beta,
                   self.pruning_size_,
                   self.nbest_,
                   self.possible_root_cats_,
                   self.unary_rules_,
                   seen_rules,
                   apply_binary_rules,
                   self.apply_unary_rules_,
                   self.max_length_)
        # for i in prange(doc_size, nogil=True, schedule='dynamic'):
        #     cres[i] = ParseSentence(
        #                 i,
        #                 csents[i],
        #                 tags[i],
        #                 deps[i],
        #                 self.category_dict_,
        #                 c_tag_list,
        #                 self.beta_,
        #                 self.use_beta,
        #                 self.pruning_size_,
        #                 self.nbest_,
        #                 self.possible_root_cats_,
        #                 self.unary_rules_,
        #                 self.seen_rules_,
        #                 binary_rules[i],
        #                 self.apply_unary_rules_,
        #                 self.max_length_)
        #     # nproccessed += 1
        #     # if (nproccessed % block_size)  == block_size - 1:
        #     #     fprintf(stderr, "%d.. ", nproccessed)
        fprintf(stderr, "done.\n")

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
                 use_category_dict=True,
                 use_seen_rules=True,
                 pruning_size=50,
                 nbest=1,
                 possible_root_cats=None,
                 max_length=250):

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

        self.category_dict_ = convert_cat_dict(category_dict)
        self.tag_list_ = cat_list_to_vector(tag_list)
        self.beta_ = beta
        self.use_beta = use_beta
        self.use_category_dict = use_category_dict
        self.use_seen_rules = use_seen_rules
        self.pruning_size_ = pruning_size
        self.nbest_ = nbest
        self.py_binary_rules_ = ja_default_binary_rules
        self.binary_rules_ = combinator_list_to_vector(self.py_binary_rules)
        self.possible_root_cats_ = cat_list_to_unordered_set(possible_root_cats)
        self.unary_rules_ = convert_unary_rules(unary_rules)
        self.seen_rules_ = convert_seen_rules(seen_rules)
        self.apply_binary_rules_ = JaApplyBinaryRules
        self.apply_unary_rules_ = JaApplyUnaryRules
        self.max_length_ = max_length
        self.tagger = None
        self.lang = b'ja'
