
from __future__ import print_function
cimport numpy as np
import numpy as np
import sys, re
import tarfile
from libc.stdlib cimport malloc, free
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from .cat cimport Category, Cat, CatPair
from .tree cimport Tree, ScoredNode
from .combinator cimport en_headfirst_binary_rules, ja_headfinal_binary_rules, Op
import os
import json
import chainer
from libc.string cimport memcpy
from libcpp cimport bool
import logging


logger = logging.getLogger(__name__)


cdef extern from "depccg.h" namespace "myccg" nogil:
    cdef cppclass RuleCache

    cdef cppclass AgendaItem

    ctypedef vector[RuleCache]& (*ApplyBinaryRules)(
            unordered_map[CatPair, vector[RuleCache]]&,
            const vector[Op]&, const unordered_set[CatPair]&, Cat, Cat)

    ctypedef vector[Cat] (*ApplyUnaryRules)(
            const unordered_map[Cat, vector[Cat]]&, NodeType)

    ApplyUnaryRules EnApplyUnaryRules

    ApplyUnaryRules JaApplyUnaryRules

    ApplyBinaryRules EnGetRules

    ApplyBinaryRules JaGetRules

    vector[vector[ScoredNode]] ParseSentences(
            vector[string]& sents,
            float** tag_scores,
            float** dep_scores,
            const unordered_map[string, vector[bint]]& category_dict,
            const vector[Cat]& tag_list,
            float beta,
            bint use_beta,
            unsigned pruning_size,
            unsigned nbest,
            const unordered_set[Cat]& possible_root_cats,
            const unordered_map[Cat, vector[Cat]]& unary_rules,
            const vector[Op]& binary_rules,
            unordered_map[CatPair, vector[RuleCache]]& cache,
            const unordered_set[CatPair]& seen_rules,
            ApplyBinaryRules apply_binary_rules,
            ApplyUnaryRules apply_unary_rules,
            unsigned max_length)


cdef vector[Cat] cat_list_to_vector(list cats):
    cdef vector[Cat] results
    cdef Category cat
    for cat in cats:
        results.push_back(cat.cat_)
    return results


cdef unordered_set[Cat] cat_list_to_unordered_set(list cats):
    cdef unordered_set[Cat] results
    cdef Category cat
    for cat in cats:
        results.insert(cat.cat_)
    return results


cdef unordered_map[string, vector[bool]] convert_cat_dict(dict cat_dict, list cat_list):
    cdef unordered_map[string, vector[bool]] results
    cdef vector[bool] tmp
    cdef str py_word
    cdef string c_word
    cdef list cats
    cat_to_index = {str(cat): i for i, cat in enumerate(cat_list)}
    for py_word, cats in cat_dict.items():
        c_word = py_word.encode('utf-8')
        tmp = vector[bool](len(cat_list), False)
        for cat in cats:
            tmp[cat_to_index[str(cat)]] = True
        results[c_word] = tmp
    return results


cdef unordered_map[Cat, vector[Cat]] convert_unary_rules(list unary_rules):
    cdef unordered_map[Cat, vector[Cat]] results
    cdef vector[Cat] tmp
    cdef Category cat1, cat2
    for cat1, cat2 in unary_rules:
        if results.count(cat1.cat_) == 0:
            results[cat1.cat_] = vector[Cat]()
        results[cat1.cat_].push_back(cat2.cat_)
    return results


cpdef remove_comment(line):
    comment = line.find('#')
    if comment != -1:
        line = line[:comment]
    return line.strip()


cpdef read_unary_rules(filename):
    results = []
    for line in open(filename):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        cat1, cat2 = line.split()
        cat1 = Category.parse(cat1)
        cat2 = Category.parse(cat2)
        results.append((cat1, cat2))
    logger.info(f'load {len(results)} unary rules')
    return results


cpdef read_cat_dict(filename):
    results = {}
    for line in open(filename):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        word, *cats = line.split()
        results[word] = [Category.parse(cat) for cat in cats]
    logger.info(f'load {len(results)} cat dictionary entries')
    return results


cpdef read_cat_list(filename):
    results = []
    for line in open(filename):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        cat = line.split()[0]
        results.append(Category.parse(cat))
    logger.info(f'load {len(results)} categories')
    return results


cpdef read_seen_rules(filename, preprocess):
    cdef list results = []
    cdef Category cat1, cat2
    for line in open(filename):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        tmp1, tmp2 = line.split()
        cat1 = preprocess(Category.parse(tmp1))
        cat2 = preprocess(Category.parse(tmp2))
        results.append((cat1, cat2))
    logger.info(f'load {len(results)} seen rules')
    return results


cdef unordered_set[CatPair] convert_seen_rules(seen_rule_list):
    cdef unordered_set[CatPair] results
    cdef Category cat1, cat2
    for cat1, cat2 in seen_rule_list:
        results.insert(CatPair(cat1.cat_, cat2.cat_))
    return results


cdef unordered_set[Cat] read_possible_root_categories(list cats):
    cdef unordered_set[Cat] res
    cdef Category tmp
    for cat in cats:
        tmp = Category.parse(cat)
        res.insert(tmp.cat_)
    return res


def maybe_split_and_join(string):
    if isinstance(string, list):
        split = string
        join = ' '.join(string)
    else:
        assert isinstance(string, str)
        split = ' '.split(string)
        join = string
    return split, join


def __show_mathml(tree):
    if not tree.is_leaf:
        return """\
<mrow>
  <mfrac linethickness='2px'>
    <mrow>{}</mrow>
    <mstyle mathcolor='Red'>{}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>{}</mtext>
</mrow>""".format(
                "".join(map(__show_mathml, tree.children)),
                __show_mathml_cat(str(tree.cat)),
                tree.op_string)
    else:
        return """\
<mrow>
  <mfrac linethickness='2px'>
    <mtext mathsize='1.0' mathcolor='Black'>{}</mtext>
    <mstyle mathcolor='Red'>{}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>lex</mtext>
</mrow>""".format(
                tree.word,
                __show_mathml_cat(str(tree.cat)))


def __show_mathml_cat(cat):
    cats_feats = re.findall(r'([\w\\/()]+)(\[.+?\])*', cat)
    mathml_str = ''
    for cat, feat in cats_feats:
        cat_mathml = """\
<mi mathvariant='italic'
  mathsize='1.0' mathcolor='Red'>{}</mi>
    """.format(cat)

        if feat != '':
            mathml_str += """\
<msub>{}
  <mrow>
  <mi mathvariant='italic'
    mathsize='0.8' mathcolor='Purple'>{}</mi>
  </mrow>
</msub>""".format(cat_mathml, feat)

        else:
            mathml_str += cat_mathml
    return mathml_str


def to_mathml(trees, file=sys.stdout):
    def __show(tree):
        res = ""
        for t in tree:
            if isinstance(t, tuple):
                t, prob = t
                res += "<p>Log prob={:.5e}</p>".format(prob)
            res += "<math xmlns='http://www.w3.org/1998/Math/MathML'>{}</math>".format(
                __show_mathml(t))
        return res

    string = ""
    for i, tree in enumerate(trees):
        words = tree[0][0].word if isinstance(tree[0], tuple) else tree[0].word
        string += "<p>ID={}: {}</p>{}".format(
                i, words, __show(tree))

    results = """\
<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <style>
    body {{
      font-size: 1em;
    }}
  </style>
  <script type="text/javascript"
     src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>
<body>{}
</body></html>""".format(string)
    print(results, file=file)


cdef class EnglishCCGParser:
    cdef unordered_map[string, vector[bool]] category_dict_
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
    cdef object type_check
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
                 batchsize=16,
                 loglevel=3,
                 type_check=False):

        if possible_root_cats is None:
            possible_root_cats = ['S[dcl]', 'S[wq]', 'S[q]', 'S[qem]', 'NP']
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
        self.binary_rules_ = en_headfirst_binary_rules
        self.seen_rules_ = convert_seen_rules(seen_rules)
        self.apply_binary_rules_ = EnGetRules
        self.apply_unary_rules_ = EnApplyUnaryRules
        self.max_length_ = max_length
        self.loglevel = loglevel
        self.type_check = type_check
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
        seen_rules = read_seen_rules(seen_rules, lambda cat: cat.strip_feat('X').strip_feat('nb'))
        parser = cls(category_dict, tag_list, unary_rules, seen_rules, **kwargs)
        if tagger_model_dir:
            parser.load_default_tagger(tagger_model_dir)
        return parser

    @classmethod
    def from_gzip(cls, filename):
        tf = tarfile.open(filename, 'r')

    def parse_doc(self, sents, probs=None, batchsize=16):
        assert self.tagger is not None, 'default supertagger is not loaded.'
        splitted, sents = zip(*map(maybe_split_and_join, sents))
        sents = [sent.encode('utf-8') for sent in sents]
        if probs is None:
            probs = self.tagger.predict_doc(splitted, batchsize=batchsize)
        res = self._parse_doc_tag_and_dep(list(sents), list(probs))
        return res

    def parse_json(self, json_input):
        if isinstance(json_input, str):
            json_input = [line.strip() for line in open(json_input)]
        categories = None
        sents = []
        probs = []
        for line in json_input:
            json_dict = json.load(line)
            if categories is None:
                categories = json_dict['categories']

            words = [word for word in json_dict['words'].split(' ')]
            heads = np.array(json_dict['heads']).reshape(json_dict['heads_shape']).astype(np.float32)
            head_tags = np.array(json_dict['head_tags']).reshape(json_dict['head_tags_shape']).astype(np.float32)
            sents.append(words)
            probs.append((head_tags, heads))
        res = self._parse_doc_tag_and_dep(sents, probs, tag_list=categories)
        return res

    cdef list _parse_doc_tag_and_dep(self, list sents, list probs, tag_list=None):
        cdef int doc_size = len(sents), i, j
        cdef np.ndarray[float, ndim=2, mode='c'] cat_scores, dep_scores
        cdef vector[string] csents = sents
        cdef float **tags = <float**>malloc(doc_size * sizeof(float*))
        cdef float **deps = <float**>malloc(doc_size * sizeof(float*))

        cdef vector[Cat] c_tag_list = cat_list_to_vector(tag_list) if tag_list else self.tag_list_

        for i, (cat_scores, dep_scores) in enumerate(probs):
            tags[i] = &cat_scores[0, 0]
            deps[i] = &dep_scores[0, 0]
        logger.info('start parsing sentences')
        cdef vector[vector[ScoredNode]] cres = ParseSentences(
                        csents,
                        tags,
                        deps,
                        self.category_dict_,
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
        logger.info('finished parsing sentences')
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
                 batchsize=16,
                 loglevel=3,
                 type_check=False):

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
        self.type_check = type_check
        self.tagger = None
        self.lang = b'ja'
