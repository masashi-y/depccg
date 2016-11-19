
from utils import find_closing_bracket, \
    find_non_nested_char, drop_brackets
import re

reg_non_punct = re.compile(r"[A-Za-z]+")

WILDCARD = "X"
bracket_and_quote_cat = ["LRB", "RRB", "LQU", "RQU"]

num_cats = 0
cache = {}


class Slash(object):
    FwdApp = 0
    BwdApp = 1
    EitherApp = 2

    def __init__(self, string):
        if string == "/":
            self._slash = Slash.FwdApp
        elif string == "\\":
            self._slash = Slash.BwdApp
        elif string == "|":
            self._slash = Slash.EitherApp
        else:
            raise RuntimeError("Invalid slash: " + string)

    def __str__(self):
        return "/\\|"[self._slash]

    def __eq__(self, other):
        if isinstance(other, Slash):
            return self._slash == other._slash
        elif isinstance(other, int):
            return self._slash == other
        else:
            return False

    @staticmethod
    def Fwd():
        return Slash("/")

    @staticmethod
    def Bwd():
        return Slash("\\")

    @staticmethod
    def Either():
        return Slash("|")

    def matches(self, other):
        return self._slash == Slash.EitherApp or \
                self._slash == other._slash


class Cat(object):

    def __init__(self, string, semantics):
        self.string = string + ("" if semantics == None \
                else "{{{0}}}".format(semantics))
        global num_cats
        self.id = num_cats
        num_cats += 1

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return self.string

    def __repr__(self):
        return self.string

    def hashcode(self):
        return self.id

    def substitute(self, sub):
        if sub is None:
            return self
        return Cat.parse(self.string.replace(WILDCARD, sub))

class Functor(Cat):
    def __init__(self, left, slash, right, semantics):
        base = left.with_brackets + str(slash) + right.with_brackets
        super(Functor, self).__init__(
                base if semantics is None else "({})".format(base),
                semantics)
        self.left = left
        self.slash = slash
        self.right = right
        self.semantics = semantics

    @property
    def with_brackets(self):
        return "({})".format(self.string)

    @property
    def is_modifier(self):
        return self.left == self.right

    @property
    def is_type_raised(self):
        """
        X|(X|Y)
        """
        return self.right.is_functor and \
                self.right.left == self.left

    @property
    def is_forward_type_raised(self):
        """
        X/(X\Y)
        """
        return self.is_type_raised and \
                self.slash == Slash_E.FwdApp

    @property
    def is_backward_type_raised(self):
        """
        X\(X/Y)
        """
        return self.is_type_raised and \
                self.slash == Slash_E.BwdApp

    @property
    def is_functor(self):
        return True

    @property
    def is_punct(self):
        return False

    @property
    def is_N_or_NP(self):
        return False

    @property
    def n_args(self):
        return 1 + self.left.n_args

    @property
    def feat(self):
        raise NotImplementedError()

    @property
    def type(self):
        raise NotImplementedError()

    def get_substitution(self, other):
        res = self.right.get_substitution(other.right)
        if res is None:
            res = self.left.get_substitution(other.left)
        return res

    def matches(self, other):
        return other.is_functor and \
               self.left.matches(other.left) and \
               self.right.matches(other.right) and \
               self.slash.matches(other.slash)

    def replace_arg(self, argn, new_cat):
        if argn == self.n_args:
            return Cat.make(self.left, self.slash, new_cat)
        else:
            return Cat.make(
                    self.left.replace_arg(argn, new_cat), self.slash, self.right)

    def arg(self, argn):
        if argn == self.n_args:
            return self.right
        else:
            return self.left.arg(argn)

    @property
    def head_cat(self):
        return self.left.head_cat

    def is_function_into(self, cat):
        return cat.matches(self) or \
                self.left.is_function_into(cat)

    def is_function_into_modifier(self):
        return self.is_modifier or \
                self.left.is_modifier

    def drop_PP_and_PR_feat(self):
        return Cat.make(self.left.drop_PP_and_PR_feat(),
                         self.slash,
                         self.right.drop_PP_and_PR_feat())


class Atomic(Cat):
    def __init__(self, base, feat, semantics):
        super(Atomic, self).__init__(
                base + ("" if feat is None else "[{}]".format(feat)),
                semantics)
        self.type = base
        self.feat = feat
        self.semantics = semantics

    @property
    def with_brackets(self):
        return self.string

    @property
    def is_modifier(self):
        return False

    @property
    def is_type_raised(self):
        return False

    @property
    def is_forward_type_raised(self):
        return False

    @property
    def is_backward_type_raised(self):
        return False

    @property
    def is_functor(self):
        return False

    @property
    def is_punct(self):
        return not reg_non_punct.match(self.type) or \
                self.type in bracket_and_quote_cat

    @property
    def is_N_or_NP(self):
        return self.type == "N" or self.type == "NP"

    @property
    def n_args(self):
        return 0

    def get_substitution(self, other):
        if self.feat == WILDCARD:
            return other.feat
        elif other.feat == WILDCARD:
            return self.feat
        return None

    def matches(self, other):
        return not other.is_functor and \
               self.type == other.type and \
               (self.feat == None or \
                   self.feat == other.feat or \
                   WILDCARD == self.feat or \
                   WILDCARD == other.feat or \
                   self.feat == "nb")

    def replace_arg(self, argn, new_cat):
        if argn == 0: return new_cat
        raise RuntimeError("Error replacing argument of category")

    def arg(self, argn):
        if argn == 0: return self
        raise RuntimeError("Error getting argument of category")

    @property
    def head_cat(self):
        return self

    def is_function_into(self, cat):
        return cat.matches(self)

    def is_function_into_modifier(self):
        return False

    def add_feat(self, new_feat):
        if self.feat is not None:
            raise RuntimeError("Only one feat is allowed")
        new_feat = new_feat.replace("/", "")
        new_feat = new_feat.replace("\\", "")
        return parse("{}[{}]".format(self.type, new_feat))

    def drop_PP_and_PR_feat(self):
        if self.type == "PP" or self.type == "PR":
            return parse(self.type)
        else:
            return self


def parse(cat):
    global cache
    if cat in cache:
        return cache[cat]
    else:
        name = drop_brackets(cat)
        if name in cache:
            res = cache[name]
        else:
            res = parse_uncached(name)
            if name != cat:
                cache[name] = res
        cache[cat] = res
        return res


def parse_uncached(cat):
    new_cat = cat
    if new_cat.endswith("}"):
        open_idx = new_cat.rfind("{")
        semantics = new_cat[open_idx + 1:-1]
        new_cat = new_cat[0:open_idx]
    else:
        semantics = None

    new_cat = drop_brackets(new_cat)
    # if new_cat.startswith("("):
    #     close_idx = find_closing_bracket(new_cat, 0)
    #
    #     if not any(slash in new_cat for slash in "/\\|"):
    #         new_cat = new_cat[1:close_idx]
    #         res = parse_uncached(new_cat)
    #         return res

    end_idx = len(new_cat)
    op_idx = find_non_nested_char(new_cat, "/\\|")

    if op_idx == -1:
        # atomic category
        feat_idx = new_cat.find("[")
        feats = []
        base = new_cat if feat_idx == -1 else new_cat[0:feat_idx]
        while feat_idx > -1:
            feats.append(new_cat[feat_idx + 1:new_cat.find("]", feat_idx)])
            feat_idx = new_cat.find("[", feat_idx + 1)
        if len(feats) > 1:
            pass
            # raise RuntimeError("Can only handle single features: " + cat)

        feat = None if len(feats) == 0 else feats[0]
        return Atomic(base, feat, semantics)
    else:
        # functor category
        left = parse(new_cat[:op_idx])
        slash = Slash(new_cat[op_idx:op_idx + 1])
        right = parse(new_cat[op_idx + 1:end_idx])
        return Functor(left, slash, right, semantics)


def make(left, op, right):
    return parse(left.with_brackets + str(op) + right.with_brackets)


COMMA       = parse(",")
SEMICOLON   = parse(";")
CONJ        = parse("conj")
N           = parse("N")
LQU         = parse("LQU")
LRB         = parse("LRB")
NP          = parse("NP")
PP          = parse("PP")
PREPOSITION = parse("PP/NP")
PR          = parse("PR")

