
import utils

# implemented using AtomicInteger
WILDCARD = "X"
bracket_and_quote_cat = ["LRB", "RRB", "LQU", "RQU"]
cache = {}

def parse(cat):
    """
    cat: str
    """
    if cat in cache:
        return cache[cat]
    else:
        name = utils.drop_brackets(cat)
        if name in cache:
            res = cache[name]
        else:
            res = value_of_uncached(name)
            if name != cat:
                cache[name] = res
        cache[cat] = res
        return res

def parse_uncached(source):
    new_source = source
    if new_source.endswith("}"):
        open_idx = new_source.find("{")
        semantics = new_source[idx:-1]
        new_source = new_source[0:idx]
    else:
        semantics = None

    if new_source.startswith("("):
        close_idx = utils.find_closing_bracket(new_source, 0)

        if not any(slash in new_source for slash in ["/", "\\", "|"]):
            new_source = new_source[1:close_idx]
            res = value_of_uncached(new_source)
            return res

    end_idx = len(new_source)
    op_idx = utils.find_non_nested_char(new_source, "/\\|")

    if op_idx == -1:
        # atomic category
        feat_idx = new_source.find("[")
        feats = []
        base = new_source if feat_idx == -1 else new_source[0:feat_idx]
        while feat_idx > -1:
            feats.append(new_source[feat_idx+1:new_source.find("]", feat_idx)])
            feat_idx = new_source.find("[", feat_idx + 1)
        if len(feats) > 1:
            raise RuntimeError("Can only handle single features: " + source)

        return Atomic(base, None if len(feats) == 0 else feats[0], semantics)
    else:
        # functor category
        left = value_of(new_source[:op_idx])
        slash = new_source[op_idx:op_idx+1]
        right = value_of(new_source[op_idx+1:end_idx])
        return Functor(left, slash, right, semantics)

class Cat(object):
    num_cats = 0
    def __init__(self, string, semantics):
        self.string = string + ("" if semantics == None \
                else "{{{0}}}".format(semantics))
        self.id = Cat.num_cats
        Cat.num_cats += 1

    @property
    def with_brackets(self):
        return "({})".format(self.string)

class Functor(Cat):
    def __init__(self, left, slash, right, semantics):
        super(Functor, self).__init__(
                left.with_brackets + slash + right.with_brackets,
                semantics)
        self.left = left
        self.slash = slash
        self.right = right
        self.semantics = semantics


class Atomic(Cat):
    def __init__(self, base, feat, semantics):
        super(Atomic, self).__init__(
                base + ("" if feat is None else "[{}]".format(feat)),
                semantics)
        self.base = base
        self.feat = feat
        self.semantics = semantics

