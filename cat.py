
import utils

# implemented using AtomicInteger
num_cats = 0
WILDCARD = "X"
bracket_and_quote_cat = ["LRB", "RRB", "LQU", "RQU"]
cache = {}

class Cat(object):
    def __init__(self, string, semantics):
        self.string = string + ("" if semantics == None else "{{{0}}}" % semantics)
        self.id = num_cats
        num_cats += 1

    @staticmethod
    def value_of(cat):
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
                res = self.value_of_uncached(name)
                if name != cat:
                    cache[name] = res
            cache[cat] = res
            return res

    def value_of_uncached(source):
        new_source = source
        if new_source.endswith("}"):
            open_idx = new_source.find("{")
            semantics = new_source[idx:-1]
            new_source = new_source[0:idx]
        else:
            semantics = None

        if new_source.startswith("("):
            close_idx = utils.find_closing_brackets(new_source)

            if not any(slash in new_source for slash in ["/", "\\", "|"]):
                new_source = new_source[1:close_idx]
                res = self.value_of_uncached(new_source)
                return res

        end_idx = len(new_source)
        op_idx = utils.find_non_nested_char(new_source, "/\\|")

        # atomic category
        if op_idx == -1:
            feat_idx = new_source.find("[")
            feats = []
            base = new_source if feat_idx == -1 else new_source[0:feat_idx]
            while feat_idx > -1:
                feats.append(new_source[feature_idx+1:new_source.find("]")])



class Functor(Cat):
    pass

class Atomic(Cat):
    pass
