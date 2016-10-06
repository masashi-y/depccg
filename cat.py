
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

class Functor(Cat):
    pass

class Atomic(Cat):
    pass
