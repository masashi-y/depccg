
from cat import Cat

class RuleType(object):
    FA = 0
    BA = 1
    FC = 2
    BX = 3
    GFC = 4
    GBX = 5
    CONJ = 6
    RP = 7
    LP = 8
    NOISE = 9
    UNARY = 10
    LEXICON = 11


class RuleProduction(object):
    def __init__(rule_type, result, head_is_left):
        self.rule_type = rule_type
        self.cat = result
        self.head_is_left = head_is_left


class Combinator(object):
    STANDARD_COMBINATORS = []
    def __init__(self, rule_type):
        self.rule_type = rule_type

    @staticmethod
    def load_special_combinators(filename):
        res = []
        for line in open(filename):
            comment = line.find("#")
            if comment > -1:
                line = line[:comment]
            line = line.strip()
            if len(line) == 0: continue

            items = line.split(" ")
            head_is_left = items[0] == "l"
            left = Cat.parse(items[1])
            right = Cat.parse(items[2])
            result = Cat.parse(items[3])
            res.append(SpecialCombinator(
                left, right, result, head_is_left))
        return res

    @staticmethod
    def correct_wildcard_features(to_correct, match1, match2):
        return to_correct.substitute(
                match1.get_substitution(match2))

    @staticmethod
    def get_rules(left, right, rules):
        res = []
        for rule in rules:
            if rule.can_apply(left, right):
                res.append(RuleProduction(rule.rule_type,
                    rule.apply(left, right), c.head_is_left(left, right)))
        return res


class Conjunction(Combinator):
    def __init__(self):
        super(Conjunction, self).__init__(RuleType.CONJ)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class RemovePunctuation(Combinator):
    def __init__(self):
        super(RemovePunctuation, self).__init__(RuleType.RP)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class RemovePunctuationLeft(Combinator):
    def __init__(self):
        super(RemovePunctuationLeft, self).__init__(RuleType.LP)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class SpecialCombinator(Combinator):
    def __init__(self):
        super(SpecialCombinator, self).__init__(RuleType.NOISE)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class ForwardApplication(Combinator):
    def __init__(self):
        super(ForwardApplication, self).__init__(RuleType.FA)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class BackwardApplication(Combinator):
    def __init__(self):
        super(BackwardApplication, self).__init__(RuleType.BA)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class ForwardComposition(Combinator):
    def __init__(self):
        super(ForwardComposition, self).__init__(RuleType.FC)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class BackwardComposition(Combinator):
    def __init__(self):
        super(BackwardComposition, self).__init__(RuleType.BX)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class GeneralizedForwardComposition(Combinator):
    def __init__(self):
        super(GeneralizedForwardComposition, self).__init__(RuleType.GFC)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


class GeneralizedBackwardComposition(Combinator):
    def __init__(self):
        super(GeneralizedBackwardComposition, self).__init__(RuleType.GBX)

    @staticmethod
    def can_apply(left, right):
        pass

    @staticmethod
    def head_is_left(left, right):
        pass

    def apply(self, left, right):
        pass


