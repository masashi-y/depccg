# -*- coding: utf-8 -*-

from py.cat import Slash
import py.cat

class RuleType:
    FA      = 0
    BA      = 1
    FC      = 2
    BC      = 3
    GFC     = 4
    GBC     = 5
    FX      = 6
    BX      = 7
    CONJ    = 8
    RP      = 9
    LP      = 10
    NOISE   = 11
    UNARY   = 12
    LEXICON = 13
    NONE    = 14
    SSEQ    = 15

class Combinator(object):
    def __init__(self, rule_type):
        self.rule_type = rule_type

    @staticmethod
    def load_special_combinators(filename):
        for line in open(filename):
            comment = line.find("#")
            if comment > -1:
                line = line[:comment]
            line = line.strip()
            if len(line) == 0: continue

            items = line.split(" ")
            head_is_left = items[0] == "l"
            left = cat.parse(items[1])
            right = cat.parse(items[2])
            result = cat.parse(items[3])
            res.append(SpecialCombinator(
                left, right, result, head_is_left))
        return res

    @staticmethod
    def correct_wildcard_features(to_correct, match1, match2):
        """
        Returns:
            (Cat)
        """
        return to_correct.substitute(
                match1.get_substitution(match2))

    @staticmethod
    def get_rules(left, right, rules):
        res = []
        for rule in rules:
            if rule.can_apply(left, right):
                res.append((rule,
                    rule.apply(left, right),
                    rule.head_is_left(left, right)))
        return res

    def __str__(self):
        return "<*>"

class UnaryRule(Combinator):
    def __init__(self):
        super(UnaryRule, self).__init__(RuleType.UNARY)

    def __str__(self):
        return "<un>"

class Conjunction(Combinator):
    # A   ,   B  and  C
    # NP CONJ NP CONJ NP
    #            -----(Conj)
    #            NP\NP
    #         --------<
    #            NP
    #     ------------(Conj)
    #        NP\NP
    # ----------------<
    #         NP
    def __init__(self):
        super(Conjunction, self).__init__(RuleType.CONJ)

    def __str__(self):
        return "<Φ>"

    def can_apply(self, left, right):
        # Comments from easyCCG:
        # * Don't start making weird ,\, categories...
        # * Improves coverage of C&C evaluation script.
        #   Categories can just conjoin first, then type-raise.
        #   (not self.right.is_type_raised)
        # * Blocks noun conjunctions, which should normally be NP conjunctions.
        #   In a better world, conjunctions would have categories like (NP\NP/NP.
        #   Doesn't affect F-scopes, but makes output semantically nicer.
        #
        # issues related to C&C evaluation script?
        # """C&C evaluation script does't let you do this, for some reason"""
        if cat.parse("NP\\NP").matches(right):
            return False

        return (left == cat.CONJ or \
                left == cat.COMMA or \
                left == cat.SEMICOLON) and \
                not right.is_punct and \
                not right.is_type_raised and \
                not (not right.is_functor and right.type == "N")


    def head_is_left(self, left, right):
        return False

    def apply(self, left, right):
        return cat.make(right, Slash.Bwd(), right)


class RemovePunctuation(Combinator):
    def __init__(self, punct_is_left):
        super(RemovePunctuation, self).__init__(RuleType.RP)
        self.punct_is_left = punct_is_left

    def __str__(self):
        return "<rp>"

    def can_apply(self, left, right):
        # Disallow punctuation combining with nouns,
        # to avoid getting NPs like Barack Obama .
        return left.is_punct if self.punct_is_left else \
                right.is_punct and not cat.N.matches(left)

    def head_is_left(self, left, right):
        return not self.punct_is_left

    def apply(self, left, right):
        return right if self.punct_is_left else left


class RemovePunctuationLeft(Combinator):
    # Open Brackets and Quotation
    def __init__(self):
        super(RemovePunctuationLeft, self).__init__(RuleType.LP)

    def __str__(self):
        return "<rp>"

    def can_apply(self, left, right):
        return left == cat.LQU or left == cat.LRB

    def head_is_left(self, left, right):
        return False

    def apply(self, left, right):
        return right


class SpecialCombinator(Combinator):
    def __init__(self, left, right, result, head_is_left):
        super(SpecialCombinator, self).__init__(RuleType.NOISE)
        self.left = left
        self.right = right
        self.result = result
        self.head_is_left = head_is_left

    def __str__(self):
        return "<Sp>"

    def can_apply(self, left, right):
        return self.left.matches(left) and \
                self.right.matches(right)

    def head_is_left(self, left, right):
        return self.head_is_left

    def apply(self, left, right):
        return self.result


class ForwardApplication(Combinator):
    def __init__(self):
        super(ForwardApplication, self).__init__(RuleType.FA)

    def __str__(self):
        return ">"

    def can_apply(self, left, right):
        return left.is_functor and \
                left.slash == Slash.Fwd() and \
                left.right.matches(right)

    def head_is_left(self, left, right):
        return not ( left.is_modifier_without_feat or left.is_type_raised_without_feat )

    def apply(self, left, right):
        if left.is_modifier:
            return right
        res = left.left
        res = Combinator.correct_wildcard_features(res, left.right, right)
        return res


class BackwardApplication(Combinator):
    def __init__(self):
        super(BackwardApplication, self).__init__(RuleType.BA)

    def __str__(self):
        return "<"

    def can_apply(self, left, right):
        return right.is_functor and \
                right.slash == Slash.Bwd() and \
                right.right.matches(left)

    def head_is_left(self, left, right):
        return right.is_modifier_without_feat or right.is_type_raised_without_feat

    def apply(self, left, right):
        res = right.left
        return Combinator.correct_wildcard_features(res, right.right, left)


# class ForwardComposition(Combinator):
#     # S/NP NP/(S/NP) --> S/(S/NP)
#     def __init__(self, left, right, slash):
#         super(ForwardComposition, self).__init__(RuleType.FC)
#         self.left_slash = left
#         self.right_slash = right
#         self.result_slash = slash
#
#     def __str__(self):
#         return ">B"
#
#     def can_apply(self, left, right):
#         return left.is_functor and \
#                 right.is_functor and \
#                 left.right.matches(right.left) and \
#                 left.slash == self.left_slash and \
#                 right.slash == self.right_slash
#
#     def head_is_left(self, left, right):
#         return not ( left.is_modifier or left.is_type_raised )
#
#     def apply(self, left, right):
#         res = right if left.is_modifier else \
#                 cat.make(left.left, self.result_slash, right.right)
#         return Combinator.correct_wildcard_features(
#                 res, right.left, left.right)
#
#
# class BackwardComposition(Combinator):
#     # NP\(S/NP) S\NP --> S\(S/NP)
#     def __init__(self, left, right, slash):
#         super(BackwardComposition, self).__init__(RuleType.BX)
#         self.left_slash = left
#         self.right_slash = right
#         self.result_slash = slash
#
#     def __str__(self):
#         return "<B"
#
#     def can_apply(self, left, right):
#         return left.is_functor and \
#                 right.is_functor and \
#                 right.right.matches(left.left) and \
#                 left.slash == self.left_slash and \
#                 right.slash == self.right_slash and \
#                 not left.left.is_N_or_NP # Additional constraint from Steedman (2000)
#
#     def head_is_left(self, left, right):
#         return right.is_modifier or right.is_type_raised
#
#     def apply(self, left, right):
#         res = left if right.is_modifier else \
#                     cat.make(right.left, self.result_slash, left.right)
#         return Combinator.correct_wildcard_features(
#                 res, left.left, right.right)


class GeneralizedForwardComposition(Combinator):
    # X/Y (Y|Z_1)|Z_2 --> (X|Z_1)|Z_2
    def __init__(self, order, rule, left, right, slash):
        super(GeneralizedForwardComposition, self).__init__(rule)
        self.order = order
        self.left_slash = left
        self.right_slash = right
        self.result_slash = slash

    def __str__(self):
        return ">B" + str(self.order)

    def can_apply(self, left, right):
        order = self.order
        return left.is_functor and \
               right.has_functor_at_left(order) and \
               left.right.matches(right.get_left(order+1)) and \
               left.slash == self.left_slash and \
               right.get_left(order).slash == self.right_slash

    def head_is_left(self, left, right):
        return not ( left.is_modifier_without_feat or left.is_type_raised_without_feat )

    def apply(self, left, right):
        order = self.order
        res = right if left.is_modifier else \
                cat.compose(order, left.left, self.result_slash, right)
        return Combinator.correct_wildcard_features(
                res, right.get_left(order+1), left.right)


class GeneralizedBackwardComposition(Combinator):
    # (Y|Z_1)|Z_2 X\Y --> (X|Z_1)|Z_2
    def __init__(self, order, rule, left, right, slash):
        super(GeneralizedBackwardComposition, self).__init__(rule)
        self.order = order
        self.left_slash = left
        self.right_slash = right
        self.result_slash = slash

    def __str__(self):
        return "<B" + str(self.order)

    def can_apply(self, left, right):
        order = self.order
        return right.is_functor and \
                left.has_functor_at_left(order) and \
                right.right.matches(left.get_left(order+1)) and \
                left.get_left(order).slash == self.left_slash and \
                not left.get_left(order+1).is_N_or_NP

    def head_is_left(self, left, right):
        return right.is_modifier_without_feat or right.is_type_raised_without_feat

    def apply(self, left, right):
        order = self.order
        res = left if right.is_modifier else \
                cat.compose(order, right.left, self.result_slash, left)
        return Combinator.correct_wildcard_features(
                res, left.get_left(order+1), right.right)


class SSEQ(Combinator):
    def __init__(self):
        super(SSEQ, self).__init__(RuleType.SSEQ)

    def __str__(self):
        return "SSEQ"

    def can_apply(self, left, right):
        return left == right and \
                not left.is_functor and \
                left.type == "S"

    def head_is_left(self, left, right):
        return False

    def apply(self, left, right):
        return left


# standard_combinators = \
#     [Conjunction(),
#     RemovePunctuation(False),
#     RemovePunctuationLeft(),
#     ForwardApplication(),
#     BackwardApplication(),
#     ForwardComposition(Slash.Fwd(), Slash.Fwd(), Slash.Fwd()),
#     BackwardComposition(Slash.Fwd(), Slash.Bwd(), Slash.Fwd()),
#     GeneralizedForwardComposition(Slash.Fwd(), Slash.Fwd(), Slash.Fwd()),
#     GeneralizedBackwardComposition(Slash.Fwd(), Slash.Bwd(), Slash.Fwd())]
#
unary_rule = UnaryRule()

