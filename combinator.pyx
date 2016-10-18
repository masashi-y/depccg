# -*- coding: utf-8 -*-

import cat
cimport cat

cdef class Combinator(object):
    def __init__(self, int rule_type):
        self.rule_type = rule_type

    @staticmethod
    def load_special_combinators(str filename):
        cdef list res = []
        cdef str line
        cdef int comment
        cdef list items
        cdef Cat left, right, result
        cdef bint head_is_left

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
    def correct_wildcard_features(Cat to_correct, Cat match1, Cat match2):
        """
        Returns:
            (Cat)
        """
        return to_correct.substitute(
                match1.get_substitution(match2))

    @staticmethod
    def get_rules(Cat left, Cat right, list rules):
        cdef list res = []
        cdef Combinator rule

        for rule in rules:
            if rule.can_apply(left, right):
                res.append((rule,
                    rule.apply(left, right),
                    rule.head_is_left(left, right)))
        return res

    def __str__(self):
        return "<*>"

cdef class UnaryRule(Combinator):
    def __init__(self):
        super(UnaryRule, self).__init__(UNARY)

    def __str__(self):
        return "<un>"

cdef class Conjunction(Combinator):
    # A   ,   B  and  C
    # NP CONJ NP CONJ NP
    #            -----(Conj)
    #            NP\NP
    #         --------<
    #            NP
    #     ------------(Conj)
    #        NP\NP
    # ----------------<
   #          NP
    def __init__(self):
        super(Conjunction, self).__init__(CONJ)

    def __str__(self):
        return "<Φ>"

    def can_apply(self, Cat left, Cat right):
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


    def head_is_left(self, Cat left, Cat right):
        return False

    def apply(self, Cat left, Cat right):
        return cat.make(right, Slash.Bwd(), right)


cdef class RemovePunctuation(Combinator):
    def __init__(self, bint punct_is_left):
        super(RemovePunctuation, self).__init__(RP)
        self.punct_is_left = punct_is_left

    def __str__(self):
        return "<rp>"

    def can_apply(self, Cat left, Cat right):
        # Disallow punctuation combining with nouns,
        # to avoid getting NPs like Barack Obama .
        return left.is_punct if self.punct_is_left else \
                right.is_punct and not cat.N.matches(left)

    def head_is_left(self, Cat left, Cat right):
        return not self.punct_is_left

    def apply(self, Cat left, Cat right):
        return right if self.punct_is_left else left


cdef class RemovePunctuationLeft(Combinator):
    # Open Brackets and Quotation
    def __init__(self):
        super(RemovePunctuationLeft, self).__init__(LP)

    def __str__(self):
        return "<rp>"

    def can_apply(self, Cat left, Cat right):
        return left == cat.LQU or left == cat.LRB

    def head_is_left(self, Cat left, Cat right):
        return False

    def apply(self, Cat left, Cat right):
        return right


cdef class SpecialCombinator(Combinator):
    def __init__(self, Cat left, Cat right, Cat result, bint head_is_left):
        super(SpecialCombinator, self).__init__(NOISE)
        self.left = left
        self.right = right
        self.result = result
        self.head_is_left = head_is_left

    def __str__(self):
        return "<Sp>"

    def can_apply(self, Cat left, Cat right):
        return self.left.matches(left) and \
                self.right.matches(right)

    def head_is_left(self, Cat left, Cat right):
        return self.head_is_left

    def apply(self, Cat left, Cat right):
        return self.result


cdef class ForwardApplication(Combinator):
    def __init__(self):
        super(ForwardApplication, self).__init__(FA)

    def __str__(self):
        return ">"

    def can_apply(self, Cat left, Cat right):
        return left.is_functor and \
                left.slash == Slash.Fwd() and \
                left.right.matches(right)

    def head_is_left(self, Cat left, Cat right):
        return not ( left.is_modifier or left.is_type_raised )

    def apply(self, Cat left, Cat right):
        if left.is_modifier:
            return right
        cdef Cat res = left.left
        res = Combinator.correct_wildcard_features(
                            res, left.right, right)
        return res


cdef class BackwardApplication(Combinator):
    def __init__(self):
        super(BackwardApplication, self).__init__(BA)

    def __str__(self):
        return "<"

    def can_apply(self, Cat left, Cat right):
        return right.is_functor and \
                right.slash == Slash.Bwd() and \
                right.right.matches(left)

    def head_is_left(self, Cat left, Cat right):
        return right.is_modifier or right.is_type_raised

    def apply(self, Cat left, Cat right):
        cdef Cat res = right.left
        return Combinator.correct_wildcard_features(
                res, right.right, left)


cdef class ForwardComposition(Combinator):
    # S/NP NP/(S/NP) --> S/(S/NP)
    def __init__(self, Slash left, Slash right, Slash slash):
        super(ForwardComposition, self).__init__(FC)
        self.left_slash = left
        self.right_slash = right
        self.result_slash = slash

    def __str__(self):
        return ">B"

    def can_apply(self, Cat left, Cat right):
        return left.is_functor and \
                right.is_functor and \
                left.right.matches(right.left) and \
                left.slash == self.left_slash and \
                right.slash == self.right_slash

    def head_is_left(self, Cat left, Cat right):
        return not ( left.is_modifier or left.is_type_raised )

    def apply(self, Cat left, Cat right):
        cdef Cat res = right if left.is_modifier else \
                cat.make(left.left, self.result_slash, right.right)
        return Combinator.correct_wildcard_features(
                res, right.left, left.right)


cdef class BackwardComposition(Combinator):
    # NP\(S/NP) S\NP --> S\(S/NP)
    def __init__(self, Slash left, Slash right, Slash slash):
        super(BackwardComposition, self).__init__(BX)
        self.left_slash = left
        self.right_slash = right
        self.result_slash = slash

    def __str__(self):
        return "<B"

    def can_apply(self, Cat left, Cat right):
        return left.is_functor and \
                right.is_functor and \
                right.right.matches(left.left) and \
                left.slash == self.left_slash and \
                right.slash == self.right_slash and \
                not left.left.is_N_or_NP # Additional constraint from Steedman (2000)

    def head_is_left(self, Cat left, Cat right):
        return right.is_modifier or right.is_type_raised

    def apply(self, Cat left, Cat right):
        cdef Cat res = left if right.is_modifier else \
                    cat.make(right.left, self.result_slash, left.right)
        return Combinator.correct_wildcard_features(
                res, left.left, right.right)


cdef class GeneralizedForwardComposition(Combinator):
    # X/Y (Y|Z_1)|Z_2 --> (X|Z_1)|Z_2
    def __init__(self, Slash left, Slash right, Slash slash):
        super(GeneralizedForwardComposition, self).__init__(GFC)
        self.left_slash = left
        self.right_slash = right
        self.result_slash = slash

    def __str__(self):
        return ">Bx"

    def can_apply(self, Cat left, Cat right):
        if left.is_functor and \
            right.is_functor and \
            right.left.is_functor:
                return left.right.matches(right.left.left) and \
                        left.slash == self.left_slash and \
                        right.left.slash == self.right_slash
        return False

    def head_is_left(self, Cat left, Cat right):
        return not ( left.is_modifier or left.is_type_raised )

    def apply(self, Cat left, Cat right):
        if left.is_modifier:
            return right
        cdef Cat res = cat.make(cat.make(
                        left.left, self.result_slash, right.left.right),
                        right.slash, right.right)
        return Combinator.correct_wildcard_features(
                res, right.left.left, left.right)


cdef class GeneralizedBackwardComposition(Combinator):
    # (Y|Z_1)|Z_2 X\Y --> (X|Z_1)|Z_2
    def __init__(self, Slash left, Slash right, Slash slash):
        super(GeneralizedBackwardComposition, self).__init__(GBX)
        self.left_slash = left
        self.right_slash = right
        self.result_slash = slash

    def __str__(self):
        return "<Bx"

    def can_apply(self, Cat left, Cat right):
        if left.is_functor and \
            right.is_functor and \
            left.left.is_functor:
                return right.right.matches(left.left.left) and \
                        left.left.slash == self.left_slash and \
                        right.slash == self.right_slash and \
                        not left.left.is_N_or_NP
        return False

    def head_is_left(self, Cat left, Cat right):
        return right.is_modifier or right.is_type_raised

    def apply(self, Cat left, Cat right):
        if right.is_modifier:
            return left
        cdef Cat res = cat.make(cat.make(
                    right.left, self.result_slash, left.left.right),
                left.slash, left.right)
        return Combinator.correct_wildcard_features(
                res, left.left.left, right.right)


standard_combinators = \
    [Conjunction(),
    RemovePunctuation(False),
    RemovePunctuationLeft(),
    ForwardApplication(),
    BackwardApplication(),
    ForwardComposition(Slash.Fwd(), Slash.Fwd(), Slash.Fwd()),
    BackwardComposition(Slash.Fwd(), Slash.Bwd(), Slash.Fwd()),
    GeneralizedForwardComposition(Slash.Fwd(), Slash.Fwd(), Slash.Fwd()),
    GeneralizedBackwardComposition(Slash.Fwd(), Slash.Bwd(), Slash.Fwd())]

unary_rule = UnaryRule()
