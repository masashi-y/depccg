

cdef class Combinator:
    def __cinit__(self):
        self.op_ = NULL

    def __str__(self):
        if not self.op_:
            return ''
        return self.op_.ToStr().decode('utf-8')

    def __repr__(self):
        if not self.op_:
            return ''
        return self.op_.ToStr().decode('utf-8')

    cdef bool _can_apply(self, Category left, Category right):
        return self.op_.CanApply(left.cat_, right.cat_)

    def can_apply(self, left, right):
        if self.op_ and isinstance(left, Category) and isinstance(right, Category):
            return self._can_apply(left, right)
        else:
            return False

    cdef Category _apply(self, Category left, Category right):
        return Category.from_ptr(self.op_.Apply(left.cat_, right.cat_))

    def apply(self, left, right):
        if self.can_apply(left, right):
            return self._apply(left, right)
        else:
            return None

    cdef bool _head_is_left(self, Category left, Category right):
        return self.op_.HeadIsLeft(left.cat_, right.cat_)

    def head_is_left(self, left, right):
        if self.op_ and isinstance(left, Category) and isinstance(right, Category):
            return self._head_is_left(left, right)
        else:
            return False


cdef class UnaryRule(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CUnaryRule()


cdef class Conjunction(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CConjunction()


cdef class Conjunction2(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CConjunction2()


cdef class Coordinate(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CCoordinate()


cdef class CommaAndVerbPhraseToAdverb(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CCommaAndVerbPhraseToAdverb()


cdef class ParentheticalDirectSpeech(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CParentheticalDirectSpeech()


cdef class RemovePunctuation(Combinator):
    def __cinit__(self, bool punct_is_left):
        self.op_ = <Op>new CRemovePunctuation(punct_is_left)


cdef class RemovePunctuationLeft(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CRemovePunctuationLeft()


cdef class ForwardApplication(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CForwardApplication()


cdef class BackwardApplication(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CBackwardApplication()


def slash_check(slashes):
    for slash in slashes:
        if slash not in ['/', '\\']:
            raise RuntimeError(f'slashes must be either of "\\" or "/": [{", ".join(slashes)}]')


cdef class ForwardComposition(Combinator):
    def __cinit__(self, str left, str right, str result):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        self.op_ = <Op>new CForwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]))


cdef class BackwardComposition(Combinator):
    def __cinit__(self, str left, str right, str result):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        self.op_ = <Op>new CBackwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]))


cdef class GeneralizedForwardComposition(Combinator):
    def __cinit__(self, str left, str right, str result):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        self.op_ = <Op>new CGeneralizedForwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]))


cdef class GeneralizedBackwardComposition(Combinator):
    def __cinit__(self, str left, str right, str result):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        self.op_ = <Op>new CGeneralizedBackwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]))


cdef class HeadfirstCombinator(Combinator):
    def __cinit__(self, Combinator combinator):
        self.op_ = <Op>new CHeadFirstCombinator(combinator.op_)


cdef class HeadfinalCombinator(Combinator):
    def __cinit__(self, Combinator combinator):
        self.op_ = <Op>new CHeadFinalCombinator(combinator.op_)


cdef class EnForwardApplication(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CENForwardApplication()


cdef class EnBackwardApplication(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CENBackwardApplication()


cdef class Conjoin(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CConjoin()


cdef class JaForwardApplication(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CJAForwardApplication()


cdef class JaBackwardApplication(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CJABackwardApplication()


cdef class JaGeneralizedBackwardComposition0(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedBackwardComposition0(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class JaGeneralizedBackwardComposition1(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedBackwardComposition1(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class JaGeneralizedBackwardComposition2(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedBackwardComposition2(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class JaGeneralizedBackwardComposition3(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedBackwardComposition3(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class JaGeneralizedForwardComposition0(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedForwardComposition0(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class JaGeneralizedForwardComposition1(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedForwardComposition1(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class JaGeneralizedForwardComposition2(Combinator):
    def __cinit__(self, str left, str right, str result, str name):
        slash_check([left, right, result])
        cdef string c_left = left.encode('utf-8')
        cdef string c_right = right.encode('utf-8')
        cdef string c_result = result.encode('utf-8')
        cdef string c_name = name.encode('utf-8')
        self.op_ = <Op>new CJAGeneralizedForwardComposition2(
                    Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name)


cdef class SpecialCombinator(Combinator):
    def __cinit__(self, Category left, Category right, Category result, bool head_is_left):
        self.op_ = <Op>new CSpecialCombinator(
                    left.cat_, right.cat_, result.cat_, head_is_left)


cdef class RemoveDisfluency(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CRemoveDisfluency()


cdef class UnknownCombinator(Combinator):
    def __cinit__(self):
        self.op_ = <Op>new CUnknownCombinator()


cdef vector[Op] combinator_list_to_vector(list combinators):
    cdef vector[Op] results
    cdef Combinator combinator
    for combinator in combinators:
        results.push_back(combinator.op_)
    return results


def guess_combinator_by_triplet(binary_rules, parent, child1, child2):
    for rule in binary_rules:
        guess = rule.apply(child1, child2)
        if guess and guess.matches(parent):
            return rule
    return None


UNKNOWN_COMBINATOR = UnknownCombinator()

