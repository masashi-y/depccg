

cdef class Combinator:
    def __cinit__(self):
        pass

    @staticmethod
    cdef Combinator from_ptr(Op op):
        combinator = Combinator()
        combinator.op_ = op
        return combinator

    def __str__(self):
        return self.op_.ToStr().decode('utf-8')

    def __repr__(self):
        return self.op_.ToStr().decode('utf-8')

    cdef bool _can_apply(self, Category left, Category right):
        return self.op_.CanApply(left.cat_, right.cat_)

    def can_apply(self, left, right):
        if isinstance(left, Category) and isinstance(right, Category):
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
        if isinstance(left, Category) and isinstance(right, Category):
            return self._head_is_left(left, right)
        else:
            return False


cpdef unary_rule():
    return Combinator.from_ptr(<Op>new UnaryRule())

cpdef conjunction():
    return Combinator.from_ptr(<Op>new Conjunction())

cpdef conjunction2():
    return Combinator.from_ptr(<Op>new Conjunction2())

cpdef coordinate():
    return Combinator.from_ptr(<Op>new Coordinate())

cpdef comma_and_verb_phrase_to_adverb():
    return Combinator.from_ptr(<Op>new CommaAndVerbPhraseToAdverb())

cpdef parenthetical_direct_speech():
    return Combinator.from_ptr(<Op>new ParentheticalDirectSpeech())

cpdef remove_punctuation(bool punct_is_left):
    return Combinator.from_ptr(<Op>new RemovePunctuation(punct_is_left))

cpdef remove_punctuation_left():
    return Combinator.from_ptr(<Op>new RemovePunctuationLeft())

cpdef forward_application():
    return Combinator.from_ptr(<Op>new ForwardApplication())

cpdef backward_application():
    return Combinator.from_ptr(<Op>new BackwardApplication())

def slash_check(slashes):
    for slash in slashes:
        if slash not in ['/', '\\']:
            raise RuntimeError(f'slashes must be either of "\\" or "/": [{", ".join(slashes)}]')

cpdef forward_composition(str left, str right, str result):
    slash_check([left, right, result])
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new ForwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0])))

cpdef backward_composition(str left, str right, str result):
    slash_check([left, right, result])
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new BackwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0])))

cpdef generalized_forward_composition(str left, str right, str result):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new GeneralizedForwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0])))

cpdef generalized_backward_composition(str left, str right, str result):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new GeneralizedBackwardComposition(Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0])))

cpdef headfirst_combinator(Combinator combinator):
    return Combinator.from_ptr((<Op>new HeadFirstCombinator(combinator.op_)))

cpdef headfinal_combinator(Combinator combinator):
    return Combinator.from_ptr((<Op>new HeadFinalCombinator(combinator.op_)))

cpdef en_forward_application():
    return Combinator.from_ptr(<Op>new ENForwardApplication())

cpdef en_backward_application():
    return Combinator.from_ptr(<Op>new ENBackwardApplication())

cpdef conjoin():
    return Combinator.from_ptr(<Op>new Conjoin())

cpdef ja_forward_application():
    return Combinator.from_ptr(<Op>new JAForwardApplication())

cpdef ja_backward_application():
    return Combinator.from_ptr(<Op>new JABackwardApplication())

cpdef ja_generalized_backward_composition0(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedBackwardComposition0(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef ja_generalized_backward_composition1(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedBackwardComposition1(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef ja_generalized_backward_composition2(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedBackwardComposition2(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef ja_generalized_backward_composition3(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedBackwardComposition3(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef ja_generalized_forward_composition0(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedForwardComposition0(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef ja_generalized_forward_composition1(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedForwardComposition1(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef ja_generalized_forward_composition2(str left, str right, str result, str name):
    cdef string c_left = left.encode('utf-8')
    cdef string c_right = right.encode('utf-8')
    cdef string c_result = result.encode('utf-8')
    cdef string c_name = name.encode('utf-8')
    return Combinator.from_ptr(
        <Op>new JAGeneralizedForwardComposition2(
            Slash(c_left[0]), Slash(c_right[0]), Slash(c_result[0]), c_name))

cpdef remove_disfluency():
    return Combinator.from_ptr(<Op>new RemoveDisfluency())

cpdef unknown_combinator():
    return Combinator.from_ptr(<Op>new UnknownCombinator())


cdef vector[Op] combinator_list_to_vector(list combinators):
    cdef vector[Op] results
    cdef Combinator combinator
    for combinator in combinators:
        results.push_back(combinator.op_)
    return results


en_default_binary_rules = [
    headfirst_combinator(en_forward_application()),
    headfirst_combinator(en_backward_application()),
    headfirst_combinator(forward_composition('/', '/', '/')),
    headfirst_combinator(backward_composition('/', '\\', '/')),
    headfirst_combinator(generalized_forward_composition('/', '/', '/')),
    headfirst_combinator(generalized_backward_composition('/', '/', '/')),
    headfirst_combinator(conjunction()),
    headfirst_combinator(conjunction2()),
    headfirst_combinator(remove_punctuation(False)),
    headfirst_combinator(remove_punctuation(True)),
    headfirst_combinator(remove_punctuation_left()),
    headfirst_combinator(comma_and_verb_phrase_to_adverb()),
    headfirst_combinator(parenthetical_direct_speech())
]


ja_default_binary_rules = [
    headfinal_combinator(conjoin()),
    headfinal_combinator(ja_forward_application()),
    headfinal_combinator(ja_backward_application()),
    headfinal_combinator(ja_generalized_forward_composition0('/', '/', '/', '>B')),
    headfinal_combinator(ja_generalized_backward_composition0('\\', '\\', '\\', '<B1')),
    headfinal_combinator(ja_generalized_backward_composition1('\\', '\\', '\\', '<B2')),
    headfinal_combinator(ja_generalized_backward_composition2('\\', '\\', '\\', '<B3')),
    headfinal_combinator(ja_generalized_backward_composition3('\\', '\\', '\\', '<B4')),
    headfinal_combinator(ja_generalized_forward_composition0('/', '\\', '\\', '>Bx1')),
    headfinal_combinator(ja_generalized_forward_composition1('/', '\\', '\\', '>Bx2')),
    headfinal_combinator(ja_generalized_forward_composition2('/', '\\', '\\', '>Bx3')),
]
