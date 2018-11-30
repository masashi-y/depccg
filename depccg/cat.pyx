from libcpp.string cimport string
from libcpp.pair cimport pair


cdef extern from "feat.h" namespace "myccg" nogil:
    ctypedef const Feature* Feat
    cdef cppclass Feature:
        string ToStr() const
        bint IsEmpty() const
        bint Matches(Feat other) const
        bint ContainsWildcard() const
        string SubstituteWildcard(const string& string) const
        bint ContainsKeyValue(const string& key, const string& value) const
        Feat ToMultiValue() const


cdef extern from "cat.h" namespace "myccg" nogil:
    cdef cppclass Slash:
        bint IsForward() const
        bint IsBackward() const
        string ToStr() const

    ctypedef const Category* Cat
    ctypedef pair[Cat, Cat] CatPair
    cdef cppclass Category:
        @staticmethod
        Cat Parse(const string& cat)
        string ToStr()
        string ToStrWithoutFeat()

        Cat StripFeat() const
        Cat StripFeat(string& f1) const
        Cat StripFeat(string& f1, string& f2) const
        Cat StripFeat(string& f1, string& f2, string& f3) const
        Cat StripFeat(string& f1, string& f2, string& f3, string& f4) const

        const string& GetType() const
        Feat GetFeat() const
        Cat GetLeft() const
        Cat GetRight() const

        Slash GetSlash() const

        const string WithBrackets() const
        bint IsModifier() const
        bint IsModifierWithoutFeat() const
        bint IsTypeRaised() const
        bint IsTypeRaisedWithoutFeat() const
        bint IsForwardTypeRaised() const
        bint IsBackwardTypeRaised() const
        bint IsFunctor() const
        bint IsPunct() const
        bint IsNorNP() const
        int NArgs() const
        bint Matches(Cat other) const
        Cat Arg(int argn) except +
        Cat LeftMostArg() const
        bint IsFunctionInto(Cat cat) const
        Cat ToMultiValue() const


cdef class PyCat:
    cdef Cat cat_

    def __cinit__(self):
        pass

    def __str__(self):
        return self.cat_.ToStr().decode('utf-8')

    def __repr__(self):
        return self.cat_.ToStr().decode('utf-8')

    @staticmethod
    def parse(cat):
        if not isinstance(cat, bytes):
            cat = cat.encode('utf-8')
        c = PyCat()
        c.cat_ = Category.Parse(cat)
        return c

    @staticmethod
    cdef PyCat from_ptr(Cat cat):
        c = PyCat()
        c.cat_ = cat
        return c

    property multi_valued:
        def __get__(self):
            return PyCat.from_ptr(self.cat_.ToMultiValue())

    property without_feat:
        def __get__(self):
            return self.cat_.ToStrWithoutFeat()

    property left:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            return PyCat.from_ptr(self.cat_.GetLeft())

    property right:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            return PyCat.from_ptr(self.cat_.GetRight())

    property is_modifier:
        def __get__(self):
            return self.cat_.IsModifier()

    property is_modifier_without_feat:
        def __get__(self):
            return self.cat_.IsModifierWithoutFeat()

    property is_type_raised:
        def __get__(self):
            return self.cat_.IsTypeRaised()

    property is_type_raised_without_feat:
        def __get__(self):
            return self.cat_.IsTypeRaisedWithoutFeat()

    property is_forward_type_raised:
        def __get__(self):
            return self.cat_.IsForwardTypeRaised()

    property is_backward_type_raised:
        def __get__(self):
            return self.cat_.IsBackwardTypeRaised()

    property is_functor:
        def __get__(self):
            return self.cat_.IsFunctor()

    property is_punct:
        def __get__(self):
            return self.cat_.IsPunct()

    property is_NorNP:
        def __get__(self):
            return self.cat_.IsNorNP()

    def is_function_into(self, cat):
        return self._is_function_into(cat)

    cdef bint _is_function_into(self, PyCat cat):
        return self.cat_.IsFunctionInto(cat.cat_)

    property n_args:
        def __get__(self):
            return self.cat_.NArgs()

    def matches(self, other):
        return self._matches(other)

    cdef bint _matches(self, PyCat other):
        return self.cat_.Matches(other.cat_)

    def arg(self, i):
        return PyCat.from_ptr(self.cat_.Arg(i))

    property slash:
        def __get__(self):
            assert self.is_functor, 'Category "{}" is not a functor type.'.format(str(self))
            return self.cat_.GetSlash().ToStr()

    cpdef strip_feat(self, feat):
        # assert len(feats) <= 4, 'Pycat.strip_feat does not stripping more than 4 features.'
        cdef string c_feat = feat.encode('utf-8')
        return PyCat.from_ptr(self.cat_.StripFeat(c_feat))

