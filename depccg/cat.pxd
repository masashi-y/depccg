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

    ctypedef const CCategory* Cat
    ctypedef pair[Cat, Cat] CatPair
    cdef cppclass CCategory:
        @staticmethod
        Cat Parse(const string& cat)
        string ToStr()
        string ToStrWithoutFeat()

        Cat StripFeat() const
        Cat StripFeat(string& f1) const
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


cdef class Category:
    cdef Cat cat_

    @staticmethod
    cdef Category from_ptr(Cat cat)

    cdef bint _matches(self, Category other)

    cdef bint _is_function_into(self, Category cat)

    cpdef strip_feat(self, feat)

