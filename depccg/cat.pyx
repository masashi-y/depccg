from libcpp.string cimport string
from libcpp.pair cimport pair


cdef class Category:
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
        c = Category()
        c.cat_ = CCategory.Parse(cat)
        return c

    @staticmethod
    cdef Category from_ptr(Cat cat):
        c = Category()
        c.cat_ = cat
        return c

    property multi_valued:
        def __get__(self):
            return Category.from_ptr(self.cat_.ToMultiValue())

    property without_feat:
        def __get__(self):
            return self.cat_.ToStrWithoutFeat()

    property left:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            return Category.from_ptr(self.cat_.GetLeft())

    property right:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            return Category.from_ptr(self.cat_.GetRight())

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

    cdef bint _is_function_into(self, Category cat):
        return self.cat_.IsFunctionInto(cat.cat_)

    property n_args:
        def __get__(self):
            return self.cat_.NArgs()

    def matches(self, other):
        return self._matches(other)

    cdef bint _matches(self, Category other):
        return self.cat_.Matches(other.cat_)

    def arg(self, i):
        return Category.from_ptr(self.cat_.Arg(i))

    property slash:
        def __get__(self):
            assert self.is_functor, 'Category "{}" is not a functor type.'.format(str(self))
            return self.cat_.GetSlash().ToStr()

    cpdef strip_feat(self, feat):
        # assert len(feats) <= 4, 'Pycat.strip_feat does not stripping more than 4 features.'
        cdef string c_feat = feat.encode('utf-8')
        return Category.from_ptr(self.cat_.StripFeat(c_feat))

