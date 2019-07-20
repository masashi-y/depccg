from libcpp.string cimport string
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref


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

    cdef bool equals_to(self, Category other):
        return deref(self.cat_) == deref(other.cat_)

    def __eq__(self, other):
        if isinstance(other, Category):
            return self.equals_to(other)
        else:
            return False

    def __hash__(self):
        return self.cat_.GetId()

    property multi_valued:
        def __get__(self):
            return Category.from_ptr(self.cat_.ToMultiValue())

    property base:
        def __get__(self):
            return self.cat_.ToStrWithoutFeat().decode('utf-8')

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
            return self.cat_.GetSlash().ToStr().decode('utf-8')

    property features:
        def __get__(self):
            cdef unordered_map[string, string] c_features
            cdef pair[string, string] tmp
            cdef str key, val
            if self.is_functor:
                return {}
            else:
                c_features = self.cat_.GetFeat().Values()
                res = {}
                for tmp in c_features:
                    key = tmp.first.decode('utf-8')
                    val = tmp.second.decode('utf-8')
                    res[key] = val
                return res


    cpdef strip_feat(self, feat):
        assert feat.startswith('[') and feat.endswith(']'), \
            'please enclose a feature with [] as in Category.parse(\'S[dcl]\\NP\').strip_feat(\'[dcl]\')'
        cdef string c_feat = feat.encode('utf-8')
        return Category.from_ptr(self.cat_.StripFeat(c_feat))

    def json(self):
        def rec(node):
            if node.is_functor:
                return {
                    'slash': node.slash,
                    'left': rec(node.left),
                    'right': rec(node.right)
                }
            else:
                feature = node.features
                return {
                    'base': node.base,
                    'feature': feature if len(feature) > 0 else None
                }
        return rec(self)

