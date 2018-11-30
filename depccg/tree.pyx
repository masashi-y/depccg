from libcpp.string cimport string
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
from .cat cimport Cat, Category


cdef extern from "grammar.h" namespace "myccg" nogil:
    cdef cppclass En:
        @staticmethod
        string ResolveCombinatorName(const Node*)

    cdef cppclass Ja:
        @staticmethod
        string ResolveCombinatorName(const Node*)


## TODO: ugly code
cdef ResolveCombinatorName(const Node* tree, bytes lang):
    cdef string res;
    if lang == b"en":
        res = En.ResolveCombinatorName(tree)
    elif lang == b"ja":
        res = Ja.ResolveCombinatorName(tree)
    else:
        res = b"error: " + lang
    return res.decode("utf-8")


cdef class Tree:
    @staticmethod
    cdef Tree from_ptr(NodeType node, lang):
        p = Tree()
        p.node_ = node
        p.lang = lang
        return p

    def __cinit__(self):
        self.suppress_feat = False

    property cat:
        def __get__(self):
            return Category.from_ptr(deref(self.node_).GetCategory())

    property op_string:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have combinator!"
            cdef const Node* c_node = &deref(self.node_)
            return ResolveCombinatorName(c_node, self.lang)

    def __len__(self):
        return deref(self.node_).GetLength()

    property children:
        def __get__(self):
            res = [self.left_child]
            if not self.is_unary:
                res.append(self.right_child)
            return res

    property left_child:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have any child!"
            return Tree.from_ptr(<NodeType>deref(self.node_).GetLeftChild(), self.lang)

    property right_child:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have any child!"
            assert not self.is_unary, "This node does not have right child!"
            return Tree.from_ptr(<NodeType>deref(self.node_).GetRightChild(), self.lang)

    property is_leaf:
        def __get__(self):
            return deref(self.node_).IsLeaf()

    property start_of_span:
        def __get__(self):
            return deref(self.node_).GetStartOfSpan()

    property word:
        def __get__(self):
            cdef string res = deref(self.node_).GetWord()
            return res.decode("utf-8")

    property head_id:
        def __get__(self):
            return deref(self.node_).GetHeadId()

    property dependency_length:
        def __get__(self):
            return deref(self.node_).GetDependencyLength()

    property head_is_left:
        def __get__(self):
            return deref(self.node_).HeadIsLeft()

    property is_unary:
        def __get__(self):
            return deref(self.node_).IsUnary()

    property num_descendants:
        def __get__(self):
            return deref(self.node_).NumDescendants()

    property right_num_descendants:
        def __get__(self):
            return deref(self.node_).RightNumDescendants()

    property left_num_descendants:
        def __get__(self):
            return deref(self.node_).LeftNumDescendants()

    def __str__(self):
        return self.auto

    def __repr__(self):
        return self.auto

    property auto:
        def __get__(self):
            cdef string res = AUTO(self.node_).Get()
            return res.decode("utf-8")

    property deriv:
        def __get__(self):
            cdef string res = Derivation(self.node_, not self.suppress_feat).Get()
            return res.decode("utf-8")

    property xml:
        def __get__(self):
            cdef string res = PyXML(self.node_, not self.suppress_feat).Get()
            return res.decode("utf-8")

    property prolog:
        def __get__(self):
            cdef string res = Prolog(self.node_).Get()
            return res.decode("utf-8")

    property ja:
        def __get__(self):
            cdef string res = JaCCG(self.node_).Get()
            return res.decode("utf-8")

    property conll:
        def __get__(self):
            cdef string res = CoNLL(self.node_).Get()
            return res.decode("utf-8")

