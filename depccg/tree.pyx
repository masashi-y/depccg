from libcpp.string cimport string
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr

include "cat.pyx"

cdef extern from "combinator.h" namespace "myccg" nogil:
    cdef cppclass Combinator:
        const string ToStr() const
    ctypedef const Combinator* Op


cdef extern from "tree.h" namespace "myccg" nogil:
    cdef cppclass Leaf:
        Leaf(const string& word, Cat cat, int position)

    cdef cppclass Tree:
        Op GetRule() const

    cdef cppclass Node:
        Cat GetCategory() const
        const int GetLength() const
        shared_ptr[const Node] GetLeftChild() const
        shared_ptr[const Node] GetRightChild() const
        bint IsLeaf() const
        const Leaf* GetHeadLeaf() const
        int GetStartOfSpan() const
        string GetWord() const
        int GetHeadId() const
        int GetDependencyLength() const
        bint HeadIsLeft() const
        bint IsUnary() const
        int NumDescendants() const
        int RightNumDescendants() const
        int LeftNumDescendants() const

    ctypedef shared_ptr[const Node] NodeType
    ctypedef pair[NodeType, float] ScoredNode

    cdef cppclass AUTO:
        AUTO(NodeType tree)
        string Get()

    cdef cppclass Derivation:
        Derivation(NodeType tree, bint feat)
        string Get()

    cdef cppclass JaCCG:
        JaCCG(NodeType tree)
        string Get()

    cdef cppclass PyXML:
        PyXML(NodeType tree, bint feat)
        string Get()

    cdef cppclass Prolog:
        Prolog(NodeType tree)
        string Get()

    cdef cppclass CoNLL:
        CoNLL(NodeType tree)
        string Get()


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


cdef class Parse:
    cdef NodeType node
    cdef public bint suppress_feat
    cdef bytes lang

    @staticmethod
    cdef Parse from_ptr(NodeType node, lang):
        p = Parse()
        p.node = node
        p.lang = lang
        return p

    def __cinit__(self):
        self.suppress_feat = False

    property cat:
        def __get__(self):
            return PyCat.from_ptr(deref(self.node).GetCategory())

    property op_string:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have combinator!"
            cdef const Node* c_node = &deref(self.node)
            return ResolveCombinatorName(c_node, self.lang)

    def __len__(self):
        return deref(self.node).GetLength()

    property children:
        def __get__(self):
            res = [self.left_child]
            if not self.is_unary:
                res.append(self.right_child)
            return res

    property left_child:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have any child!"
            return Parse.from_ptr(<NodeType>deref(self.node).GetLeftChild(), self.lang)

    property right_child:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have any child!"
            assert not self.is_unary, "This node does not have right child!"
            return Parse.from_ptr(<NodeType>deref(self.node).GetRightChild(), self.lang)

    property is_leaf:
        def __get__(self):
            return deref(self.node).IsLeaf()

    property start_of_span:
        def __get__(self):
            return deref(self.node).GetStartOfSpan()

    property word:
        def __get__(self):
            cdef string res = deref(self.node).GetWord()
            return res.decode("utf-8")

    property head_id:
        def __get__(self):
            return deref(self.node).GetHeadId()

    property dependency_length:
        def __get__(self):
            return deref(self.node).GetDependencyLength()

    property head_is_left:
        def __get__(self):
            return deref(self.node).HeadIsLeft()

    property is_unary:
        def __get__(self):
            return deref(self.node).IsUnary()

    property num_descendants:
        def __get__(self):
            return deref(self.node).NumDescendants()

    property right_num_descendants:
        def __get__(self):
            return deref(self.node).RightNumDescendants()

    property left_num_descendants:
        def __get__(self):
            return deref(self.node).LeftNumDescendants()

    def __str__(self):
        return self.auto

    def __repr__(self):
        return self.auto

    property auto:
        def __get__(self):
            cdef string res = AUTO(self.node).Get()
            return res.decode("utf-8")

    property deriv:
        def __get__(self):
            cdef string res = Derivation(self.node, not self.suppress_feat).Get()
            return res.decode("utf-8")

    property xml:
        def __get__(self):
            cdef string res = PyXML(self.node, not self.suppress_feat).Get()
            return res.decode("utf-8")

    property prolog:
        def __get__(self):
            cdef string res = Prolog(self.node).Get()
            return res.decode("utf-8")

    property ja:
        def __get__(self):
            cdef string res = JaCCG(self.node).Get()
            return res.decode("utf-8")

    property conll:
        def __get__(self):
            cdef string res = CoNLL(self.node).Get()
            return res.decode("utf-8")

