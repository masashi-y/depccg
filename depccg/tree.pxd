from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp cimport bool
from .cat cimport Cat, CatPair
from .combinator cimport Op, Combinator

cdef extern from "tree.h" namespace "myccg" nogil:
    cdef cppclass Node:
        Cat GetCategory() const
        const int GetLength() const
        shared_ptr[const Node] GetLeftChild() const
        shared_ptr[const Node] GetRightChild() const
        bint IsLeaf() const
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

    cdef cppclass Leaf(Node):
        Leaf(const string&, Cat, int)

    cdef cppclass CTree(Node):
        CTree(Cat, bool, NodeType, NodeType, Op)
        CTree(Cat, NodeType)

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

    string EnResolveCombinatorName(const Node*)
    string JaResolveCombinatorName(const Node*)


cdef class Tree:
    cdef NodeType node_
    cdef public bint suppress_feat
    cdef bytes lang

    @staticmethod
    cdef Tree from_ptr(NodeType node, lang)

