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
        string GetWord() const
        bint HeadIsLeft() const
        bint IsUnary() const
        int NumDescendants() const
        int RightNumDescendants() const
        int LeftNumDescendants() const

    ctypedef shared_ptr[const Node] NodeType
    ctypedef pair[NodeType, float] ScoredNode

    cdef cppclass Leaf(Node):
        Leaf(const string&, Cat)

    cdef cppclass CTree(Node):
        CTree(Cat, bool, NodeType, NodeType, Op)
        CTree(Cat, NodeType)
        Op GetRule()

    cdef cppclass Prolog:
        Prolog(NodeType tree)
        string Get()

    string EnResolveCombinatorName(const Node*)


cdef class Tree:
    cdef NodeType node_
    cdef public bint suppress_feat
    cdef bytes lang

    @staticmethod
    cdef Tree from_ptr(NodeType node, lang)

