
cdef enum RuleType:
    FA = 0
    BA = 1
    FC = 2
    BX = 3
    GFC = 4
    GBX = 5
    CONJ = 6
    RP = 7
    LP = 8
    NOISE = 9
    UNARY = 10
    LEXICON = 11
    NONE = 12

# cdef struct LeafC:
#     char* word
#     int pos
#     Cat cat
#     int rule_type
#
# cdef struct TreeC:
#     void* lchild
#     void* rchild
#     bint left_is_head
#     Combinator op
#     Cat cat
#     int rule_type
#
# cdef struct NodeC:



