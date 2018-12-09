from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from .cat cimport Cat, CatPair


cdef vector[Cat] cat_list_to_vector(list cats)

cdef unordered_set[Cat] cat_list_to_unordered_set(list cats)

cdef unordered_map[string, unordered_set[Cat]] convert_cat_dict(dict cat_dict)

cdef unordered_map[Cat, unordered_set[Cat]] convert_unary_rules(list unary_rules)

cpdef read_unary_rules(filename)

cpdef read_cat_dict(filename)

cpdef read_cat_list(filename)

cpdef read_seen_rules(filename, preprocess)

cdef unordered_set[CatPair] convert_seen_rules(seen_rule_list)

cdef unordered_set[Cat] read_possible_root_categories(list cats)

