
cimport numpy as np

ctypedef np.float32_t FLOAT_T

cpdef str drop_brackets(str cat)

cpdef int find_closing_bracket(str source, int start) except -1

cpdef int find_non_nested_char(str haystack, str needles)

cpdef list get_context_by_window(list items, int window_size, object lpad, object rpad)

cpdef np.ndarray[FLOAT_T, ndim=2] read_pretrained_embeddings(str filepath)

cpdef dict read_model_defs(str filepath)

cdef np.ndarray[FLOAT_T, ndim=2] compute_outsize_probs(list supertags)

cpdef dict load_unary(str filename)

cpdef dict load_seen_rules(str filename)
