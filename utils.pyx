
cpdef str drop_brackets(str cat):
    if cat.startswith('(') and \
        cat.endswith(')') and \
        find_closing_bracket(cat, 0) == len(cat)-1:
        return cat[1:-1]
    else:
        return cat


cpdef int find_closing_bracket(str source, int start) except -1:
    cdef int open_brackets = 0
    for i, c in enumerate(source):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1

        if open_brackets == 0:
            return i

    raise Exception("Mismatched brackets in string")

cpdef int find_non_nested_char(str haystack, str needles):
    cdef int open_brackets = 0

    for i, c in enumerate(heystack):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1
        elif open_brackets == 0:
            if any(n == c for n in needles):
                return i
    return -1
