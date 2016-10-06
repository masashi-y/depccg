
cpdef str drop_brackets(str cat):
    if cat.startswith('(') and
        cat.endswith(')') and
        find_closing_bracket(cat, 0) == len(cat):
        return cat[1:-1]
    else:
        return cat


cdef int find_closing_bracket(str source, int start):
    cdef int open_brackets = 0
    for i, c in enumerate(source):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1

        if open_brackets == 0:
            return i

    raise Exception("Mismatched brackets in string": source)
