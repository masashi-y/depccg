
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

    for i, c in enumerate(haystack):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1
        elif open_brackets == 0:
            for n in needles:
                if n == c: return i
    return -1

cpdef list get_context_by_window(
        list items, int window_size, object lpad=None, object rpad=None):
    cdef list res = []
    cdef list context
    cdef int i, j
    cdef object item
    for i, item in enumerate(items):
        context = []
        if window_size - i > 0:
            for j in xrange(window_size - i):
                context.append(lpad)
            for j in xrange(i):
                context.append(items[j])
        else:
            for j in xrange(i - window_size, i):
                context.append(items[j])
        context.append(item)
        if i + window_size >= len(items):
            for j in xrange(i + 1, len(items)):
                context.append(items[j])
            for j in xrange(i + window_size - len(items) + 1):
                context.append(rpad)
        else:
            for j in xrange(i + 1, i + window_size + 1):
                context.append(items[j])
        assert len(context) == window_size * 2 + 1

        res.append(context)
    return res

