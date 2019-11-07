from libcpp.string cimport string
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr, make_shared
from lxml import etree
from io import StringIO
from .cat cimport Cat, Category
from .combinator import UNKNOWN_COMBINATOR, guess_combinator_by_triplet
from .tokens import Token
from .utils import denormalize, normalize
from .lang import BINARY_RULES


class _AutoLineReader(object):
    def __init__(self, line, lang):
        self.line = line
        self.index = 0
        self.word_id = -1
        self.lang = lang
        self.binary_rules = BINARY_RULES[self.lang]
        self.tokens = []

    def next(self):
        end = self.line.find(' ', self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError(f'failed to parse: {self.line}')

    def peek(self):
        return self.line[self.index]

    def parse(self):
        tree = self.next_node()
        return tree, self.tokens

    @property
    def next_node(self):
        if self.line[self.index+2] == 'L':
            return self.parse_leaf
        elif self.line[self.index+2] == 'T':
            return self.parse_tree
        else:
            raise RuntimeError(f'failed to parse: {self.line}')

    def parse_leaf(self):
        self.word_id += 1
        self.check('(')
        self.check('<', 1)
        self.check('L', 2)
        self.next()
        cat = Category.parse(self.next())
        tag1 = self.next()  # modified POS tag
        tag2 = self.next()  # original POS
        word = self.next().replace('\\', '')
        token = Token(word=word, pos=tag1, tag1=tag1, tag2=tag2)
        self.tokens.append(token)
        if word == '-LRB-':
            word = "("
        elif word == '-RRB-':
            word = ')'
        dep = self.next()[:-2]
        return Tree.make_terminal(word, cat, self.lang)

    def parse_tree(self):
        self.check('(')
        self.check('<', 1)
        self.check('T', 2)
        self.next()
        cat = Category.parse(self.next())
        left_is_head = self.next() == '0'
        self.next()
        children = []
        while self.peek() != ')':
            children.append(self.next_node())
        self.next()
        if len(children) == 2:
            left, right = children
            op = guess_combinator_by_triplet(self.binary_rules, cat, left.cat, right.cat)
            op = op or UNKNOWN_COMBINATOR
            return Tree.make_binary(
                cat, left, right, op, self.lang, left_is_head=left_is_head)
        elif len(children) == 1:
            return Tree.make_unary(cat, children[0], self.lang)
        else:
            raise RuntimeError(f'failed to parse: {self.line}')


cdef class Tree:
    @staticmethod
    cdef Tree from_ptr(NodeType node, lang):
        if isinstance(lang, str):
            lang = lang.encode('utf-8')
        p = Tree()
        p.node_ = node
        p.lang = lang
        return p

    @staticmethod
    def make_terminal(str word, Category cat, lang):
        cdef string c_word = word.encode('utf-8')
        cdef NodeType node = <NodeType>make_shared[Leaf](c_word, cat.cat_)
        return Tree.from_ptr(node, lang)

    @staticmethod
    def make_binary(Category cat, Tree left, Tree right, Combinator op, lang, left_is_head=None):
        cdef NodeType node
        cdef bool cleft_is_head
        if left_is_head is None:
            left_is_head = op.head_is_left(left, right)
        cleft_is_head = left_is_head
        node = <NodeType>make_shared[CTree](cat.cat_, cleft_is_head, left.node_, right.node_, op.op_)
        return Tree.from_ptr(node, lang)

    @staticmethod
    def make_unary(Category cat, Tree child, lang):
        cdef NodeType node = <NodeType>make_shared[CTree](cat.cat_, child.node_)
        return Tree.from_ptr(node, lang)

    @staticmethod
    def of_auto(line, lang='en'):
        return _AutoLineReader(line, lang).parse()

    @staticmethod
    def of_nltk_tree(tree, lang='en'):
        def rec(node):
            cat = Category.parse(node.label())
            if isinstance(node[0], str):
                word = node[0]
                return Tree.make_terminal(word, cat, lang)
            else:
                children = [rec(child) for child in node]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0], lang)
                else:
                    assert len(children) == 2
                    left, right = children
                    op = guess_combinator_by_triplet(cat, left.cat, right.cat)
                    op = op or UNKNOWN_COMBINATOR
                    return Tree.make_binary(cat, left, right, op, lang)
        return rec(tree)

    def __cinit__(self):
        self.suppress_feat = False

    property cat:
        def __get__(self):
            return Category.from_ptr(deref(self.node_).GetCategory())

    property op_string:
        """C&C-style string representing a combinator. e.g. fa, ba fx, bx"""
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have combinator!"
            cdef const Node* c_node = &deref(self.node_)
            return EnResolveCombinatorName(c_node).decode('utf-8')

    property op_symbol:
        """standard CCG style string representing a combinator. e.g. >, <, >B"""
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have combinator!"
            cdef const CTree* c_node = <const CTree*>&deref(self.node_)
            # tentatively put this here
            if self.lang == b'ja' and self.is_unary:
                child_features = self.child.cat.arg(0).features.items()
                if ('mod', 'adn') in child_features:
                    if self.child.cat.base == 'S':
                        return 'ADNext'
                    else:
                        return 'ADNint'
                elif ('mod', 'adv') in child_features:
                    if self.cat.base == '(S\\NP)/(S\\NP)':
                        return 'ADV1'
                    else:
                        return 'ADV0'
                # else:
                #     raise RuntimeError('this tree is not supported in `ja` format')
            return c_node.GetRule().ToStr().decode('utf-8')

    def __len__(self):
        return deref(self.node_).GetLength()

    property children:
        def __get__(self):
            if self.is_leaf:
                return []
            res = [self.left_child]
            if not self.is_unary:
                res.append(self.right_child)
            return res

    property leaves:
        def __get__(self):
            def rec(node):
                if node.is_leaf:
                    res.append(node)
                else:
                    for child in node.children:
                        rec(child)
            res = []
            rec(self)
            return res

    property child:
        def __get__(self):
            assert self.is_unary, "This node is not unary node! Please use `Tree.children`"
            return self.left_child

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

    property word:
        def __get__(self):
            cdef string res = deref(self.node_).GetWord()
            return res.decode("utf-8")

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
        return self.auto()

    def __repr__(self):
        return self.auto()

    def ptb(self):
        def rec(node):
            if node.is_leaf:
                cat = node.cat
                word = node.word
                return f'({cat} {word})'
            else:
                cat = node.cat
                children = ' '.join(rec(child) for child in node.children)
                return f'({cat} {children})'
        return f'(ROOT {rec(self)})'

    def nltk_tree(self):
        from nltk.tree import Tree
        def rec(node):
            if node.is_leaf:
                cat = node.cat
                children = [node.word]
            else:
                cat = node.cat
                children = [rec(child) for child in node.children]
            return Tree(str(cat), children)
        return rec(self)

    def auto(self, tokens=None):
        def rec(node):
            if node.is_leaf:
                cat = node.cat
                word = denormalize(node.word)
                pos = poss.pop(0)
                return f'(<L {cat} {pos} {pos} {word} {cat}>)'
            else:
                cat = node.cat
                children = ' '.join(rec(child) for child in node.children)
                num_children = len(node.children)
                head_is_left = 0 if node.head_is_left else 1
                return f'(<T {cat} {head_is_left} {num_children}> {children} )'
        if tokens:
            poss = [token.pos for token in tokens]
        else:
            poss = ['POS' for _ in range(len(self))]
        return rec(self)

    def auto_flat(self, tokens=None):
        def rec(node):
            if node.is_leaf:
                cat = node.cat
                word = normalize(node.word)
                word = word.replace('/', '\\/')
                pos = poss.pop(0)
                return f'(<L *** {cat} {pos} {word}>\n)'
            else:
                cat = node.cat
                children = '\n'.join(rec(child) for child in node.children)
                num_children = len(node.children)
                head_is_left = 0 if node.head_is_left else 1
                return f'(<T *** {cat} * {head_is_left} {num_children}>\n{children}\n)'
        if tokens:
            poss = [token.pos for token in tokens]
        else:
            poss = ['POS' for _ in range(len(self))]
        return f'###\n{rec(self)}\n'

    def auto_extended(self, tokens=None):
        def rec(node):
            if node.is_leaf:
                cat = node.cat
                word = denormalize(node.word)
                token = tokens.pop(0)
                lemma = token.get('lemma', 'XX')
                pos = token.get('pos', 'XX')
                entity = token.get('entity', 'XX')
                chunk = token.get('chunk', 'XX')
                return f'(<L {cat} {word} {lemma} {pos} {entity} {chunk} {cat}>)'
            else:
                cat = node.cat
                children = ' '.join(rec(child) for child in node.children)
                num_children = len(node.children)
                head_is_left = 0 if node.head_is_left else 1
                rule = node.op_string
                return f'(<T {cat} {rule} {head_is_left} {num_children}> {children} )'
        if tokens is None:
            tokens = [Token.from_word(word) for word in self.word.split(' ')]
        return rec(self)

    def deriv(self):
        catstr  = ''
        wordstr = ''
        for leaf in self.leaves:
            str_cat = str(leaf.cat)
            str_word = leaf.word
            nextlen = 2 + max(len(str_word), len(str_cat))
            lcatlen = (nextlen - len(str_cat)) // 2
            rcatlen = lcatlen + (nextlen - len(str_cat)) % 2
            catstr += ' ' * lcatlen + str_cat + ' ' * rcatlen
            lwordlen = (nextlen - len(str_word)) // 2
            rwordlen = lwordlen + (nextlen - len(str_word)) % 2
            wordstr += ' ' * lwordlen + str_word + ' ' * rwordlen

        def rec(lwidth, node):
            rwidth = lwidth

            if node.is_leaf:
                return max(rwidth,
                           2 + lwidth + len(str(node.cat)),
                           2 + lwidth + len(node.word))
            else:
                for child in node.children:
                    rwidth = max(rwidth, rec(rwidth, child))
                op = node.op_symbol
                print(lwidth * ' ' + (rwidth - lwidth) * '-' + str(op), file=output)
                str_res = str(node.cat)
                respadlen = (rwidth - lwidth - len(str_res)) // 2 + lwidth
                print(respadlen * ' ' + str_res, file=output)
                return rwidth

        with StringIO() as output:
            print(catstr.rstrip(), file=output)
            print(wordstr.rstrip(), file=output)
            rec(0, self)
            return output.getvalue()

    def xml(self, tokens=None):
        def rec(node, parent):
            if node.is_leaf:
                leaf_node = etree.SubElement(parent, 'lf')
                start, token = tokens.pop(0)
                leaf_node.set('start', str(start))
                leaf_node.set('span', '1')
                leaf_node.set('cat', str(node.cat))
                for k, v in token.items():
                    leaf_node.set(k, v)
            else:
                rule_node = etree.SubElement(parent, 'rule')
                rule_node.set('type', node.op_string)
                rule_node.set('cat', str(node.cat))
                for child in node.children:
                    rec(child, rule_node)

        if tokens is None:
            tokens = [Token.from_word(word) for word in self.word.split(' ')]
        tokens = list(enumerate(tokens))
        res = etree.Element("ccg")
        rec(self, res)
        return res

    def json(self, tokens=None, full=False):
        def rec(node):
            if node.is_leaf:
                token = tokens.pop(0)
                res = dict(token)
                res['cat'] = node.cat.json if full else str(node.cat)
                return res
            else:
                return {
                    'type': node.op_string,
                    'cat': node.cat.json if full else str(node.cat),
                    'children': [rec(child) for child in node.children]
                }

        if tokens is None:
            tokens = [Token.from_word(word) for word in self.word.split(' ')]
        tokens = list(tokens)
        return rec(self)

    def prolog(self):
        cdef string res = Prolog(self.node_).Get()
        return res.decode("utf-8")

    def ja(self, tokens=None):
        def rec(node):
            if node.is_leaf:
                cat = node.cat
                word = normalize(node.word)
                token = tokens.pop(0)
                base = token.get('base', word)
                poss = [token.get(pos, '*') for pos in ('pos', 'pos1', 'pos2', 'pos3')]
                poss = [pos for pos in poss if pos != '*']
                pos = '-'.join(poss) if len(poss) else '_'
                inflections = [token.get(i, '*') for i in ('inflectionForm', 'inflectionType')]
                inflections = [i for i in inflections if i != '*']
                inflection = '-'.join(inflections) if len(inflections) else '_'
                return f'{{{cat} {word}/{word}/{pos}/{inflection}}}'
            else:
                children = ' '.join(rec(child) for child in node.children)
                return f'{{{node.op_symbol} {node.cat} {children}}}'
        if tokens is None:
            tokens = [Token.from_word(word) for word in self.word.split(' ')]
        return rec(self)

    def conll(self, tokens=None):
        stack = []
        counter = 1
        dependencies = []
        def resolve_dependencies(node):
            if node.is_leaf:
                index = len(dependencies)
                dependencies.append(-1)
                return index
            else:
                if node.is_unary:
                    return resolve_dependencies(node.child)
                else:
                    left_head = resolve_dependencies(node.left_child)
                    right_head = resolve_dependencies(node.right_child)
                    if node.head_is_left:
                        dependencies[right_head] = left_head
                        return left_head
                    else:
                        dependencies[left_head] = right_head
                        return right_head

        def rec(node):
            nonlocal stack, counter, dependencies
            if node.is_leaf:
                cat = node.cat
                word = denormalize(node.word)
                token = tokens.pop(0)
                lemma = token.get('lemma', '_')
                pos = token.get('pos', '_')
                stack.append(f'(<L {cat} {pos} {pos} {word} {cat}>)')
                subtree = ' '.join(stack)
                line = '\t'.join(
                        (str(counter), word, lemma, pos, pos, '_', str(dependencies[counter-1] + 1), str(cat), '_', subtree))
                stack = []
                counter += 1
                return line
            else:
                cat = node.cat
                num_children = len(node.children)
                head_is_left = 0 if node.head_is_left else 1
                stack.append(f'(<T {cat} {head_is_left} {num_children}>')
                children = '\n'.join(rec(child) for child in node.children) + ' )'
                return children
        if tokens is None:
            tokens = [Token.from_word(word) for word in self.word.split(' ')]
        resolve_dependencies(self)
        assert len([dependency for dependency in dependencies if dependency == -1]) == 1
        return rec(self)


