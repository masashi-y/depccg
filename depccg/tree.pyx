from libcpp.string cimport string
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr, make_shared
from .cat cimport Cat, Category
from .combinator import unknown_combinator
from lxml import etree
from .tokens import Token
from depccg.utils import denormalize, normalize


## TODO: ugly code
cdef ResolveCombinatorName(const Node* tree, bytes lang):
    cdef string res;
    if lang == b"en":
        res = EnResolveCombinatorName(tree)
    elif lang == b"ja":
        res = JaResolveCombinatorName(tree)
    else:
        res = b"error: " + lang
    return res.decode("utf-8")


unknown_op = unknown_combinator()


class _AutoLineReader(object):
    def __init__(self, line, lang):
        self.line = line
        self.index = 0
        self.word_id = -1
        self.lang = lang
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
        return Tree.make_terminal(word, cat, self.word_id, self.lang)

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
            return Tree.make_binary(
                cat, left_is_head, left, right, unknown_op, self.lang)
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
    def make_terminal(str word, Category cat, int position, lang):
        cdef string c_word = word.encode('utf-8')
        cdef NodeType node = <NodeType>make_shared[Leaf](c_word, cat.cat_, position)
        return Tree.from_ptr(node, lang)

    @staticmethod
    def make_binary(Category cat, bool left_is_head, Tree left, Tree right, Combinator op, lang):
        cdef NodeType node = <NodeType>make_shared[CTree](
            cat.cat_, left_is_head, left.node_, right.node_, op.op_)
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
        position = [-1]
        def rec(node):
            cat = Category.parse(node.label())
            if isinstance(node[0], str):
                word = node[0]
                position[0] += 1
                return Tree.make_terminal(word, cat, position[0], lang)
            else:
                children = [rec(child) for child in node]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0], lang)
                else:
                    assert len(children) == 2
                    left, right = children
                    return Tree.make_binary(
                        cat, True, left, right, unknown_op, lang)
        return rec(tree)

    def __cinit__(self):
        self.suppress_feat = False

    property cat:
        def __get__(self):
            return Category.from_ptr(deref(self.node_).GetCategory())

    property op_string:
        def __get__(self):
            assert not self.is_leaf, "This node is leaf and does not have combinator!"
            cdef const Node* c_node = &deref(self.node_)
            return ResolveCombinatorName(c_node, self.lang)

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

    property start_of_span:
        def __get__(self):
            return deref(self.node_).GetStartOfSpan()

    property word:
        def __get__(self):
            cdef string res = deref(self.node_).GetWord()
            return res.decode("utf-8")

    property head_id:
        def __get__(self):
            return deref(self.node_).GetHeadId()

    property dependency_length:
        def __get__(self):
            return deref(self.node_).GetDependencyLength()

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

    def deriv(self):
        cdef string res = Derivation(self.node_, not self.suppress_feat).Get()
        return res.decode("utf-8")

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

    def ja(self):
        cdef string res = JaCCG(self.node_).Get()
        return res.decode("utf-8")

    def conll(self):
        cdef string res = CoNLL(self.node_).Get()
        return res.decode("utf-8")

