
from overrides import overrides
from allennlp_models.structured_prediction.dataset_readers.penn_tree_bank import \
    PennTreeBankConstituencySpanDatasetReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from depccg.tools.reader import read_auto
from nltk.tree import Tree


def ccg_to_nltk_tree(tree):
    def rec(node):
        if node.is_leaf:
            cat = node.cat
            children = [Tree('XX', [node.word])]  # add dummy pos non-terminal
        else:
            cat = node.cat
            children = [rec(child) for child in node.children]
        return Tree(str(cat), children)
    return rec(tree)


@DatasetReader.register("ccgbank_as_ptb")
class CCGBankAsConstituencyReader(PennTreeBankConstituencySpanDatasetReader):
    @overrides
    def _read(self, file_path):
        for _, _, ccg_tree in read_auto(file_path):
            tree = ccg_to_nltk_tree(ccg_tree)
            pos_tags = [x[1]
                        for x in tree.pos()] if self._use_pos_tags else None
            yield self.text_to_instance(
                tree.leaves(), pos_tags, tree)
