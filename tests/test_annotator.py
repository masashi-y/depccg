from depccg.annotator import annotate_using_janome, annotate_using_spacy
from depccg.types import Token


def test_spacy():
    sentences = [
        ["The Penn Treebank has recently implemented a new syntactic annotation scheme,",
         "designed to highlight aspects of predicate-argument structure."],
        ["This paper discusses the implementation of crucial aspects of this new annotation scheme."],
    ]

    tokens = [
        [
            Token(**{'word': 'The', 'pos': 'DT', 'entity': 'B-ORG',
                  'lemma': 'the', 'chunk': 'XX'}),
            Token(**{'word': 'Penn', 'pos': 'NNP', 'entity': 'I-ORG',
                  'lemma': 'penn', 'chunk': 'XX'}),
            Token(**{'word': 'Treebank', 'pos': 'NNP', 'entity': 'I-ORG',
                  'lemma': 'treebank', 'chunk': 'XX'}),
            Token(**{'word': 'has', 'pos': 'VBZ', 'entity': 'O',
                  'lemma': 'have', 'chunk': 'XX'}),
            Token(**{'word': 'recently', 'pos': 'RB', 'entity': 'O',
                  'lemma': 'recently', 'chunk': 'XX'}),
            Token(**{'word': 'implemented', 'pos': 'VBN', 'entity': 'O',
                  'lemma': 'implement', 'chunk': 'XX'}),
            Token(**{'word': 'a', 'pos': 'DT', 'entity': 'O',
                  'lemma': 'a', 'chunk': 'XX'}),
            Token(**{'word': 'new', 'pos': 'JJ', 'entity': 'O',
                  'lemma': 'new', 'chunk': 'XX'}),
            Token(**{'word': 'syntactic', 'pos': 'JJ', 'entity': 'O',
                  'lemma': 'syntactic', 'chunk': 'XX'}),
            Token(**{'word': 'annotation', 'pos': 'NN',
                     'entity': 'O', 'lemma': 'annotation', 'chunk': 'XX'}),
            Token(**{'word': 'scheme', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'scheme', 'chunk': 'XX'}),
            Token(**{'word': ',', 'pos': ',', 'entity': 'O',
                  'lemma': ',', 'chunk': 'XX'}),
            Token(**{'word': 'designed', 'pos': 'VBN', 'entity': 'O',
                  'lemma': 'design', 'chunk': 'XX'}),
            Token(**{'word': 'to', 'pos': 'TO', 'entity': 'O',
                  'lemma': 'to', 'chunk': 'XX'}),
            Token(**{'word': 'highlight', 'pos': 'VB', 'entity': 'O',
                  'lemma': 'highlight', 'chunk': 'XX'}),
            Token(**{'word': 'aspects', 'pos': 'NNS', 'entity': 'O',
                  'lemma': 'aspect', 'chunk': 'XX'}),
            Token(**{'word': 'of', 'pos': 'IN', 'entity': 'O',
                  'lemma': 'of', 'chunk': 'XX'}),
            Token(**{'word': 'predicate', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'predicate', 'chunk': 'XX'}),
            Token(**{'word': '-', 'pos': 'HYPH', 'entity': 'O',
                  'lemma': '-', 'chunk': 'XX'}),
            Token(**{'word': 'argument', 'pos': 'NN', 'entity': 'O', 'lemma': 'argument',
                     'chunk': 'XX'}),
            Token(**{'word': 'structure', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'structure', 'chunk': 'XX'}),
            Token(**{'word': '.', 'pos': '.', 'entity': 'O',
                  'lemma': '.', 'chunk': 'XX'})
        ],
        [
            Token(**{'word': 'This', 'pos': 'DT', 'entity': 'O',
                  'lemma': 'this', 'chunk': 'XX'}),
            Token(**{'word': 'paper', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'paper', 'chunk': 'XX'}),
            Token(**{'word': 'discusses', 'pos': 'VBZ', 'entity': 'O',
                  'lemma': 'discuss', 'chunk': 'XX'}),
            Token(**{'word': 'the', 'pos': 'DT', 'entity': 'O',
                  'lemma': 'the', 'chunk': 'XX'}),
            Token(**{'word': 'implementation', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'implementation', 'chunk': 'XX'}),
            Token(**{'word': 'of', 'pos': 'IN', 'entity': 'O',
                  'lemma': 'of', 'chunk': 'XX'}),
            Token(**{'word': 'crucial', 'pos': 'JJ', 'entity': 'O',
                  'lemma': 'crucial', 'chunk': 'XX'}),
            Token(**{'word': 'aspects', 'pos': 'NNS', 'entity': 'O',
                  'lemma': 'aspect', 'chunk': 'XX'}),
            Token(**{'word': 'of', 'pos': 'IN', 'entity': 'O',
                  'lemma': 'of', 'chunk': 'XX'}),
            Token(**{'word': 'this', 'pos': 'DT', 'entity': 'O',
                  'lemma': 'this', 'chunk': 'XX'}),
            Token(**{'word': 'new', 'pos': 'JJ', 'entity': 'O',
                  'lemma': 'new', 'chunk': 'XX'}),
            Token(**{'word': 'annotation', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'annotation', 'chunk': 'XX'}),
            Token(**{'word': 'scheme', 'pos': 'NN', 'entity': 'O',
                  'lemma': 'scheme', 'chunk': 'XX'}),
            Token(**{'word': '.', 'pos': '.', 'entity': 'O',
                  'lemma': '.', 'chunk': 'XX'})
        ]
    ]

    # raw_sentences = [
    #     ['The', 'Penn', 'Treebank', 'has', 'recently', 'implemented', 'a', 'new', 'syntactic', 'annotation', 'scheme',
    #         ',', 'designed', 'to', 'highlight', 'aspects', 'of', 'predicate', '-', 'argument', 'structure', '.'],
    #     ['This', 'paper', 'discusses', 'the', 'implementation', 'of', 'crucial',
    #         'aspects', 'of', 'this', 'new', 'annotation', 'scheme', '.']
    # ]

    assert tokens == annotate_using_spacy(sentences, tokenize=True)


test_spacy()


def test_janome():
    sentences = [
        ["メロスは激怒した。"],
        ["メロスには政治がわからぬ。"],
    ]

    tokens = [
        [
            Token(**{'word': 'メロス', 'surf': 'メロス', 'pos': '名詞', 'pos1': '一般', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': '*', 'base': 'メロス'}),
            Token(**{'word': 'は', 'surf': 'は', 'pos': '助詞', 'pos1': '係助詞', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': 'ハ', 'base': 'は'}),
            Token(**{'word': '激怒', 'surf': '激怒', 'pos': '名詞', 'pos1': 'サ変接続', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': 'ゲキド', 'base': '激怒'}),
            Token(**{'word': 'し', 'surf': 'し', 'pos': '動詞', 'pos1': '自立', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '連用形', 'inflectionType': 'サ変・スル', 'reading': 'シ', 'base': 'する'}),
            Token(**{'word': 'た', 'surf': 'た', 'pos': '助動詞', 'pos1': '*', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '基本形', 'inflectionType': '特殊・タ', 'reading': 'タ', 'base': 'た'}),
            Token(**{'word': '。', 'surf': '。', 'pos': '記号', 'pos1': '句点', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': '。', 'base': '。'})
        ],
        [
            Token(**{'word': 'メロス', 'surf': 'メロス', 'pos': '名詞', 'pos1': '一般', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': '*', 'base': 'メロス'}),
            Token(**{'word': 'に', 'surf': 'に', 'pos': '助詞', 'pos1': '格助詞', 'pos2': '一般', 'pos3': '*',
                     'inflectionForm': '*', 'inflectionType': '*', 'reading': 'ニ', 'base': 'に'}),
            Token(**{'word': 'は', 'surf': 'は', 'pos': '助詞', 'pos1': '係助詞', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': 'ハ', 'base': 'は'}),
            Token(**{'word': '政治', 'surf': '政治', 'pos': '名詞', 'pos1': '一般', 'pos2': '*', 'pos3': '*',
                     'inflectionForm': '*', 'inflectionType': '*', 'reading': 'セイジ', 'base': '政治'}),
            Token(**{'word': 'が', 'surf': 'が', 'pos': '助詞', 'pos1': '格助詞', 'pos2': '一般', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': 'ガ', 'base': 'が'}),
            Token(**{'word': 'わから', 'surf': 'わから', 'pos': '動詞', 'pos1': '自立', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '未然形', 'inflectionType': '五段・ラ行', 'reading': 'ワカラ', 'base': 'わかる'}),
            Token(**{'word': 'ぬ', 'surf': 'ぬ', 'pos': '助動詞', 'pos1': '*', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '基本形', 'inflectionType': '特殊・ヌ', 'reading': 'ヌ', 'base': 'ぬ'}),
            Token(**{'word': '。', 'surf': '。', 'pos': '記号', 'pos1': '句点', 'pos2': '*', 'pos3': '*',
                  'inflectionForm': '*', 'inflectionType': '*', 'reading': '。', 'base': '。'})
        ]
    ]

    # raw_sentences = [
    #     ['メロス', 'は', '激怒', 'し', 'た', '。'],
    #     ['メロス', 'に', 'は', '政治', 'が', 'わから', 'ぬ', '。']
    # ]

    assert tokens == annotate_using_janome(sentences, tokenize=True)
