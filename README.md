# depccg
Codebase for [A\* CCG Parsing with a Supertag and Dependency Factored Model](https://arxiv.org/abs/1704.06936)

__More stable and easier to use ocaml version is also available__: [depccg.ml](https://github.com/masashi-y/depccg.ml)

#### Requirements
* Python (Either 2 or 3)
* [Chainer](http://chainer.org/) (version 1.x)
* [Cython](http://cython.org/)
* A C++ compiler supporting [C++11 standard](https://en.wikipedia.org/wiki/C%2B%2B11) (in case of gcc, must be >= 4.9)
* OpenMP (optional, for faster batch parsing)

#### Build

```
pip install cython chainer==1.23
git clone https://github.com/masashi-y/depccg.git
cd depccg/src
python setup.py build_ext --inplace
```

Having successfully built the sources, you'll see `depccg.so` in the `src` directory.

`cmake` based build option is also available.  
This is convenient when you build the program many times.  
#### Pretrained models
Pretrained models are available:
* [English](http://cl.naist.jp/~masashi-y/resources/depccg/en_hf_tri.tar.gz) (189M)
* [Japanese](http://cl.naist.jp/~masashi-y/resources/depccg/ja_hf_ccgbank.tar.gz) (56M)

Please unpack these before passing to the parser.

#### Running parser
`src/run.py` implements a main parser program.  
Several options are available (N best parsing, input/output formats, etc.). Please do `python run.py -h` for the detail.

```
$ echo "this is a test sentence ." | python run.py ../models/tri_headfirst en --format deriv
ID=0
 this        is           a      test  sentence  .
  NP   (S[dcl]\NP)/NP  NP[nb]/N  N/N      N      .
                                ---------------->
                                      N ->
                      -------------------------->
                                NP ->
      ------------------------------------------>
                     S[dcl]\NP ->
------------------------------------------------<
                   S[dcl] ->
---------------------------------------------------<rp>
                     S[dcl] ->
```

In Python,
```python
from depccg import PyAStarParser
model = "/path/to/model/directory"
parser = PyAStarParser(model)
res = parser.parse("this is a test sentence .")

# parser.parse_doc performs A* search in threads (using OpenMP), which is highly efficient.
res = praser.parse_doc(sents) # sents: list of (python2: unicode, 3: str)
for nbests in res:
    for tree, log_prob in nbests:
        print tree.deriv
```

For Japanese CCG parsing, use `depccg.PyJaAStarParser`,
which has the exactly same interface.  
Note that the Japanese parser accepts pre-tokenized sentences as input.

#### Training model

For training, please use `src/py/lstm_parser_bi` for English and `src/py/ja_lstm_parser_bi` for Japanese.  

```
$ python -m py.lstm_parser_bi create
usage: CCG parser's LSTM supertag tagger create [-h]
                                                [--cat-freq-cut CAT_FREQ_CUT]
                                                [--word-freq-cut WORD_FREQ_CUT]
                                                [--afix-freq-cut AFIX_FREQ_CUT]
                                                [--subset {train,test,dev,all}]
                                                [--mode {train,test}]
                                                path out
```

```
$ python -m py.lstm_parser_bi train
usage: CCG parser's LSTM supertag tagger train [-h] [--gpu GPU]
                                               [--tritrain TRITRAIN]
                                               [--tri-weight TRI_WEIGHT]
                                               [--batchsize BATCHSIZE]
                                               [--epoch EPOCH]
                                               [--word-emb-size WORD_EMB_SIZE]
                                               [--afix-emb-size AFIX_EMB_SIZE]
                                               [--nlayers NLAYERS]
                                               [--hidden-dim HIDDEN_DIM]
                                               [--dep-dim DEP_DIM]
                                               [--dropout-ratio DROPOUT_RATIO]
                                               [--initmodel INITMODEL]
                                               [--pretrained PRETRAINED]
                                               model train val
```

We make tri-training dataset publicly available:
[English Tri-training Dataset](http://cl.naist.jp/~masashi-y/resources/depccg/headfirst_parsed.conll.stagged.gz) (309M)

#### Evaluation
You can evaluate the performance of a supertagger with `src/py/eval_tagger.py`:
```
$ python eval_tagger.py
usage: evaluate lstm tagger [-h] [--save SAVE] model defs_dir test_data
```

For the evaluation in CCG-based dependencies, please use
evaluation scripts in [EasyCCG](https://github.com/mikelewis0/easyccg) and
[C&C](http://www.cl.cam.ac.uk/~sc609/candc-1.00.html).

#### Citation

If you make use of this software, please cite the following:

    @inproceedings{yoshikawa:2017acl,
      author={Yoshikawa, Masashi and Noji, Hiroshi and Matsumoto, Yuji},
      title={A* CCG Parsing with a Supertag and Dependency Factored Model},
      booktitle={Proc. ACL},
      year=2017,
    }

#### Licence
MIT Licence

#### Contact
For questions and usage issues, please contact yoshikawa.masashi.yh8@is.naist.jp .

#### Acknowledgement
In creating the parser, I owe very much to:
- [EasyCCG](https://github.com/mikelewis0/easyccg): from which I learned everything
- [NLTK](http://www.nltk.org/): for nice pretty printing for parse derivation
