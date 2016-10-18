
Reimplementation of:
    "A* CCG Parsing with a Supertag-factored Model", Lewis and Steedman, EMNLP 2014

```
In [1]: from astar import AStarParser

In [2]: parser = AStarParser("model")

In [3]: sent = "this is a new sentence ."

In [4]: res = parser.parse(sent)

In [5]: res.show_derivation()
  NP   (S[dcl]\NP)/NP  NP/NP  NP[nb]/N     N      .
 this        is          a      new     sentence  .
------<un>
S[X]/(S[X]\NP)
                             -------------------->
                                    NP[nb]
                             -----------------------<rp>
                                     NP[nb]
                      ------------------------------>
                                  NP[nb]
      ---------------------------------------------->
                        S[dcl]\NP
---------------------------------------------------->
                       S[dcl]

```

I owe very much to:
- easyCCG: https://github.com/mikelewis0/easyccg
- NLTK: for nice pretty printing for derivation tree
- spacy: https://github.com/explosion/spaCy
- http://andreinc.net/2011/06/01/implementing-a-generic-priority-queue-in-c/

