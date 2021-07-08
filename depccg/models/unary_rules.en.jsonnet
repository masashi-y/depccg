{
  unary_rules: [
    ['N', 'NP'],
    ['S[pss]\\NP', 'NP\\NP'],
    ['S[ng]\\NP', 'NP\\NP'],
    ['S[adj]\\NP', 'NP\\NP'],
    ['S[to]\\NP', 'NP\\NP'],
    ['S[dcl]/NP', 'NP\\NP'],
    ['S[to]\\NP', '(S/S)'],
    ['S[pss]\\NP', '(S/S)'],
    ['S[ng]\\NP', '(S/S)'],
    ['NP', '(S[X]/(S[X]\\NP))'],
    ['NP', '((S[X]\\NP)\\((S[X]\\NP)/NP))'],
    ['PP', '((S[X]\\NP)\\((S[X]\\NP)/PP))'],
    ['NP', '(((S[X]\\NP)/NP)\\(((S[X]\\NP)/NP)/NP))'],
    ['NP', '(((S[X]\\NP)/PP)\\(((S[X]\\NP)/PP)/NP))'],
    ['(S[ng]\\NP)', 'NP'],
  ],
}
