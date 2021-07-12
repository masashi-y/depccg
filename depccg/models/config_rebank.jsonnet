local targets = (import 'targets.en_rebank.jsonnet').targets;
local seen_rules = (import 'seen_rules.en_rebank.jsonnet').seen_rules;

{
  targets: targets,
  binary_rules: [
    [',', 'S[to]\\NP', 'NP\\NP', false],
    [',', 'S[adj]\\NP', 'NP\\NP', false],
    [',', 'S[pss]\\NP', 'NP\\NP', false],
    [',', 'S[ng]\\NP', 'NP\\NP', false],
  ],
  unary_rules: [
    ['N', 'NP'],
    ['S[pss]\\NP', 'N\\N'],
    ['S[ng]\\NP', 'N\\N'],
    ['S[adj]\\NP', 'N\\N'],
    ['S[dcl]/NP', 'N\\N'],
    ['S[to]\\NP', 'S/S'],
    ['S[pss]\\NP', 'S/S'],
    ['S[ng]\\NP', 'S/S'],
    ['NP', 'S[X]/(S[X]\\NP)'],
    ['NP', '(S[X]\\NP)\\((S[X]\\NP)/NP)'],
    ['PP', '(S[X]\\NP)\\((S[X]\\NP)/PP)'],
  ],
  seen_rules: seen_rules,
}
