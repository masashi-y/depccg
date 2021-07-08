local targets = (import 'targets.ja.jsonnet').targets;
local unary_rules = (import 'unary_rules.ja.jsonnet').unary_rules;
local tokens = (import 'tokens.ja.jsonnet').tokens;
local chars = (import 'chars.ja.jsonnet').chars;
local seen_rules = (import 'seen_rules.ja.jsonnet').seen_rules;

{
  n_chars: 2378,
  nlayers: 4,
  char_dim: 100,
  dep_dim: 100,
  word_dim: 200,
  hidden_dim: 300,
  n_words: 26246,
  model: 'BiaffineJaLSTMParser',
  targets: targets,
  unary_rules: unary_rules,
  words: tokens,
  chars: chars,
  seen_rules: seen_rules,
  cat_dict: {},
}
