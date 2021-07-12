local targets = (import 'targets.en.jsonnet').targets;
local unary_rules = (import 'unary_rules.en.jsonnet').unary_rules;
local tokens = (import 'tokens.en.jsonnet').tokens;
local prefixes = (import 'prefixes.en.jsonnet').prefixes;
local suffixes = (import 'suffixes.en.jsonnet').suffixes;
local seen_rules = (import 'seen_rules.en.jsonnet').seen_rules;
local cat_dict = (import 'cat_dict.en.jsonnet').cat_dict;

{
  word_dim: 100,
  n_suffixes: 40106,
  nlayers: 4,
  afix_dim: 30,
  dep_dim: 100,
  hidden_dim: 300,
  n_words: 400003,
  model: 'FastBiaffineLSTMParser',
  n_prefixes: 47046,
  targets: targets,
  unary_rules: unary_rules,
  words: tokens,
  prefixes: prefixes,
  suffixes: suffixes,
  seen_rules: seen_rules,
  cat_dict: cat_dict,
}
