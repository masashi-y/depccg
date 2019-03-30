(import 'supertagger.jsonnet') + {
  dataset_reader+: {
    type: 'tritrain_supertagging_dataset',
    lazy: true,
    token_indexers: super.token_indexers,
  },
  train_data_path: {
    ccgbank: train_data,
    tritrain: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/headfirst_filtered.json',
  },
  validation_data_path: test_data,
}
