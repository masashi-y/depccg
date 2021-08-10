{
  token_indexers: {
    tokens: {
      type: 'single_id',
      lowercase_tokens: true,
    },
  },
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'embedding',
        pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz',
        embedding_dim: 100,
        sparse: true,
      },
    },
  },
  encoder_input_dim: 100,
}