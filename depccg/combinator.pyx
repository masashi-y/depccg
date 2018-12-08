
# cdef vector[Op] headfirst_binary_rules = vector[Op]([
#     HeadFirst(ENForwardApplication()),
#     HeadFirst(ENBackwardApplication()),
#     HeadFirst(GeneralizedForwardComposition[0, FC](Slash.Fwd(), Slash.Fwd(), Slash.Fwd())),
#     HeadFirst(GeneralizedBackwardComposition[0, BC](Slash.Fwd(), Slash.Bwd(), Slash.Fwd())),
#     HeadFirst(GeneralizedForwardComposition[1, GFC](Slash.Fwd(), Slash.Fwd(), Slash.Fwd())),
#     HeadFirst(GeneralizedBackwardComposition[1, GBC](Slash.Fwd(), Slash.Fwd(), Slash.Fwd())),
#     HeadFirst(Conjunction()),
#     HeadFirst(Conjunction2()),
#     HeadFirst(RemovePunctuation(False)),
#     HeadFirst(RemovePunctuation(True)),
#     HeadFirst(CommaAndVerbPhraseToAdverb()),
#     HeadFirst(ParentheticalDirectSpeech()) ])
