
#ifndef INCLUDE_GRAMMAR_H_
#define INCLUDE_GRAMMAR_H_

#include "cat.h"
#include "combinator.h"
#include "tree.h"

namespace myccg {

struct En {

static bool IsAcceptable(RuleType rule_type, NodeType left, NodeType right);

static bool IsModifier(Cat cat);

static bool IsVerb(NodeType tree);

static bool IsAdjective(Cat cat);

static bool IsAdverb(Cat cat);

static bool IsAUX(Cat cat);

static bool IsPeriod(Cat cat);

static const std::unordered_set<Cat> possible_root_cats;
static const std::vector<Combinator*> binary_rules;

};

struct Ja {
            
static bool IsAcceptable(RuleType rule_type, NodeType left, NodeType right);

static bool IsModifier(Cat cat);

static bool IsVerb(NodeType tree);

static bool IsAdjective(Cat cat);

static bool IsAdverb(Cat cat);

static bool IsAUX(Cat cat);

static bool IsPeriod(Cat cat);

static const std::unordered_set<Cat> possible_root_cats;
static const std::vector<Combinator*> binary_rules;

};

} // namespace myccg

#endif // include
