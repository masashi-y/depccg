
#ifndef INCLUDE_GRAMMAR_H_
#define INCLUDE_GRAMMAR_H_

#include "cat.h"
#include "combinator.h"

namespace myccg {
namespace grammar {

using namespace combinator;
using cat::Cat;

struct en {

static const std::unordered_set<Cat> possible_root_cats;
static const std::vector<Combinator*> binary_rules;

};

struct ja {
            
static const std::unordered_set<Cat> possible_root_cats;
static const std::vector<Combinator*> binary_rules;

};

} // namespace grammar
} // namespace myccg

#endif // include
