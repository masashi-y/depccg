
#ifndef INCLUDE_PARSER_H_
#define INCLUDE_PARSER_H_

#include "tree.h"
#include "chainer_tagger.h"

namespace myccg {
namespace parser {

class Parser
{
};

class AStarParser: public Parser
{
    AStarParser() {}

    tree::Node* parse(std::string& sent);
};
        
} // namespace parser
} // namespace myccg
#endif
