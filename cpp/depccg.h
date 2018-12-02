
#ifndef INCLUDE_DEPCCG_H_
#define INCLUDE_DEPCCG_H_

#include <unordered_map>
#include <vector>
#include "cat.h"
#include "combinator.h"
#include "tree.h"

namespace myccg {

struct RuleCache
{
    RuleCache() {}
    RuleCache(Cat result, bool left_is_head, Op combinator)
    : result(result), left_is_head(left_is_head), combinator(combinator) {}

    Cat result;
    bool left_is_head;
    Op combinator;
};

struct AgendaItem
{
    AgendaItem(bool fin, int id, NodeType parse_, float in_prob_, float out_prob_,
            unsigned start_of_span_, unsigned span_length_)
    : fin(fin), id(id), parse(parse_), in_prob(in_prob_), out_prob(out_prob_),
    prob(in_prob_ + out_prob_), start_of_span(start_of_span_), span_length(span_length_) {}
    ~AgendaItem() {}

    bool fin;
    int id;
    NodeType parse;
    float in_prob;
    float out_prob;
    float prob;
    unsigned start_of_span;
    unsigned span_length;

};


typedef std::vector<RuleCache>& (*ApplyBinaryRules)(
        std::unordered_map<CatPair, std::vector<RuleCache>>&,
        const std::vector<Op>&,
        const std::unordered_set<CatPair>&,
        Cat, Cat);

typedef std::vector<Cat> (*ApplyUnaryRules)(
        const std::unordered_map<Cat, std::vector<Cat>>&,
        NodeType);


std::vector<Cat> EnApplyUnaryRules(
        const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
        NodeType parse);

std::vector<Cat> JaApplyUnaryRules(
        const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
        NodeType parse);

std::vector<RuleCache>& EnGetRules(
        std::unordered_map<CatPair, std::vector<RuleCache>>& rule_cache,
        const std::vector<Op>& binary_rules,
        const std::unordered_set<CatPair>& seen_rules,
        Cat left1,
        Cat right1);

std::vector<RuleCache>& JaGetRules(
        std::unordered_map<CatPair, std::vector<RuleCache>>& rule_cache,
        const std::vector<Op>& binary_rules,
        const std::unordered_set<CatPair>& seen_rules,
        Cat left1,
        Cat right1);

std::vector<ScoredNode> ParseSentence(
        unsigned id,
        const std::string& sent,
        float* tag_scores,
        float* dep_scores,
        const std::unordered_map<std::string, std::vector<bool>>& category_dict,
        const std::vector<Cat>& tag_list,
        float beta,
        bool use_beta,
        unsigned pruning_size,
        unsigned nbest,
        const std::unordered_set<Cat>& possible_root_cats,
        const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
        const std::vector<Op>& binary_rules,
        std::unordered_map<CatPair, std::vector<RuleCache>>& cache,
        const std::unordered_set<CatPair>& seen_rules,
        ApplyBinaryRules apply_binary_rules,
        ApplyUnaryRules apply_unary_rules,
        unsigned max_length);

std::vector<std::vector<ScoredNode>> ParseSentences(
        std::vector<std::string>& sents,
        float** tag_scores,
        float** dep_scores,
        const std::unordered_map<std::string, std::vector<bool>>& category_dict,
        const std::vector<Cat>& tag_list,
        float beta,
        bool use_beta,
        unsigned pruning_size,
        unsigned nbest,
        const std::unordered_set<Cat>& possible_root_cats,
        const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
        const std::vector<Op>& binary_rules,
        std::unordered_map<CatPair, std::vector<RuleCache>>& cache,
        const std::unordered_set<CatPair>& seen_rules,
        ApplyBinaryRules apply_binary_rules,
        ApplyUnaryRules apply_unary_rules,
        unsigned max_length);

} // namespace myccg
#endif
