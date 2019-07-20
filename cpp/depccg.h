
#ifndef INCLUDE_DEPCCG_H_
#define INCLUDE_DEPCCG_H_

#include <unordered_map>
#include <vector>
#include <functional>
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

struct PartialConstraint
{
    PartialConstraint(Cat cat, unsigned start_of_span, unsigned span_length)
        : cat(cat), start_of_span(start_of_span), span_length(span_length) {}

    PartialConstraint(unsigned start_of_span, unsigned span_length)
        : cat(nullptr), start_of_span(start_of_span), span_length(span_length) {}

    bool SpanOverlap(unsigned start_of_span0, unsigned span_length0) const {
    unsigned end_of_span = start_of_span + span_length - 1;
    unsigned end_of_span0 = start_of_span0 + span_length0 - 1;
    return (start_of_span0 <= end_of_span &&
            end_of_span < end_of_span0 &&
            start_of_span < start_of_span0 &&
            start_of_span0 <= end_of_span) || 
           (start_of_span0 < start_of_span &&
            start_of_span <= end_of_span0 &&
            start_of_span <= end_of_span0 &&
            end_of_span0 < end_of_span);
    }

    bool Violates(Cat cat0, unsigned start_of_span0, unsigned span_length0) const {
        if (cat && start_of_span == start_of_span0 && span_length == span_length0) {
            return ! cat->Matches(cat0);
        }
        return false;
    }

    bool Violates(Cat cat0, unsigned start_of_span0, unsigned span_length0,
            const std::unordered_map<Cat, std::unordered_set<Cat>>& unary_rules) const {
        if (cat && start_of_span == start_of_span0 && span_length == span_length0) {
            if (cat->Matches(cat0))
                return false;
            if (unary_rules.count(cat0) > 0) {
                for (auto&& unary: unary_rules.at(cat0)) {
                    if (unary->Matches(cat))
                        return false;
                }
            }
            return true;
        }
        return SpanOverlap(start_of_span0, span_length0);
    }

    Cat cat;
    unsigned start_of_span;
    unsigned span_length;
};

class PartialConstraints
{
public:
    PartialConstraints() {}
    PartialConstraints(const std::unordered_map<Cat, std::unordered_set<Cat>>& unary_rules)
        : unary_rules(unary_rules) {}

    PartialConstraints(const PartialConstraints&) = default;

    void Add(Cat cat0, unsigned start_of_span0, unsigned span_length0) {
        constraints.emplace_back(cat0, start_of_span0, span_length0);
    }

    void Add(unsigned start_of_span0, unsigned span_length0) {
        constraints.emplace_back(start_of_span0, span_length0);
    }

    // std::vector<NodeType> AcceptableTrees(NodeType node) const {
    //     std::vector<NodeType> results;
    //     bool add_node = false;
    //     unsigned start_of_span0 = node->GetStartOfSpan();
    //     unsigned span_length0 = node->GetLength();
    //
    //     for (auto&& constraint: this->constraints) {
    //         if (constraint.cat && 
    //             constraint.start_of_span == start_of_span0 && 
    //             constraint.span_length == span_length0) {
    //
    //             if (constraint.cat->Matches(node->GetCategory())) {
    //                 results.push_back(node);
    //                 add_node = true;
    //             }
    //             if (unary_rules.count(node->GetCategory()) > 0) {
    //                for (auto&& unary: unary_rules.at(node->GetCategory())) {
    //                    if (unary->Matches(constraint.cat)) {
    //                        NodeType unary_tree = std::make_shared<const CTree>(unary, node);
    //                        results.push_back(unary_tree);
    //                    }
    //                }
    //             }
    //         }
    //         if (! add_node && ! constraint.SpanOverlap(start_of_span0, span_length0))
    //             results.push_back(node);
    //     }
    //     return results;
    // }

    bool Violates(Cat cat0, unsigned start_of_span0, unsigned span_length0) const {
        for (auto&& constraint: constraints) {
            if (constraint.Violates(cat0, start_of_span0, span_length0, unary_rules)) {
                return true;
            }
        }
        return false;
    }

    bool ViolatesNoRec(NodeType node) const {
        for (auto&& constraint: constraints) {
            if (constraint.Violates(
                    node->GetCategory(), node->GetStartOfSpan(), node->GetLength())) {
                return true;
            }
        }
        return false;
    }

    bool Violates(NodeType node) const {
        bool result = false;
        if (! node->IsLeaf() && ! node->IsUnary() ) {
            result |= ViolatesNoRec(node->GetLeftChild());
            result |= ViolatesNoRec(node->GetRightChild());
        }
        if (result)
            return true;
        for (auto&& constraint: constraints) {
            if (constraint.Violates(
                    node->GetCategory(), node->GetStartOfSpan(), node->GetLength(), unary_rules)) {
                return true;
            }
        }
        return false;
    }

private:
    std::unordered_map<Cat, std::unordered_set<Cat>> unary_rules;
    std::vector<PartialConstraint> constraints;
};

// typedef std::vector<Cat> (*ApplyUnaryRules)(
//         const std::unordered_map<Cat, std::vector<Cat>>&,
//         NodeType);


typedef std::function<std::unordered_set<Cat>(
        const std::unordered_map<Cat, std::unordered_set<Cat>>&,
        NodeType)> ApplyUnaryRules;

typedef std::function<std::vector<NodeType>(
        const std::unordered_set<CatPair>&, NodeType, NodeType, unsigned, unsigned)>
        ApplyBinaryRules;

std::unordered_set<Cat> DefaultApplyUnaryRules(
        const std::unordered_map<Cat, std::unordered_set<Cat>>& unary_rules,
        NodeType parse);

ApplyBinaryRules MakeDefaultApplyBinaryRules(const std::vector<Op>& binary_rules);

std::unordered_set<Cat> EnApplyUnaryRules(
        const std::unordered_map<Cat, std::unordered_set<Cat>>& unary_rules,
        NodeType parse);

ApplyBinaryRules MakeEnApplyBinaryRules(const std::vector<Op>& binary_rules);

std::vector<ScoredNode> ParseSentence(
        unsigned id,
        const std::string& sent,
        float* tag_scores,
        float* dep_scores,
        const std::unordered_map<std::string, std::unordered_set<Cat>>& category_dict,
        const std::vector<Cat>& tag_list,
        float unary_penalty,
        float beta,
        bool use_beta,
        unsigned pruning_size,
        unsigned nbest,
        const std::unordered_set<Cat>& possible_root_cats,
        const std::unordered_map<Cat, std::unordered_set<Cat>>& unary_rules,
        const std::unordered_set<CatPair>& seen_rules,
        ApplyBinaryRules apply_binary_rules,
        ApplyUnaryRules apply_unary_rules,
        PartialConstraints& constraints,
        unsigned max_length,
        unsigned max_steps);

std::vector<std::vector<ScoredNode>> ParseSentences(
        std::vector<std::string>& sents,
        float** tag_scores,
        float** dep_scores,
        const std::unordered_map<std::string, std::unordered_set<Cat>>& category_dict,
        const std::vector<Cat>& tag_list,
        float unary_penalty,
        float beta,
        bool use_beta,
        unsigned pruning_size,
        unsigned nbest,
        const std::unordered_set<Cat>& possible_root_cats,
        const std::unordered_map<Cat, std::unordered_set<Cat>>& unary_rules,
        const std::unordered_set<CatPair>& seen_rules,
        const std::vector<ApplyBinaryRules>& apply_binary_rules,
        ApplyUnaryRules apply_unary_rules,
        std::vector<PartialConstraints>& constraints,
        unsigned max_length,
        unsigned max_steps,
        bool silent);

} // namespace myccg
#endif
