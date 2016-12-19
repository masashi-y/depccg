
#include <cmath>
#include <queue>
#include <utility>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parser.h"
#include "configure.h"
#include "debug.h"
#include "parser_tools.h"

namespace myccg {
namespace parser {


bool JapaneseComparator(const AgendaItem& left, const AgendaItem& right) {
    if ( fabs(left.prob - right.prob) > 0.00001 )
        return left.prob < right.prob;
#ifdef JAPANESE
    // if ((IsVerb(left.parse) || IsAdjective(left.parse->GetCategory())) &&
    //         !(IsVerb(right.parse) || IsAdjective(right.parse->GetCategory())))
    //     return false;
    // if ((IsVerb(right.parse) || IsAdjective(right.parse->GetCategory())) &&
    //         !(IsVerb(left.parse) || IsAdjective(left.parse->GetCategory())))
    //     return true;
    // if (IsPeriod(right.parse->GetCategory()))
    //     return false;
    // if (IsPeriod(left.parse->GetCategory()))
    //     return true;
    // if (left.parse->LeftNumDescendants() != right.parse->LeftNumDescendants())
        // return left.parse->LeftNumDescendants() <= right.parse->LeftNumDescendants();
// #else
    if (left.parse->GetDependencyLength() != right.parse->GetDependencyLength())
        return left.parse->GetDependencyLength() < right.parse->GetDependencyLength();
#endif
    return left.id > right.id;
}


bool IsValid(combinator::RuleType rule_type, NodeType left, NodeType right) {
    if (right->IsUnary())
        return false;
    if (left->IsUnary())
        return rule_type == combinator::FA;
    if (IsPeriod(right->GetCategory()))
        return rule_type == combinator::BA;
    // std::cout << left->GetCategory()->ToStrWithoutFeat() << std::endl;
    if (left->GetCategory()->ToStrWithoutFeat() == "(NP/NP)") {
        std::string rcat = right->GetCategory()->ToStrWithoutFeat();
        return ((rcat == "NP" && rule_type == combinator::FA) || 
                (rcat == "(NP/NP)" && rule_type == combinator::FC));
    }
    if (IsAUX(left->GetCategory()) && IsAUX(right->GetCategory()))
        return false;
    if (IsVerb(left) && IsAUX(right->GetCategory()))
        return rule_type == combinator::BC || rule_type == combinator::BA;
    if (rule_type == combinator::FC) {
        // only allow S/S S/S or NP/NP NP/NP pairs
        Cat lcat = left->GetCategory();
        Cat rcat = right->GetCategory();
        return (IsModifier(lcat) && lcat->GetSlash().IsForward() &&
                IsModifier(rcat) && rcat->GetSlash().IsForward());
    }
    if (rule_type == combinator::FX) {
        return (IsVerb(right) &&
                IsAdverb(left->GetCategory()));
    }
    return true;
}

NodeType DepAStarParser::Parse(const std::string& sent, float beta) NO_IMPLEMENTATION

std::vector<NodeType>
DepAStarParser::Parse(const std::vector<std::string>& doc, float beta) {
    std::unique_ptr<float*[]> cat_scores, dep_scores;
    std::tie(cat_scores, dep_scores) = dep_tagger_->predict(doc);
    std::vector<NodeType> res(doc.size());
    #pragma omp parallel for schedule(PARALLEL_SCHEDULE)
    for (unsigned i = 0; i < doc.size(); i++) {
        res[i] = Parse(doc[i], cat_scores[i], dep_scores[i], beta);
    }
    return res;
}

#define NORMALIZED_PROB(x, y) std::log( std::exp((x)) / (y) )

NodeType DepAStarParser::Parse(
        const std::string& sent,
        float* tag_scores,
        float* dep_scores,
        float beta) {
    int pruning_size = 50;
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    int sent_size = (int)tokens.size();
    float best_in_probs[MAX_LENGTH];
    float out_probs[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];
    AgendaType agenda(JapaneseComparator);
    int agenda_id = 0;

    float tag_totals[MAX_LENGTH], dep_totals[MAX_LENGTH];

    std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat> scored_cats[MAX_LENGTH];

#ifdef DEBUGGING
    for (int i = 0; i < sent_size; i++) {
        std::cerr << tokens[i] << " --> ";
        std::cerr << dep_tagger_->TagAt( utils::ArgMax(tag_scores + (i * TagSize()),
                    tag_scores + (i * TagSize() + TagSize() - 1))) << std::endl;
    }
    std::cerr << std::endl;

    for (int i = 0; i < sent_size; i++) {
        std::cerr << i + 1 << " " << tokens[i] << " --> ";
        std::cerr << utils::ArgMax(dep_scores + (i * (sent_size + 1)),
                    dep_scores + (i * (sent_size + 1) + (sent_size + 1) - 1)) << std::endl;
    }
    std::cerr << std::endl;
#endif

    for (int i = 0; i < sent_size; i++) {
        tag_totals[i] = 0.0;
        dep_totals[i] = 0.0;
        for (int j = 0; j < TagSize(); j++) {
            float score = tag_scores[i * TagSize() + j];
            tag_totals[i] += std::exp(score);
            scored_cats[i].emplace(score, TagAt(j));
        }
        best_in_probs[i] = NORMALIZED_PROB( scored_cats[i].top().first, tag_totals[i] );
    }
    ComputeOutsideProbs(best_in_probs, sent_size, out_probs);

    for (int i = 0; i < sent_size; i++) {
        float threshold = scored_cats[i].top().first * beta;
        float out_prob = out_probs[i * sent_size + (i + 1)];
        float dep_score = std::log(0.000000000001); //dep_scores[i * sent_size + i + 1];

        for (int j = 0; j < pruning_size; j++) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (prob_and_cat.first > threshold) {
                float in_prob = dep_score + NORMALIZED_PROB( prob_and_cat.first, tag_totals[i] );
                agenda.emplace(agenda_id++, std::make_shared<const tree::Leaf>(
                            tokens[i], prob_and_cat.second, i), in_prob, out_prob, i, 1);
            } else
                break;
        }
    }

    ChartCell chart[MAX_LENGTH * MAX_LENGTH];

    while (chart[sent_size - 1].IsEmpty() && agenda.size() > 0) {

        const AgendaItem item = agenda.top();
        agenda.pop();
        NodeType parse = item.parse;

#ifdef DEBUGGING
        POPPED;
        std::cerr << tree::Derivation(parse);
        std::cerr << "score: " << item.prob << std::endl;
        DEBUG(item.start_of_span);
        DEBUG(item.span_length);
        BORDER;
#endif

        ChartCell& cell = chart[item.start_of_span * sent_size + (item.span_length - 1)];

        if (cell.update(parse, item.in_prob)) {
            if (item.span_length != sent_size) {
                for (Cat unary: unary_rules_[parse->GetCategory()]) {
                    NodeType subtree = std::make_shared<const tree::Tree>(unary, parse);
#ifdef DEBUGGING
        TREETYPE("UNARY");
        std::cerr << tree::Derivation(subtree);
        BORDER;
#endif
                    agenda.emplace(agenda_id++, subtree, item.in_prob, item.out_prob,
                                        item.start_of_span, item.span_length);
                }
            }
            for (int span_length = item.span_length + 1
                ; span_length < 1 + sent_size - item.start_of_span
                ; span_length++) {
                ChartCell& other = chart[(item.start_of_span + item.span_length) * 
                                sent_size + (span_length - item.span_length - 1)];
                for (auto&& pair: other.items) {
                    NodeType right = pair.second.first;
#ifdef DEBUGGING
        TREETYPE("RIGHT ARG");
        std::cerr << tree::Derivation(right);
        std::cerr << "start_of_span: " << item.start_of_span + item.span_length << std::endl;
        std::cerr << "span_length: " << span_length - item.span_length << std::endl;
        BORDER;
#endif
                    float prob = pair.second.second;
                    if (! IsSeen(parse->GetCategory(), right->GetCategory())) continue;
                    for (auto&& rule: GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (IsValid(rule.combinator->GetRuleType(), parse, right) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const tree::Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
#ifdef DEBUGGING
        ACCEPT;
        std::cerr << tree::Derivation(subtree);
        BORDER;
#endif
                            int head = rule.left_is_head ? parse->GetHeadId() : right->GetHeadId();
                            int dep  = rule.left_is_head ? right->GetHeadId() : parse->GetHeadId();
                            float dep_score = dep_scores[dep * (sent_size + 1) + head + 1];
                            std::cerr << dep << ", " << head << ", " << sent_size << std::endl;
                            std::cerr << tokens[dep] << " --> " << tokens[head] << ": " << dep_score << std::endl;
                            float in_prob = item.in_prob + prob + dep_score;
                            float out_prob = out_probs[item.start_of_span *
                                            sent_size + item.start_of_span + span_length];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                item.start_of_span, span_length);
                        }
                    }
                }
            }
            for (int start_of_span = 0; start_of_span < item.start_of_span; start_of_span++) {
                int span_length = item.start_of_span + item.span_length - start_of_span;
                ChartCell& other = chart[start_of_span * sent_size +
                                    (span_length - item.span_length - 1)];
                for (auto&& pair: other.items) {
                    NodeType left = pair.second.first;
                    float prob = pair.second.second;
#ifdef DEBUGGING
        TREETYPE("LEFT ARG");
        std::cerr << tree::Derivation(left);
        std::cerr << "start_of_span: " << start_of_span << std::endl;
        std::cerr << "span_length: " << span_length - item.span_length << std::endl;
        BORDER;
#endif
                    if (! IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (IsValid(rule.combinator->GetRuleType(), left, parse) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const tree::Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
#ifdef DEBUGGING
        ACCEPT;
        std::cerr << tree::Derivation(subtree);
        BORDER;
#endif
                            int head  = rule.left_is_head ? left->GetHeadId() : parse->GetHeadId();
                            int dep = rule.left_is_head ? parse->GetHeadId() : left->GetHeadId();
                            float dep_score = dep_scores[dep * (sent_size + 1) + head + 1];
                            std::cerr << dep << ", " << head << ", " << sent_size << std::endl;
                            std::cerr << tokens[dep] << " --> " << tokens[head] << ": " << dep_score << std::endl;
                            float in_prob = item.in_prob + prob + dep_score;
                            float out_prob = out_probs[start_of_span * sent_size +
                                            start_of_span + span_length];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                start_of_span, span_length);
                        }
                    }
                }
            }
        }
    }
    if (chart[sent_size - 1].IsEmpty())
        return failure_node;
    auto res = chart[sent_size - 1].GetBestParse();
    return res;
}

} // namespace parser
} // namespace myccg

