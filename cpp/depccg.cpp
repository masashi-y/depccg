
#include <cmath>
#include <queue>
#include <utility>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "depccg.h"
#include "debug.h"
#include "matrix.h"
#include "chart.h"

namespace myccg {



std::vector<Cat> EnApplyUnaryRules(
        const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
        NodeType parse) {
    std::vector<Cat> results;
    for (Cat unary: unary_rules.at(parse->GetCategory())) {
        Cat result = parse->GetCategory();
        bool is_not_punct = parse->GetRuleType() != LP && parse->GetRuleType() != RP;
        if (is_not_punct || result->IsTypeRaised())
            results.push_back(result);
    }
    return results;
}


std::vector<Cat> JaApplyUnaryRules(
        const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
        NodeType parse) {
    return unary_rules.at(parse->GetCategory());
}


std::vector<RuleCache>& EnGetRules(
        std::unordered_map<CatPair, std::vector<RuleCache>>& rule_cache,
        const std::vector<Op>& binary_rules,
        const std::unordered_set<CatPair>& seen_rules,
        Cat left1,
        Cat right1) {
    Cat left = left1->StripFeat("[nb]");
    Cat right = right1->StripFeat("[nb]");
    auto key = std::make_pair(left, right);
    if (rule_cache.count(key) > 0) {
        return rule_cache[key];
    } else {
        std::vector<RuleCache> tmp;
        bool is_seen = seen_rules.size() > 0 &&
            seen_rules.count(
                std::make_pair(left1->StripFeat("[X]", "[nb]"),
                               right1->StripFeat("[X]", "[nb]"))) > 0;
        if (is_seen) {
            for (auto rule: binary_rules) {
                if (rule->CanApply(left, right)) {
                    tmp.emplace_back(rule->Apply(left, right),
                                rule->HeadIsLeft(left, right), rule);
                }
            }
        }
#pragma omp critical(ApplyBinaryRules)
        rule_cache.emplace(key, tmp);
        return rule_cache[key];
    }
}

std::vector<RuleCache>& JaGetRules(
        std::unordered_map<CatPair, std::vector<RuleCache>>& rule_cache,
        const std::vector<Op>& binary_rules,
        const std::unordered_set<CatPair>& seen_rules,
        Cat left,
        Cat right) {
    auto key = std::make_pair(left, right);
    if (rule_cache.count(key) > 0) {
        return rule_cache[key];
    } else {
        std::vector<RuleCache> tmp;
        bool is_seen = seen_rules.count(std::make_pair(left, right)) > 0;
        if (is_seen) {
            for (auto rule: binary_rules) {
                if (rule->CanApply(left, right)) {
                    tmp.emplace_back(rule->Apply(left, right),
                                rule->HeadIsLeft(left, right), rule);
                }
            }
        }
#pragma omp critical(ApplyBinaryRules)
        rule_cache.emplace(key, tmp);
        return rule_cache[key];
    }
}


struct CompareFloatCat {
    bool operator() (const std::pair<float, Cat>& left, const std::pair<float, Cat>& right) const {
        return left.first < right.first;
    }
};

std::vector<ScoredNode> Failed() {
    static ScoredNode failure_node = std::make_pair(
        std::make_shared<Leaf>("fail", CCategory::Parse("NP"), 0), 0);
    return std::vector<ScoredNode>({failure_node});
}


bool NormalComparator(const AgendaItem& left, const AgendaItem& right) {
    return left.prob < right.prob;
}


void ComputeOutsideProbs(
        std::vector<float>& probs, unsigned sent_size, Matrix<float>& out) {
    std::vector<float> from_left(sent_size + 1);
    std::vector<float> from_right(sent_size + 1);
    from_left[0] = 0.0;
    from_right[sent_size] = 0.0;

    for (unsigned i = 0; i < sent_size - 1; i++) {
        unsigned j = sent_size - i;
        from_left[i + 1] = from_left[i] + probs[i];
        from_right[j - 1] = from_right[j] + probs[j - 1];
    }

    for (unsigned i = 0; i < sent_size + 1; i++) {
        for (unsigned j = i; j < sent_size + 1; j++) {
            out(i, j) = from_left[i] + from_right[j];
        }
    }
}

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
        unsigned max_length) {
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    unsigned tag_size = tag_list.size();
    unsigned sent_size = tokens.size();
    if (sent_size >= max_length)
        return Failed();

    std::vector<float> best_tag_probs(sent_size, 0);
    std::vector<float> best_dep_probs(sent_size, 0);

    Matrix<float> tag_out_probs(sent_size + 1, sent_size + 1);
    Matrix<float> dep_out_probs(sent_size + 1, sent_size + 1);
    Matrix<float> tag_in_probs(tag_scores, sent_size, tag_size);
    Matrix<float> dep_in_probs(dep_scores, sent_size, sent_size + 1);

    std::priority_queue<AgendaItem, std::vector<AgendaItem>,
                bool (*)(const AgendaItem&, const AgendaItem&)> agenda(NormalComparator);
    unsigned agenda_id = 0;

    std::vector<std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat>> scored_cats(sent_size);

    float dep_leaf_out_prob = 0.0;
    for (unsigned i = 0; i < sent_size; i++) {
        bool do_pruning = ! category_dict.empty() &&
                            category_dict.count(tokens[i]) > 0;
        for (unsigned j = 0; j < tag_size; j++) {
            if ( ! do_pruning ||
                    (do_pruning && category_dict.at(tokens[i])[j])) {
                float score = tag_in_probs(i, j);
                scored_cats[i].emplace(score, tag_list[j]);
            }
        }
        best_tag_probs[i] = scored_cats[i].top().first;
        int idx = dep_in_probs.ArgMax(i);
        best_dep_probs[i] = dep_in_probs(i, idx);
        dep_leaf_out_prob += dep_in_probs(i, idx);
    }

    ComputeOutsideProbs(best_tag_probs, sent_size, tag_out_probs);
    ComputeOutsideProbs(best_dep_probs, sent_size, dep_out_probs);

    for (unsigned i = 0; i < sent_size; i++) {
        float threshold = use_beta ?
            scored_cats[i].top().first * beta : std::numeric_limits<float>::lowest();
        float out_prob = tag_out_probs(i, i + 1) + dep_leaf_out_prob;

        unsigned j = 0;
        while (j++ < pruning_size && 0 < scored_cats[i].size()) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (std::exp(prob_and_cat.first) > threshold) {
                float in_prob = prob_and_cat.first;
                agenda.emplace(false, agenda_id++, std::make_shared<const Leaf>(
                            tokens[i], prob_and_cat.second, i), in_prob, out_prob, i, 1);
            } else
                break;
        }
    }

    Chart chart(sent_size, nbest > 1);
    Chart goal(1, nbest > 1);
    ChartCell* goal_cell = goal(0, 0);

    while (nbest > goal.Size() && agenda.size() > 0) {

        const AgendaItem item = agenda.top();
        if (item.fin) {
            goal_cell->update(item.parse, item.in_prob);
            agenda.pop();
            continue;
        }
        agenda.pop();
        NodeType parse = item.parse;

        ChartCell* cell = chart(item.start_of_span, item.span_length - 1);

        if (cell->update(parse, item.in_prob)) {

            if ( parse->GetLength() == sent_size &&
                    possible_root_cats.count(parse->GetCategory()) ) {
                float dep_score = dep_in_probs(parse->GetHeadId(), 0);
                float in_prob = item.in_prob +  dep_score;
                agenda.emplace(true, agenda_id++, parse, in_prob, 0.0,
                                    item.start_of_span, item.span_length);
            }

            if ((sent_size == 1 || item.span_length != sent_size)
                    && unary_rules.count(parse->GetCategory()) > 0) {
                for (Cat unary: apply_unary_rules(unary_rules, parse)) {
                    NodeType subtree = std::make_shared<const Tree>(unary, parse);
                    agenda.emplace(false, agenda_id++, subtree, item.in_prob - 0.1, item.out_prob,
                                        item.start_of_span, item.span_length);
                }
            }
            for (auto&& other: chart.GetCellsStartingAt(item.start_of_span + item.span_length)) {
                for (auto&& pair: other->items) {
                    NodeType right = pair.second.first;
                    float prob = pair.second.second;
                    int span_length = parse->GetLength() + right->GetLength();

                    for (auto&& rule: apply_binary_rules(
                                cache, binary_rules, seen_rules,
                                parse->GetCategory(), right->GetCategory())) {
                        NodeType subtree = std::make_shared<const Tree>(
                                rule.result, rule.left_is_head, parse, right, rule.combinator);
                        NodeType head = rule.left_is_head ? parse : right;
                        NodeType dep  = rule.left_is_head ? right : parse;
                        float dep_score = dep_in_probs(dep->GetHeadId(), head->GetHeadId() + 1);
                        float in_prob = item.in_prob + prob + dep_score;
                        float out_prob = tag_out_probs(item.start_of_span,
                                        item.start_of_span + span_length)
                                       + dep_out_probs(item.start_of_span,
                                        item.start_of_span + span_length)
                                       - best_dep_probs[head->GetHeadId()];
                        agenda.emplace(false, agenda_id++, subtree, in_prob, out_prob,
                                            item.start_of_span, span_length);
                    }
                }
            }
            for (auto&& other: chart.GetCellsEndingAt(item.start_of_span)) {
                for (auto&& pair: other->items) {
                    NodeType left = pair.second.first;
                    float prob = pair.second.second;
                    int span_length = parse->GetLength() + left->GetLength();
                    int start_of_span = left->GetStartOfSpan();

                    for (auto&& rule: apply_binary_rules(
                                cache, binary_rules, seen_rules,
                                left->GetCategory(), parse->GetCategory())) {
                        NodeType subtree = std::make_shared<const Tree>(
                                rule.result, rule.left_is_head, left, parse, rule.combinator);
                        NodeType head  = rule.left_is_head ? left : parse;
                        NodeType dep   = rule.left_is_head ? parse : left;
                        float dep_score = dep_in_probs(dep->GetHeadId(), head->GetHeadId() + 1);
                        float in_prob = item.in_prob + prob + dep_score;
                        float out_prob = tag_out_probs(start_of_span,
                                        start_of_span + span_length)
                                       + dep_out_probs(start_of_span,
                                        start_of_span + span_length)
                                       - best_dep_probs[head->GetHeadId()];
                        agenda.emplace(false, agenda_id++, subtree, in_prob, out_prob,
                                            start_of_span, span_length);
                    }
                }
            }
        }
    }

    if (goal.IsEmpty())
        return Failed();

    auto res = goal_cell->GetNBestParses();
    return res;
}

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
        unsigned max_length) {
    std::vector<std::vector<ScoredNode>> res(sents.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned i = 0; i < sents.size(); i++) {
        res[i] = ParseSentence(i,
                    sents[i],
                    tag_scores[i],
                    dep_scores[i],
                    category_dict,
                    tag_list,
                    beta,
                    use_beta,
                    pruning_size,
                    nbest,
                    possible_root_cats,
                    unary_rules,
                    binary_rules,
                    cache,
                    seen_rules,
                    apply_binary_rules,
                    apply_unary_rules,
                    max_length);
    }
    return res;
}

} // namespace myccg
