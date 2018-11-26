
#include <cmath>
#include <queue>
#include <utility>
#include <limits>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parser.h"
#include "configure.h"
#include "debug.h"
#include "matrix.h"
#include "grammar.h"

namespace myccg {

bool Parser::keep_going = true;

template<typename Lang>
void AStarParser<Lang>::LoadSeenRules() {
    logger_(Info) << "loading seen rules .. ";
    use_seen_rules_ = true;
    try {
        seen_rules_ = SeenRules<Lang>(model_ + "/seen_rules.txt");
    } catch(std::runtime_error) {
        logger_(Info) << "failed loading. will not use seen rules";
        use_seen_rules_ = false;
    }
    logger_(Info) << "done";
}

template<typename Lang>
void AStarParser<Lang>::LoadCategoryDict() {
    logger_(Info) << "loading category dictionary .. ";
    use_category_dict_ = true;
    try {
        category_dict_ = utils::LoadCategoryDict(
                model_ + "/cat_dict.txt", tagger_->Targets());
    } catch(std::runtime_error) {
        logger_(Info) << Red("failed loading. will not use category dictionary");
        use_category_dict_ = false;
    }
    logger_(Info) << "done";
}

std::vector<ScoredNode> Parser::Failed(const std::string& sent, const std::string& message) {
    static ScoredNode failure_node = std::make_pair(
        std::make_shared<Leaf>("fail", Category::Parse("NP"), 0), 0);
    logger_(Info) << "failed to parse: " << sent
                  << " : " << message << std::endl;
    return std::vector<ScoredNode>({failure_node});
}

template<typename Lang>
bool AStarParser<Lang>::IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const {
    if (span_len == s_len)
        return (possible_root_cats_.count(cat) > 0);
    return true;
}

template<typename Lang>
bool AStarParser<Lang>::IsSeen(Cat left, Cat right) const {
    if (! use_seen_rules_)
        return true;
    return seen_rules_.IsSeen(left, right);
}

template<> template<>
Cat AStarParser<En>::SeenRules<En>::Preprocess(Cat cat) {
    return cat->StripFeat("[X]", "[nb]");
}

template<> template<>
Cat AStarParser<En>::CachedRules<En>::Preprocess(Cat cat) {
    return cat->StripFeat("[nb]");
}
template<typename Lang>
std::vector<RuleCache>& AStarParser<Lang>::GetRules(Cat left, Cat right) {
    return rule_cache_.GetRules(left, right);
}

template<typename Lang>
std::vector<std::vector<ScoredNode>> AStarParser<Lang>::Parse(const std::vector<std::string>& doc) {
    logger_.InitStatistics(doc.size());

    logger_.RecordTimeStartRunning();
    std::unique_ptr<float*[]> scores = tagger_->PredictTags(doc);
    logger_.RecordTimeEndOfTagging();
    std::vector<std::vector<ScoredNode>> res = Parse(doc, scores.get());
    logger_.RecordTimeEndOfParsing();
    logger_.Report();
    return res;
}

template<typename Lang>
std::vector<std::vector<ScoredNode>> AStarParser<Lang>::Parse(const std::vector<std::string>& doc,
                                               float** scores) {
    std::vector<std::vector<ScoredNode>> res(doc.size());
    #pragma omp parallel for schedule(PARALLEL_SCHEDULE)
    for (unsigned i = 0; i < doc.size(); i++) {
        if ( keep_going )
            res[i] = Parse(i, doc[i], scores[i]);
    }
    return res;
}

float GetLengthPenalty(NodeType left, NodeType right) {
    float penalty = abs(left->GetHeadId() - right->GetHeadId()) * 0.00001;
    if (right->GetLength() == 1 && right->GetWord()[0] == '\'')
        penalty *= 10;
    return penalty;
}

template<typename Lang>
std::vector<ScoredNode> AStarParser<Lang>::Parse(int id, const std::string& sent, float* scores) {
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    int sent_size = (int)tokens.size();
    if (sent_size >= MAX_LENGTH)
        return Failed(sent, "input sentence exceeding max length");
    float best_in_probs[MAX_LENGTH];
    float p_out[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];
    Matrix<float> out_probs(p_out, sent_size + 1, sent_size + 1);
    AgendaType agenda(comparator_);
    int agenda_id = 0;

    float totals[MAX_LENGTH];

    std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat> scored_cats[MAX_LENGTH];


    logger_.ShowTaggingOneBest(tagger_, scores, tokens);

    for (int i = 0; i < sent_size; i++) {
        totals[i] = 0.0;
        bool do_pruning = use_category_dict_ &&
                            category_dict_.count(tokens[i]) > 0;
        for (int j = 0; j < TagSize(); j++) {
            if ( ! do_pruning ||
                    (do_pruning && category_dict_[tokens[i]][j])) {
                float score = scores[i * TagSize() + j];
                totals[i] += std::exp(score);
                scored_cats[i].emplace(score, TagAt(j));
            }
        }
        best_in_probs[i] = std::log( std::exp(scored_cats[i].top().first) / totals[i] );
    }
    ComputeOutsideProbs(best_in_probs, sent_size, p_out);

    for (int i = 0; i < sent_size; i++) {
        float threshold = use_beta_ ? std::numeric_limits<float>::lowest() : scored_cats[i].top().first * beta_;
        float out_prob = out_probs(i, i + 1);

        int j = 0;
        while (j++ < pruning_size_ && 0 < scored_cats[i].size()) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (std::exp(prob_and_cat.first) > threshold) {
                float in_prob = std::log( std::exp(prob_and_cat.first) / totals[i] );
                agenda.emplace(agenda_id++, std::make_shared<const Leaf>(
                            tokens[i], prob_and_cat.second, i), in_prob, out_prob, i, 1);
            } else
                break;
        }
    }

    Chart chart(sent_size, nbest_ > 1);

    while (keep_going && chart.IsEmpty() && agenda.size() > 0) {

        const AgendaItem item = agenda.top();
        agenda.pop();
        NodeType parse = item.parse;
        logger_.RecordAgendaItem("POPPED", item);

        ChartCell* cell = chart(item.start_of_span, item.span_length - 1);

        if (cell->update(parse, item.in_prob)) {

            if (item.span_length != sent_size) {
                for (Cat unary: unary_rules_[parse->GetCategory()]) {
                    if (Lang::IsAcceptableUnary(unary, parse)) {
                        NodeType subtree = std::make_shared<const Tree>(unary, parse);
                        agenda.emplace(agenda_id++, subtree, item.in_prob - 0.1, item.out_prob,
                                            item.start_of_span, item.span_length);
                        logger_.RecordTree("UNARY", subtree);
                    }
                }
            }
            for (auto&& other: chart.GetCellsStartingAt(item.start_of_span + item.span_length)) {
                for (auto&& pair: other->items) {
                    NodeType right = pair.second.first;
                    float prob = pair.second.second;
                    int span_length = parse->GetLength() + right->GetLength();
                    logger_.RecordTree("RIGHT", right);

                    if (! IsSeen(parse->GetCategory(), right->GetCategory())) continue;
                    for (auto&& rule: GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (Lang::IsAcceptableBinary(rule.combinator->GetRuleType(), rule.result, parse, right) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
                            float in_prob = item.in_prob + prob - GetLengthPenalty(parse, right);
                            float out_prob = out_probs(item.start_of_span,
                                            item.start_of_span + span_length);
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                item.start_of_span, span_length);
                            logger_.RecordTree("ACCEPTED", subtree);
                        }
                    }
                }
            }
            for (auto&& other: chart.GetCellsEndingAt(item.start_of_span)) {
                for (auto&& pair: other->items) {
                    NodeType left = pair.second.first;
                    float prob = pair.second.second;
                    int span_length = parse->GetLength() + left->GetLength();
                    int start_of_span = left->GetLeftMostChild()->GetHeadId();
                    logger_.RecordTree("LEFT", left);

                    if (! IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (Lang::IsAcceptableBinary(rule.combinator->GetRuleType(), rule.result, left, parse) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
                            float in_prob = item.in_prob + prob - GetLengthPenalty(left, parse);
                            float out_prob = out_probs(start_of_span,
                                            start_of_span + span_length);
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                start_of_span, span_length);
                            logger_.RecordTree("ACCEPTED", subtree);
                        }
                    }
                }
            }
        }
    }

    logger_.CompleteOne(id, agenda_id);

    if (chart.IsEmpty())
        return Failed(sent, "no candidate parse found");

    auto res = chart(1,  -1)->GetNBestParses();
    logger_.CalculateNumOneBestTags(id, tagger_, scores, res[0].first);
    return res;
}

template class AStarParser<En>;
template class AStarParser<Ja>;

} // namespace myccg

