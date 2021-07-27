#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <list>
#include <limits>
#include <cmath>
#include <utility>
#include <stdexcept>

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ h2;
    }
};

struct combinator_result
{
    unsigned cat_id;
    unsigned rule_id;
    bool head_is_left;
    std::string op_string;
    std::string op_symbol;
};

using category_id = unsigned;

using scored_category = std::pair<float, category_id>;

using cache_type = std::unordered_map<std::pair<unsigned, unsigned>, std::vector<combinator_result>, pair_hash>;

typedef int (*scaffold_type)(void *callback_func, unsigned x, unsigned y, std::vector<combinator_result> *results);

namespace utils
{
    template <typename T>
    int argmax(T *from, T *to)
    {
        T max_val = std::numeric_limits<T>::lowest();
        int max_idx = -1, i = 0;
        while (from != to)
        {
            if (max_val <= *from)
            {
                max_idx = i;
                max_val = *from;
            }
            i++;
            from++;
        }
        return max_idx;
    }
}

namespace parsing
{

    struct cell_item
    {

        bool fin;
        category_id cat;
        cell_item *left;
        cell_item *right;
        float in_score;
        float out_score;
        unsigned start_of_span;
        unsigned span_length;
        unsigned head_id;
        unsigned rule_id;

        float score() const { return in_score + out_score; }

        unsigned end_of_span() const { return start_of_span + span_length; }
    };

    bool operator<(const cell_item &left, const cell_item &right)
    {
        return left.score() < right.score();
    };

    class chart
    {
    public:
        class cell
        {
        public:
            using cell_items = std::list<cell_item>;

            cell() : seen(false){};
            cell(const cell &) = delete;

            bool contains(category_id cat) const { return category_ids.count(cat) > 0; }

            cell_item &emplace(const cell_item &item)
            {
                category_ids.emplace(item.cat);
                items.push_front(item);
                return items.front();
            }

            unsigned size() const { return items.size(); }
            cell_items::iterator begin() { return items.begin(); }
            cell_items::iterator end() { return items.end(); }

            void sort()
            {
                auto compare = [](parsing::cell_item &s1, parsing::cell_item &s2)
                {
                    return s1.score() > s2.score();
                };
                this->items.sort(compare);
            }

            friend chart;

        private:
            bool seen;
            std::unordered_set<category_id> category_ids;
            cell_items items;
        };

        chart(unsigned length, bool nbest)
            : length_(length),
              nbest_(nbest),
              chart_(new cell[length * length]),
              ending_cells_(new std::vector<cell *>[length + 1]),
              starting_cells_(new std::vector<cell *>[length + 1]) {}

        ~chart()
        {
            delete[] chart_;
            delete[] ending_cells_;
            delete[] starting_cells_;
        }

        cell &operator()(unsigned row, unsigned column)
        {
            cell &cell_ = chart_[row * length_ + column];

            if (!cell_.seen)
            {
                ending_cells_[row + column + 1].push_back(&cell_);
                starting_cells_[row].push_back(&cell_);
                cell_.seen = true;
            }
            return cell_;
        }

        cell_item *update(unsigned row, unsigned column, const cell_item &item)
        {
            cell &cell_ = (*this)(row, column);

            if (!nbest_ && cell_.contains(item.cat))
                return nullptr;

            return &cell_.emplace(item);
        }

        unsigned size() const
        {
            cell &final_ = chart_[length_ - 1];
            return final_.size();
        }

        bool empty() const { return size() == 0; }

        std::vector<cell *> &cells_starting_at(unsigned index)
        {
            return starting_cells_[index];
        }

        std::vector<cell *> &cells_ending_at(unsigned index)
        {
            return ending_cells_[index];
        }

    private:
        unsigned length_;
        bool nbest_;
        cell *chart_;
        std::vector<cell *> *ending_cells_, *starting_cells_;
    };

    class matrix
    {
    public:
        matrix(float *data, unsigned row, unsigned column)
            : data_(data), row_(row), column_(column), own_(false) {}

        matrix(unsigned row, unsigned column)
            : data_(new float[row * column]), row_(row), column_(column), own_(true) {}

        ~matrix()
        {
            if (own_)
                delete[] data_;
        }

        float &operator()(unsigned row, unsigned column) const
        {
            return data_[row * column_ + column];
        }

        unsigned argmax(unsigned row) const
        {
            return utils::argmax(data_ + (row * column_),
                                 data_ + (row * column_ + column_));
        }

        unsigned size() const { return column_ * row_; }

    private:
        float *data_;
        unsigned row_;
        unsigned column_;
        bool own_;
    };

    void compute_outside_probabilities(
        std::vector<float> &probs, unsigned length, matrix &out)
    {
        std::vector<float> from_left(length + 1);
        std::vector<float> from_right(length + 1);
        from_left[0] = 0.0;
        from_right[length] = 0.0;

        for (unsigned i = 0; i < length - 1; i++)
        {
            unsigned j = length - i;
            from_left[i + 1] = from_left[i] + probs[i];
            from_right[j - 1] = from_right[j] + probs[j - 1];
        }

        for (unsigned i = 0; i < length + 1; i++)
        {
            for (unsigned j = i; j < length + 1; j++)
            {
                out(i, j) = from_left[i] + from_right[j];
            }
        }
    }

} // namespace parsing

typedef unsigned (*finalizer_type)(parsing::cell_item *, unsigned *, cache_type *cache, void *);

struct config
{
    unsigned num_tags;
    float unary_penalty;
    float beta;
    bool use_beta;
    unsigned pruning_size;
    unsigned nbest;
    unsigned max_step;
};

unsigned parse_sentence(
    float *tag_scores,
    float *dep_scores,
    unsigned length,
    const std::unordered_set<unsigned> &possible_root_cats,
    void *binary_callback,
    void *unary_callback,
    finalizer_type finalizer_callback,
    scaffold_type scaffold,
    void *finalizer_args,
    cache_type *cache,
    config *config)
{
    auto apply_binary_rules = [&](unsigned x, unsigned y)
    {
        std::pair<unsigned, unsigned> key(x, y);
        if (cache->count(key) == 0)
        {
            std::vector<combinator_result> results;
            if (scaffold(binary_callback, x, y, &results) == -1)
                throw std::runtime_error(
                    "some error has occurred in the callback Python function.");
            cache->emplace(key, results);
        }
        return &cache->at(key);
    };

    auto apply_unary_rules = [&](unsigned x)
    {
        std::pair<unsigned, unsigned> key(x, UINT_MAX);
        if (cache->count(key) == 0)
        {
            std::vector<combinator_result> results;
            if (scaffold(unary_callback, x, UINT_MAX, &results) == -1)
                throw std::runtime_error(
                    "some error has occurred in the callback Python function.");
            cache->emplace(key, results);
        }
        return &cache->at(key);
    };

    std::vector<float> best_tag_scores(length, 0);
    std::vector<float> best_dep_scores(length, 0);

    parsing::matrix tag_out_scores(length + 1, length + 1);
    parsing::matrix dep_out_scores(length + 1, length + 1);
    parsing::matrix tag_in_scores(tag_scores, length, config->num_tags);
    parsing::matrix dep_in_scores(dep_scores, length, length + 1);

    std::priority_queue<parsing::cell_item> agenda;

    std::vector<std::priority_queue<scored_category>> scored_cats(length);

    float dep_leaf_out_score = 0.0;
    for (unsigned token_id = 0; token_id < length; token_id++)
    {
        for (unsigned category_id = 0; category_id < config->num_tags; category_id++)
            scored_cats[token_id].emplace(tag_in_scores(token_id, category_id), category_id);
        unsigned max_id = dep_in_scores.argmax(token_id);
        best_tag_scores[token_id] = scored_cats[token_id].top().first;
        best_dep_scores[token_id] = dep_in_scores(token_id, max_id);
        dep_leaf_out_score += dep_in_scores(token_id, max_id);
    }

    compute_outside_probabilities(best_tag_scores, length, tag_out_scores);
    compute_outside_probabilities(best_dep_scores, length, dep_out_scores);

    for (unsigned token_id = 0; token_id < length; token_id++)
    {
        float threshold = config->use_beta ? scored_cats[token_id].top().first * config->beta : std::numeric_limits<float>::lowest();
        float out_score = tag_out_scores(token_id, token_id + 1) + dep_leaf_out_score;

        for (unsigned i = 0; i < config->pruning_size && scored_cats[token_id].size(); i++)
        {
            auto score_and_cat = scored_cats[token_id].top();
            scored_cats[token_id].pop();
            if (std::exp(score_and_cat.first) > threshold)
            {
                agenda.push(
                    {false,
                     score_and_cat.second,
                     nullptr,
                     nullptr,
                     score_and_cat.first,
                     out_score,
                     token_id,
                     1,
                     token_id});
            }
            else
                break;
        }
    }

    parsing::chart chart(length, config->nbest > 1);
    parsing::chart goal(1, config->nbest > 1);

    for (unsigned s = 0; s < config->max_step && goal.size() < config->nbest && agenda.size(); s++)
    {
        parsing::cell_item top_item = agenda.top();
        agenda.pop();
        if (top_item.fin)
        {
            goal.update(0, 0, top_item);
            continue;
        }

        parsing::cell_item *item;
        if ((item = chart.update(top_item.start_of_span, top_item.span_length - 1, top_item)) != nullptr)
        {

            if (item->span_length == length && possible_root_cats.count(item->cat))
            {
                agenda.push(
                    {true,
                     item->cat,
                     item,
                     nullptr,
                     item->in_score + dep_in_scores(item->head_id, 0),
                     0.0,
                     item->start_of_span,
                     item->span_length,
                     item->head_id,
                     item->rule_id});
            }

            if (length == 1 || item->span_length != length)
            {
                for (auto &unary : *apply_unary_rules(item->cat))
                {
                    agenda.push(
                        {false,
                         unary.cat_id,
                         item,
                         nullptr,
                         item->in_score - config->unary_penalty,
                         item->out_score,
                         item->start_of_span,
                         item->span_length,
                         item->head_id,
                         0});
                }
            }

            for (auto &cell : chart.cells_starting_at(item->end_of_span()))
            {
                for (auto &other : *cell)
                {
                    unsigned span_length = item->span_length + other.span_length;
                    unsigned start_of_span = item->start_of_span;
                    unsigned end_of_span = start_of_span + span_length;

                    for (auto &rule_result : *apply_binary_rules(item->cat, other.cat))
                    {
                        auto head = rule_result.head_is_left ? item : &other;
                        auto child = rule_result.head_is_left ? &other : item;
                        float dep_score = dep_in_scores(child->head_id, head->head_id + 1);
                        float in_score = item->in_score + other.in_score + dep_score;
                        float out_score = tag_out_scores(start_of_span, end_of_span) +
                                          dep_out_scores(start_of_span, end_of_span) -
                                          best_dep_scores[head->head_id];
                        agenda.push(
                            {false,
                             rule_result.cat_id,
                             item,
                             &other,
                             in_score,
                             out_score,
                             start_of_span,
                             span_length,
                             head->head_id,
                             rule_result.rule_id});
                    }
                }
            }
            for (auto &cell : chart.cells_ending_at(item->start_of_span))
            {
                for (auto &other : *cell)
                {
                    unsigned span_length = item->span_length + other.span_length;
                    unsigned start_of_span = other.start_of_span;
                    unsigned end_of_span = start_of_span + span_length;

                    for (auto &rule_result : *apply_binary_rules(other.cat, item->cat))
                    {
                        auto head = rule_result.head_is_left ? &other : item;
                        auto child = rule_result.head_is_left ? item : &other;
                        float dep_score = dep_in_scores(child->head_id, head->head_id + 1);
                        float in_score = item->in_score + other.in_score + dep_score;
                        float out_score = tag_out_scores(start_of_span, end_of_span) +
                                          dep_out_scores(start_of_span, end_of_span) -
                                          best_dep_scores[head->head_id];
                        agenda.push(
                            {false,
                             rule_result.cat_id,
                             &other,
                             item,
                             in_score,
                             out_score,
                             start_of_span,
                             span_length,
                             head->head_id,
                             rule_result.rule_id});
                    }
                }
            }
        }
    }

    if (goal.size() == 0)
        return 1;

    parsing::chart::cell &cell = goal(0, 0);
    cell.sort();
    for (auto &item : cell)
    {
        unsigned token_id = 0;
        finalizer_callback(&item, &token_id, cache, finalizer_args);
    }

    return 0;
}
