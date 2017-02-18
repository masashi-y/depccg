
#include <limits>
#include <string.h>
#include "chart.h"

namespace myccg {

ChartCell::ChartCell()
    : items(std::unordered_map<Cat, ChartItem>()),
      best_prob(std::numeric_limits<float>::lowest()), best(NULL) {}


bool ChartCell::update(NodeType parse, float prob) {
    Cat cat = parse->GetCategory();
    if (items.count(cat) > 0)
        return false;
    items.emplace(cat, std::make_pair(parse, prob));
    if (best_prob < prob) {
        best_prob = prob;
        best = parse;
    }
    return true;
}

Chart::Chart(int sent_size)
    : sent_size_(sent_size),
      chart_size_(sent_size * sent_size),
      chart_(new ChartCell*[chart_size_]),
      ending_cells_(new std::vector<ChartCell*>[sent_size + 1]),
      starting_cells_(new std::vector<ChartCell*>[sent_size + 1]) {

    memset(chart_, 0, sizeof(ChartCell*) * chart_size_);
}

Chart::~Chart() {
    for (int i = 0; i < chart_size_; i++) {
        if (chart_[i])
            delete chart_[i];
    }
    delete[] chart_;
    delete[] ending_cells_;
    delete[] starting_cells_;
}

bool Chart::IsEmpty() const {
    ChartCell* final_ = chart_[sent_size_ - 1];
    return (! final_ || final_->IsEmpty());
}

ChartCell* Chart::operator() (int row, int column) {
    ChartCell* cell = chart_[row * sent_size_ + column];
    if (! cell) {
        cell = new ChartCell();
        chart_[row * sent_size_ + column] = cell;
        ending_cells_[row + column + 1].push_back(cell);
        starting_cells_[row].push_back(cell);
    }
    return cell;
}

} // namespace myccg
