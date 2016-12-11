

#include <omp.h>
#include "cacheable.h"

namespace myccg {

// std::unordered_map<std::string, const Cacheable*> Cacheable::cache;
int Cacheable::ids = 0;

Cacheable::Cacheable() {
    #pragma omp atomic capture
    id_ = ids++;
}

void Cacheable::RegisterCache(const std::string& key) const {
    #pragma omp critical(RegisterCache)
    cache_().emplace(key, this);
}

} // namespace myccg
