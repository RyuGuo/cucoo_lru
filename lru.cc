#include <cstdint>
#include <hashtable.h>

#include <atomic>
#include <cassert>
#include <functional>
#include <list>
#include <mutex>
#include <sched.h>
#include <shared_mutex>
#include <unistd.h>
#include <unordered_map>
#include <vector>

class hash {
  int seed;

public:
  hash() : seed(0) {}
  hash(int seed) : seed(seed) {}

  uint64_t operator()(uint64_t x) {
    return seed ^
           (std::hash<uint64_t>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  }
};

template <typename T>
static const void
atomic_max(std::atomic<T> &e, T val,
           std::memory_order s = std::memory_order::memory_order_seq_cst) {
  T _e = e.load();
  while (!e.compare_exchange_weak(_e, std::max(_e, val), s))
    ;
}

template <typename T>
inline bool atomic_compare_exchange_weak(
    T *a, T &e, T i,
    std::memory_order s = std::memory_order::memory_order_seq_cst) {
  return __atomic_compare_exchange(a, &e, &i, true, int(s), int(s));
}

template <typename T>
inline void
atomic_max(T *e, T val,
           std::memory_order s = std::memory_order::memory_order_seq_cst) {
  T _e = *e;
  while (!atomic_compare_exchange_weak(e, _e, std::max(_e, val), s))
    ;
}

template <typename T>
inline void
atomic_min(T *e, T val,
           std::memory_order s = std::memory_order::memory_order_seq_cst) {
  T _e = *e;
  while (!atomic_compare_exchange_weak(e, _e, std::min(_e, val), s))
    ;
}

template <typename T>
inline T atomic_add_fetch(
    T *a, T i, std::memory_order s = std::memory_order::memory_order_seq_cst) {
  return __atomic_add_fetch(a, i, int(s));
}

template <typename T>
inline T atomic_sub_fetch(
    T *a, T i, std::memory_order s = std::memory_order::memory_order_seq_cst) {
  return __atomic_sub_fetch(a, i, int(s));
}

inline void barrier() { __asm__ __volatile__("" : : : "memory"); }

class shared_mutex {
  alignas(alignof(uint32_t)) uint32_t shared_lck;

public:
  shared_mutex() : shared_lck(0) {}

  bool try_lock_shared() {
    uint32_t expect = (shared_lck & (~1U));
    uint32_t desire = expect + 2U;
    return atomic_compare_exchange_weak(
        &shared_lck, expect, desire, std::memory_order::memory_order_relaxed);
  }
  void unlock_shared() { atomic_sub_fetch(&shared_lck, 2U); }
  void lock() {
    uint32_t l;
    do {
      l = 0;
    } while (!atomic_compare_exchange_weak(
        &shared_lck, l, 1U, std::memory_order::memory_order_relaxed));
  }
  void unlock() {
    atomic_sub_fetch(&shared_lck, 1U, std::memory_order::memory_order_relaxed);
  }
};

std::atomic<int> trash_lease = {0};

template <typename V> class CucooHashTable {
  struct val_wrapper {
    V v;
    uint32_t shared_cnt;
    val_wrapper(const V &v) : v(v), shared_cnt(0) { ++trash_lease; }
    void Init(const V &v) {
      this->v = v;
      shared_cnt = 0;
    }
    ~val_wrapper() { --trash_lease; }
  };

  struct entry {
    uint64_t k;
    // ??????hazard??????
    val_wrapper *vw;
    shared_mutex lck;
    entry() : vw(nullptr) {}
    ~entry() {
      if (vw)
        delete vw;
    }
  };

  // ??????????????????????????????bucket??????????????????evict????????????
  uint8_t linear_detect_threshold;
  uint64_t invalid_k;
  std::atomic<size_t> entry_cnt;
  size_t bucket_size;
  std::vector<hash> hashers;
  std::vector<std::vector<entry>> buckets;
  std::mutex expand_lck;
  std::function<void(V *v)> evict_fn;

  // ?????????val_wrapper????????????
  static thread_local int vw_recycle_bin_id;
  std::vector<std::vector<val_wrapper *>> vw_recycle_bin;

public:
  class f_iterator;

  class iterator {
    CucooHashTable &_hashtable;
    size_t bid;
    size_t eid;
    val_wrapper *ptr;

    friend class f_iterator;
    friend class CucooHashTable;

    iterator(CucooHashTable &_hashtable, size_t bid, size_t eid)
        : _hashtable(_hashtable), bid(bid), eid(eid), ptr(nullptr) {
      if (bid == -1 && eid == -1)
        return;
      auto &e = _hashtable.buckets[bid][eid];
      ptr = e.vw;
      /**
       * ??????????????????erase???k????????????invalid???vm?????????nullptr
       * ???????????????????????????ptr??????????????????vw???????????????????????????
       */
      if (ptr == nullptr) {
        bid = -1;
        eid = -1;
        return;
      }
      atomic_add_fetch(&ptr->shared_cnt, 1U,
                       std::memory_order::memory_order_relaxed);
    }

  public:
    iterator(const iterator &it)
        : _hashtable(it._hashtable), bid(it.bid), eid(it.eid), ptr(it.ptr) {
      if (bid == -1 && eid == -1)
        return;
      if (it.ptr == nullptr) {
        fprintf(stderr, "fuck [%lu][%lu]\n", bid, eid);
        assert(it.ptr != nullptr);
      }
      atomic_add_fetch(&ptr->shared_cnt, 1U,
                       std::memory_order::memory_order_relaxed);
    }
    ~iterator() {
      // end()??????????????????
      if (bid == -1 && eid == -1)
        return;
      // ??????????????????
      if (ptr == nullptr)
        return;
      // ???erase??????????????????????????????????????????cnt == 0?????????ptr
      uint32_t cnt = atomic_sub_fetch(&ptr->shared_cnt, 1U,
                                      std::memory_order::memory_order_relaxed);
      if (cnt == 0) {
        _hashtable.evict_fn(&ptr->v);
        _hashtable.vw_recycle_bin[vw_recycle_bin_id].push_back(ptr);
      }
    }

    void incur() {
      /**
       * ????????????????????????????????????erase?????????ptr?????????????????????shared_cnt > 0
       * ??????????????????????????????ptr???shared_cnt?????????cnt == 0?????????ptr
       */
      uint32_t cnt = atomic_sub_fetch(&ptr->shared_cnt, 1U,
                                      std::memory_order::memory_order_relaxed);
      if (cnt == 0) {
        _hashtable.evict_fn(&ptr->v);
        _hashtable.vw_recycle_bin[vw_recycle_bin_id].push_back(ptr);
      }
      ++eid;
      while (bid < _hashtable.buckets.size()) {
        while (eid < _hashtable.bucket_size) {
          auto &e = _hashtable.buckets[bid][eid];
          if (e.k != _hashtable.invalid_k) {
            ptr = e.vw;
            // ????????????erase???vw???nullptr
            if (ptr != nullptr) {
              atomic_add_fetch(&ptr->shared_cnt, 1U,
                               std::memory_order::memory_order_relaxed);
              return;
            }
          }
          ++eid;
        }
        ++bid;
        eid = 0;
      }
      bid = -1;
      eid = -1;
      ptr = nullptr;
    }

    V &operator*() { return ptr->v; }

    V *operator->() { return &ptr->v; }

    iterator &operator++() {
      incur();
      return *this;
    }

    iterator operator++(int) {
      iterator __tmp(*this);
      incur();
      return __tmp;
    }

    bool operator==(const typename CucooHashTable<V>::iterator &y) const {
      return &_hashtable == &y._hashtable && bid == y.bid && eid == y.eid;
    }

    bool operator!=(const typename CucooHashTable<V>::iterator &y) const {
      return &_hashtable != &y._hashtable || bid != y.bid || eid != y.eid;
    }
  };

  class f_iterator {
    CucooHashTable &_hashtable;
    size_t bid;
    size_t eid;

    friend class CucooHashTable;
    friend class iterator;

  public:
    f_iterator(CucooHashTable &_hashtable, size_t bid, size_t eid)
        : _hashtable(_hashtable), bid(bid), eid(eid) {}

    f_iterator(const f_iterator &it)
        : _hashtable(it._hashtable), bid(it.bid), eid(it.eid) {}

    f_iterator(const iterator &it)
        : _hashtable(it._hashtable), bid(it.bid), eid(it.eid) {}

    bool operator==(const typename CucooHashTable<V>::iterator &y) const {
      return &_hashtable == &y._hashtable && bid == y.bid && eid == y.eid;
    }

    bool operator!=(const typename CucooHashTable<V>::iterator &y) const {
      return &_hashtable != &y._hashtable || bid != y.bid || eid != y.eid;
    }

    bool operator<(const typename CucooHashTable<V>::iterator &y) const {
      return &_hashtable != &y._hashtable ||
             ((bid < y.bid) || (bid == y.bid && eid < y.eid));
    }

    bool operator>(const typename CucooHashTable<V>::iterator &y) const {
      return &_hashtable != &y._hashtable ||
             ((bid > y.bid) || (bid == y.bid && eid > y.eid));
    }
  };

  const iterator end() { return iterator(*this, -1, -1); }
  iterator begin() {
    for (size_t bid = 0; bid < buckets.size(); ++bid) {
      for (size_t eid = 0; eid < bucket_size; ++eid) {
        if (buckets[bid][eid].k != invalid_k) {
          iterator it(*this, bid, eid);
          if (it.ptr != nullptr)
            return it;
        }
      }
    }
    return end();
  }

  iterator ranged_begin(uint64_t range_id, uint64_t range_num) {
    uint64_t rangesize = buckets.size() * bucket_size / range_num;
    uint64_t idx = range_id * rangesize;
    size_t bid = idx / bucket_size, eid = idx % bucket_size;
    for (; bid < buckets.size(); ++bid) {
      for (; eid < bucket_size && rangesize > 0; ++eid, --rangesize) {
        if (buckets[bid][eid].k != invalid_k) {
          iterator it(*this, bid, eid);
          if (it.ptr != nullptr) {
            fprintf(stderr, "rbegin [%lu / %lu] [%lu][%lu]\n", range_id,
                    range_num, bid, eid);
            return it;
          }
        }
      }
      eid = 0;
    }
    return end();
  }

  const f_iterator ranged_end(uint64_t range_id, uint64_t range_num) {
    if (range_id == range_num - 1)
      return end();
    uint64_t idx = (range_id + 1) * buckets.size() * bucket_size / range_num;
    uint64_t bid = idx / bucket_size;
    uint64_t eid = idx % bucket_size;
    fprintf(stderr, "rend [%lu / %lu] [%lu][%lu]\n", range_id, range_num, bid,
            eid);
    return f_iterator{*this, bid, eid};
  }

  CucooHashTable(size_t bucket_size = 1000, uint64_t invalid_k = -1UL,
                 std::function<void(V *v)> evict_fn = nullptr,
                 uint8_t linear_detect_threshold = 4)
      : bucket_size(bucket_size), entry_cnt(0), invalid_k(invalid_k),
        evict_fn(evict_fn), linear_detect_threshold(linear_detect_threshold),
        vw_recycle_bin(64) {
    entry e;
    e.k = invalid_k;
    hashers.emplace_back(hash());
    buckets.emplace_back(std::vector<entry>(bucket_size, e));
  }

  ~CucooHashTable() {
    fprintf(stderr, "buckets: %lu\n", buckets.size());
    for (auto bin : vw_recycle_bin) {
      for (auto vw_ptr : bin) {
        delete vw_ptr;
      }
    }
  }

  size_t size() { return entry_cnt; }

  iterator find(uint64_t k) {
    for (size_t bid = 0; bid < buckets.size(); ++bid) {
      size_t eid = hashers[bid](k) % bucket_size;
      for (uint8_t t = 0; t < linear_detect_threshold;
           ++t, eid = (eid + 1) % bucket_size) {
        /**
         * ??????????????????emplace??????k????????????CAS?????????????????????????????????
         * k??????????????????????????????????????????????????????
         * ??????k??????????????????v????????????
         */
        if (buckets[bid][eid].k == k) {
          auto it = iterator(*this, bid, eid);
          /**
           * ??????????????????????????????????????????????????????????????????v?????????
           * ??????ptr???????????????????????????
           */
          return (it.ptr != nullptr) ? it : end();
        }
      }
    }
    return end();
  }

  std::pair<iterator, bool> emplace(uint64_t k, const V &v) {
    size_t bid = 0;
  retry:
    for (; bid < buckets.size(); ++bid) {
      size_t eid = hashers[bid](k) % bucket_size;
      // ???????????????????????????????????????????????????????????????
      uint64_t _k = invalid_k;
      for (uint8_t t = 0; t < linear_detect_threshold;
           ++t, eid = (eid + 1) % bucket_size) {
        auto &e = buckets[bid][eid];
        /**
         * 1) ????????????????????????k empalce:
         * ???invalid_k?????????k????????????????????????????????????????????????????????????
         * i) ?????????k??????ii) ??????????????????????????????k
         * ???????????????k??????????????????????????????????????????bucket???
         *                  ????????????????????????????????????????????????????????????
         *
         * ??????????????????????????????
         *  ?????????k emplace???????????????t1???????????????????????????slot????????????????????????
         *  ???????????????t2???????????????????????????evict??????????????????slot?????????
         *  ??????t2????????????t1??????????????????t1???slot????????????????????????????????????
         *
         *  ??????evict??????????????????????????????t1???slot??????????????????
         */
        if (atomic_compare_exchange_weak(&e.k, _k, k)) {
          val_wrapper *vw;
          if (vw_recycle_bin[vw_recycle_bin_id].empty()) {
            vw = new val_wrapper(v);
          } else {
            vw = vw_recycle_bin[vw_recycle_bin_id].back();
            vw->Init(v);
            vw_recycle_bin[vw_recycle_bin_id].pop_back();
          }
          // ???????????????????????????
          atomic_add_fetch(&vw->shared_cnt, 1U,
                           std::memory_order::memory_order_relaxed);
          e.vw = vw;
          fprintf(stderr, "emplace [%lu][%lu] k: %lu\n", bid, eid, k);
          /**
           * ???????????????????????????
           *  e.vw???????????????????????????epoch??????????????????evict?????????????????????entry???
           *  ???vm????????????nullptr?????????k??????????????????emplace?????????????????????
           */
          iterator it(*this, bid, eid);
          if (it.ptr != nullptr) {
            entry_cnt.fetch_add(1, std::memory_order::memory_order_relaxed);
            return {it, true};
          }
          return {end(), false};
        } else if (_k == k) {
          return {end(), false};
        }
      }
    }
    /**
     * ?????????????????????????????????????????????????????????????????????bucket
     * ??????????????????expand???????????????????????????
     * ??????????????????(?????????????????????expand?????????????????????????????????????????????????????????)???
     * ?????????expand?????????????????????????????????
     */
    if (expand_lck.try_lock()) {
      entry e;
      e.k = invalid_k;
      hashers.emplace_back(hash());
      buckets.emplace_back(std::vector<entry>(bucket_size, e));
    } else {
      expand_lck.lock();
    }
    expand_lck.unlock();
    goto retry;
  }

  void erase(const iterator &pos) {
    /**
     * ?????????erase???emplace?????????k?????????
     * ????????????find???????????????
     */
    entry_cnt.fetch_sub(1, std::memory_order::memory_order_relaxed);
    // ???????????????????????????????????????????????????
    atomic_sub_fetch(&buckets[pos.bid][pos.eid].vw->shared_cnt, 1U,
                     std::memory_order::memory_order_relaxed);
    fprintf(stderr, "erase iter: [%lu][%lu] k: %lu\n", pos.bid, pos.eid,
            buckets[pos.bid][pos.eid].k);
    auto old_vw = buckets[pos.bid][pos.eid].vw;
    buckets[pos.bid][pos.eid].vw = nullptr;
    /**
     * ?????????????????????emplace
     * erase??????pos?????????????????????old_vw
     * ?????????????????????????????????????????????????????????
     */
    barrier();
    buckets[pos.bid][pos.eid].k = invalid_k;
  }
};

std::atomic<int> gen_id = {0};
template <typename V>
thread_local int CucooHashTable<V>::vw_recycle_bin_id = gen_id++;

template <typename T> class cache {
protected:
  size_t evict_threshold_size;
  std::atomic<size_t> total_cache_size;
  struct {
    std::atomic<size_t> hit;
    std::atomic<size_t> miss;
    std::atomic<size_t> load;
    std::atomic<size_t> evict;
  } statistic;

  bool need_evict() { return total_cache_size >= evict_threshold_size; }

public:
  cache(size_t evict_threshold_size)
      : evict_threshold_size(evict_threshold_size), total_cache_size(0) {
    statistic.hit = 0;
    statistic.miss = 0;
    statistic.load = 0;
    statistic.evict = 0;
  }

  size_t get_cache_size() { return total_cache_size; }
  size_t get_statistic_hit() { return statistic.hit; }
  size_t get_statistic_miss() { return statistic.miss; }
  size_t get_statistic_load() { return statistic.load; }
  size_t get_statistic_evict() { return statistic.evict; }

  virtual std::pair<size_t, T *> get_cache(uint64_t id) = 0;
};

template <typename T> class epoch_lru : public cache<T> {
  struct lru_entry {
    uint64_t epoch;
    uint64_t id;
    size_t cache_item_size;
    T *cache_item;
  };

  static const int evict_epoch_radio = 10;

  uint64_t invalid_id;
  CucooHashTable<lru_entry> lru_hashtable;
  std::atomic<uint64_t> max_epoch;
  std::atomic<uint64_t> min_epoch;
  std::function<std::pair<T *, size_t>(uint64_t id)> load_fn;
  std::function<void(uint64_t id, T *cache_item, size_t size)> evict_cache_fn;

  std::mutex evict_lck;
  const static uint64_t evict_exp_ato_lck_mask = 1UL << 63;
  std::atomic<uint64_t> evict_exp_ato;
  std::atomic<uint64_t> evict_exp_barrier_ato;

  uint64_t get_epoch() {
    static uint64_t gen = 0;
    // ?????????????????????
    return gen++;
  }

  void evict_exp() {
    uint64_t id = evict_exp_ato.load();
    do {
      if (id & evict_exp_ato_lck_mask) {
        while (evict_exp_ato != 0)
          ;
        return;
      }
    } while (!evict_exp_ato.compare_exchange_weak(id, id + 1));
    usleep(2);
    evict_exp_ato |= evict_exp_ato_lck_mask;
    uint64_t num = evict_exp_ato & ~evict_exp_ato_lck_mask;
    // barrier??????????????????evict_exp_ato?????????num???
    ++evict_exp_barrier_ato;
    while (evict_exp_barrier_ato.load() != num)
      ;
    fprintf(stderr, "range %lu / %lu\n", id, num);

    size_t need_evict_cnt = 0;
    size_t need_evict_size = 0;
    uint64_t evict_epoch =
        ((max_epoch - min_epoch) * evict_epoch_radio + 99) / 100 + min_epoch;
    int i = 0;
    auto rend = lru_hashtable.ranged_end(id, num);
    for (auto entry_it = lru_hashtable.ranged_begin(id, num); rend > entry_it;
         ++i) {
      fprintf(stderr, "%d foreach k: %lu, epoch: %lu\n", i, entry_it->id,
              entry_it->epoch);
      evict_epoch =
          ((max_epoch - min_epoch) * evict_epoch_radio + 99) / 100 + min_epoch;
      assert(evict_epoch < max_epoch);
      fprintf(stderr,
              "%lu / %lu:  min_epoch: %lu, max_epoch: %lu, evict_epoch: %lu\n",
              id, num, min_epoch.load(), max_epoch.load(), evict_epoch);
      if (entry_it->epoch <= evict_epoch) {
        ++need_evict_cnt;
        assert(entry_it->cache_item_size > 0);
        need_evict_size += entry_it->cache_item_size;
        lru_hashtable.erase(entry_it++);
      } else {
        ++entry_it;
      }
    }
    atomic_max(min_epoch, evict_epoch + 1);
    fprintf(stderr, "evict result: size: %lu\n", need_evict_size);
    this->total_cache_size.fetch_sub(need_evict_size,
                                     std::memory_order::memory_order_relaxed);
    this->statistic.evict.fetch_add(need_evict_cnt,
                                    std::memory_order::memory_order_relaxed);

    uint64_t ato = --evict_exp_ato;
    if ((ato & ~evict_exp_ato_lck_mask) == 0) {
      fprintf(stderr, "evict complete\n");
      evict_exp_barrier_ato = 0;
      evict_exp_ato &= ~evict_exp_ato_lck_mask;
    }
  }

  void evict() {
    // ??????????????????evict?????????????????????evict?????????????????????
    if (!evict_lck.try_lock())
      return;
  retry:
    uint64_t evict_epoch =
        ((max_epoch - min_epoch) * evict_epoch_radio + 99) / 100 + min_epoch;
    size_t need_evict_cnt = 0;
    size_t need_evict_size = 0;
    fprintf(stderr, "min_epoch: %lu, max_epoch: %lu, evict_epoch: %lu\n",
            min_epoch.load(), max_epoch.load(), evict_epoch);
    int i = 0;
    for (auto entry_it = lru_hashtable.begin(); entry_it != lru_hashtable.end();
         ++i) {
      // fprintf(stderr, "%d foreach k: %lu, epoch: %lu\n", i, entry_it->id,
      //         entry_it->epoch);
      if (entry_it->epoch <= evict_epoch) {
        ++need_evict_cnt;
        assert(entry_it->cache_item_size > 0);
        need_evict_size += entry_it->cache_item_size;
        lru_hashtable.erase(entry_it++);
      } else {
        ++entry_it;
      }
    }
    min_epoch = evict_epoch + 1;
    if (need_evict_cnt == 0) {
      fprintf(stderr, "retry evict\n");
      goto retry;
    }
    fprintf(stderr, "evict result: size: %lu\n", need_evict_size);
    this->total_cache_size.fetch_sub(need_evict_size,
                                     std::memory_order::memory_order_relaxed);
    this->statistic.evict.fetch_add(need_evict_cnt,
                                    std::memory_order::memory_order_relaxed);
    evict_lck.unlock();
  }

  std::pair<typename CucooHashTable<lru_entry>::iterator, bool>
  load_cache(uint64_t id) {
    if (this->need_evict()) {
      evict_exp();
    }
    uint64_t current_epoch = get_epoch();
    atomic_max(max_epoch, current_epoch);
    // ????????????????????????
    auto res =
        lru_hashtable.emplace(id, lru_entry{.epoch = UINT64_MAX, .id = id});
    if (res.second) {
      auto r = load_fn(id);
      res.first->cache_item = r.first;
      res.first->cache_item_size = r.second;
      this->total_cache_size.fetch_add(r.second,
                                       std::memory_order::memory_order_relaxed);
      barrier();
      atomic_min(&res.first->epoch, current_epoch,
                 std::memory_order::memory_order_relaxed);
    }
    return res;
  }

public:
  epoch_lru(size_t evict_threshold_size,
            std::function<std::pair<T *, size_t>(uint64_t id)> load_fn,
            std::function<void(uint64_t id, T *cache_item, size_t size)>
                evict_cache_fn,
            uint64_t invalid_id = -1UL)
      : cache<T>(evict_threshold_size), min_epoch(get_epoch()),
        max_epoch(get_epoch() + 1), evict_cache_fn(evict_cache_fn),
        invalid_id(invalid_id),
        lru_hashtable(100, invalid_id,
                      [evict_cache_fn](epoch_lru::lru_entry *e) {
                        evict_cache_fn(e->id, e->cache_item,
                                       e->cache_item_size);
                      }),
        evict_exp_ato(0), evict_exp_barrier_ato(0), load_fn(load_fn) {}

  std::pair<size_t, T *> get_cache(uint64_t id) override {
  retry:
    auto it = lru_hashtable.find(id);
    if (it == lru_hashtable.end()) {
      fprintf(stderr, "cache miss %lu\n", id);
      this->statistic.miss.fetch_add(1,
                                     std::memory_order::memory_order_relaxed);
      auto load_cache_res = load_cache(id);
      if (load_cache_res.second) {
        this->statistic.load.fetch_add(1,
                                       std::memory_order::memory_order_relaxed);
        return {load_cache_res.first->cache_item_size,
                load_cache_res.first->cache_item};
      }
      goto retry;
    }
    this->statistic.hit.fetch_add(1, std::memory_order::memory_order_relaxed);
    uint64_t current_epoch = get_epoch();
    atomic_max(max_epoch, current_epoch,
               std::memory_order::memory_order_relaxed);
    atomic_max(&it->epoch, current_epoch,
               std::memory_order::memory_order_relaxed);
    return {it->cache_item_size, it->cache_item};
  }
};

#include <iostream>
#include <thread>
using namespace std;

int main() {
  int arr[1000] = {0};
  {
    epoch_lru<int> lru(
        100 * sizeof(int),
        [&arr](uint64_t id) -> pair<int *, size_t> {
          std::cout << "load " << id << endl;
          return {arr + id, sizeof(int)};
        },
        [](uint64_t id, int *cache_item, size_t size) {
          std::cout << "evict " << id << endl;
        });

    vector<thread> ths;
    for (int id = 0; id < 4; ++id) {
      ths.push_back(thread([id, &lru]() {
        uint64_t seed = 0x90235 + id;
        for (int i = 0; i < 10000; i++) {
          seed = (seed * 0x34897) ^ 0x5a4867 ^ (seed >> 3);
          uint64_t id = seed % 1000;
          std::cout << "get " << id << endl;
          // std::cout << "cache in memory: " << lru.get_cache_size() <<
          // std::endl;
          std::pair<size_t, int *> p = lru.get_cache(id);
        }
      }));
    }

    for (int id = 0; id < ths.size(); ++id) {
      ths[id].join();
    }

#define PVAL(name) std::cout << #name ": " << name << std::endl

    PVAL(lru.get_cache_size());
    PVAL(lru.get_statistic_hit());
    PVAL(lru.get_statistic_load());
    PVAL(lru.get_statistic_evict());
    PVAL(lru.get_statistic_miss());
    PVAL(trash_lease.load());
  }
  PVAL(trash_lease.load());
  return 0;
}