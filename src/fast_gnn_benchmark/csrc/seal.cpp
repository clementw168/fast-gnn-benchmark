// SEAL: Subgraph Extraction and Labeling for link prediction.
//
// Key performance choices:
//  - SEALExtractor holds two flat arrays (state_gen, local_idx) of size
//    num_nodes. "Clearing" between calls is a single ++gen (O(1)).
//    No hash map allocation per call.
//  - vec_to_numpy: transfers vector ownership to numpy via a capsule.
//    Zero copies between C++ and Python.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using i64 = int64_t;

static constexpr int INF = std::numeric_limits<int>::max() / 2;

// ---------------------------------------------------------------------------
// Zero-copy vector → numpy array.
// Moves the vector onto the heap, wraps it in a capsule so the numpy array
// owns the memory, and returns the array pointing at the vector's buffer.
// torch.from_numpy() on the result shares the same buffer (no copy).
// ---------------------------------------------------------------------------
template<typename T>
static py::array_t<T> vec_to_numpy(std::vector<T> vec) {
    auto* ptr = new std::vector<T>(std::move(vec));
    py::capsule cap(ptr, [](void* p) {
        delete static_cast<std::vector<T>*>(p);
    });
    return py::array_t<T>(
        {static_cast<py::ssize_t>(ptr->size())},  // shape
        {static_cast<py::ssize_t>(sizeof(T))},    // strides
        ptr->data(),                               // data pointer
        cap                                        // keeps vector alive
    );
}

// ---------------------------------------------------------------------------
// BFS on a local (re-indexed) adjacency list.
// mask: treat this node as removed from the graph.
// ---------------------------------------------------------------------------
static std::vector<int> bfs_local(
    const std::vector<std::vector<int>>& adj,
    int source, int n, int mask = -1)
{
    std::vector<int> dist(n, INF);
    if (source < 0 || source >= n || source == mask) return dist;
    dist[source] = 0;
    std::queue<int> q;
    q.push(source);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (v == mask) continue;
            if (dist[v] == INF) { dist[v] = dist[u] + 1; q.push(v); }
        }
    }
    return dist;
}

// ---------------------------------------------------------------------------
// DRNL node labeling. local index 0 = src, 1 = dst.
// ---------------------------------------------------------------------------
static std::vector<i64> compute_drnl(
    const std::vector<std::vector<int>>& adj, int n)
{
    auto d_src = bfs_local(adj, 0, n, /*mask=*/1);
    auto d_dst = bfs_local(adj, 1, n, /*mask=*/0);
    std::vector<i64> z(n, 0);
    for (int i = 0; i < n; ++i) {
        if (i == 0 || i == 1) { z[i] = 1; continue; }
        int ds = d_src[i], dd = d_dst[i];
        if (ds >= INF || dd >= INF) continue;
        int d = ds + dd, d2 = d / 2, dm = d % 2;
        z[i] = (i64)1 + std::min(ds, dd) + (i64)d2 * (d2 + dm - 1);
    }
    return z;
}

// ---------------------------------------------------------------------------
// SEALExtractor
//
// Holds two flat arrays over the full node space:
//   state_gen_[v] == gen_  →  v is in the current subgraph
//   local_idx_[v]          →  v's local index (valid iff state_gen_[v]==gen_)
//
// Advancing between calls: just ++gen_ (O(1)).
// Full array reset only on uint32_t overflow (every ~4 billion calls).
//
// NOT thread-safe: each DataLoader worker process should own its own instance
// (handled automatically by Python multiprocessing fork + pickle).
// ---------------------------------------------------------------------------
class SEALExtractor {
public:
    int64_t num_nodes;

    explicit SEALExtractor(int64_t n)
        : num_nodes(n), state_gen_(n, 0), local_idx_(n, -1), gen_(0) {}


    // Single-edge extraction.
    py::dict extract(
        py::array_t<i64, py::array::c_style | py::array::forcecast> row_ptr_arr,
        py::array_t<i64, py::array::c_style | py::array::forcecast> col_idx_arr,
        i64 src, i64 dst,
        const std::vector<int>& num_neighbors,
        unsigned int seed = 42)
    {
        auto rp = row_ptr_arr.unchecked<1>();
        auto ci = col_idx_arr.unchecked<1>();
        std::mt19937 rng(seed);
        return extract_impl(rp.data(0), ci.data(0), src, dst, num_neighbors, rng);
    }

    // Batch extraction. Each edge i is seeded with base_seed ^ i.
    py::list extract_batch(
        py::array_t<i64, py::array::c_style | py::array::forcecast> row_ptr_arr,
        py::array_t<i64, py::array::c_style | py::array::forcecast> col_idx_arr,
        py::array_t<i64, py::array::c_style | py::array::forcecast> src_arr,
        py::array_t<i64, py::array::c_style | py::array::forcecast> dst_arr,
        const std::vector<int>& num_neighbors,
        unsigned int base_seed = 42)
    {
        auto rp   = row_ptr_arr.unchecked<1>();
        auto ci   = col_idx_arr.unchecked<1>();
        auto srcs = src_arr.unchecked<1>();
        auto dsts = dst_arr.unchecked<1>();

        py::list results;
        for (py::ssize_t i = 0; i < srcs.shape(0); ++i) {
            std::mt19937 rng(base_seed ^ (unsigned int)i);
            results.append(
                extract_impl(rp.data(0), ci.data(0),
                             srcs(i), dsts(i), num_neighbors, rng));
        }
        return results;
    }

private:
    std::vector<uint32_t> state_gen_;  // 4 bytes × num_nodes
    std::vector<int32_t>  local_idx_;  // 4 bytes × num_nodes
    uint32_t gen_;

    bool in_subgraph(i64 v) const { return state_gen_[v] == gen_; }

    void mark(i64 v, int li) {
        state_gen_[v] = gen_;
        local_idx_[v] = li;
    }

    void advance_gen() {
        if (++gen_ == 0) {
            // uint32 overflow: reset and start from 1
            std::fill(state_gen_.begin(), state_gen_.end(), 0);
            gen_ = 1;
        }
    }

    py::dict extract_impl(
        const i64* rp, const i64* ci,
        i64 src, i64 dst,
        const std::vector<int>& num_neighbors,
        std::mt19937& rng)
    {
        advance_gen();

        // -------------------------------------------------------------------
        // Phase 1: k-hop BFS with per-hop per-node neighbour sampling.
        // -------------------------------------------------------------------
        int num_hops = (int)num_neighbors.size();

        std::vector<i64>     nodes = {src, dst};
        std::vector<int32_t> hops  = {0, 0};
        mark(src, 0);
        mark(dst, 1);
        std::vector<i64> fringe = {src, dst};

        for (int hop = 1; hop <= num_hops; ++hop) {
            int limit = num_neighbors[hop - 1];  // -1 = no limit
            std::unordered_set<i64> new_fringe_set;

            for (i64 u : fringe) {
                std::vector<i64> nbrs;
                nbrs.reserve(rp[u + 1] - rp[u]);
                for (i64 j = rp[u]; j < rp[u + 1]; ++j) {
                    i64 v = ci[j];
                    if (!in_subgraph(v)) nbrs.push_back(v);
                }
                if (limit >= 0 && (int)nbrs.size() > limit) {
                    std::shuffle(nbrs.begin(), nbrs.end(), rng);
                    nbrs.resize(limit);
                }
                for (i64 v : nbrs) new_fringe_set.insert(v);
            }

            std::vector<i64> new_fringe(new_fringe_set.begin(), new_fringe_set.end());
            if (new_fringe.empty()) break;

            int base_li = (int)nodes.size();
            for (int k = 0; k < (int)new_fringe.size(); ++k) {
                mark(new_fringe[k], base_li + k);
                nodes.push_back(new_fringe[k]);
                hops.push_back(hop);
            }
            fringe = std::move(new_fringe);
        }

        int n = (int)nodes.size();

        // -------------------------------------------------------------------
        // Phase 2: local adjacency and edge index.
        // g2l lookup: local_idx_[v] (valid when in_subgraph(v)).
        // Target edge (local 0↔1) is removed.
        // -------------------------------------------------------------------
        std::vector<std::vector<int>> local_adj(n);
        std::vector<i64> esrc, edst;

        for (int li = 0; li < n; ++li) {
            i64 u = nodes[li];
            for (i64 j = rp[u]; j < rp[u + 1]; ++j) {
                i64 v = ci[j];
                if (!in_subgraph(v)) continue;
                int lv = local_idx_[v];
                if ((li == 0 && lv == 1) || (li == 1 && lv == 0)) continue;
                local_adj[li].push_back(lv);
                esrc.push_back((i64)li);
                edst.push_back((i64)lv);
            }
        }

        // -------------------------------------------------------------------
        // Phase 3: DRNL labeling.
        // -------------------------------------------------------------------
        py::dict d;
        d["node_ids"]  = vec_to_numpy(std::move(nodes));
        d["edge_src"]  = vec_to_numpy(std::move(esrc));
        d["edge_dst"]  = vec_to_numpy(std::move(edst));
        d["z_drnl"]    = vec_to_numpy(compute_drnl(local_adj, n));
        d["hop_dists"] = vec_to_numpy(std::move(hops));
        return d;
    }
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SEAL subgraph extraction + DRNL labeling (C++)";

    py::class_<SEALExtractor>(m, "SEALExtractor")
        .def(py::init<int64_t>(), py::arg("num_nodes"),
             "Create an extractor for a graph with num_nodes nodes.")
        .def("extract", &SEALExtractor::extract,
             py::arg("row_ptr"), py::arg("col_idx"),
             py::arg("src"), py::arg("dst"),
             py::arg("num_neighbors"), py::arg("seed") = 42,
             R"doc(
Extract the k-hop enclosing subgraph around edge (src, dst) and compute DRNL.

Args:
    row_ptr:       CSR row pointer array, int64, shape [num_nodes+1].
    col_idx:       CSR column index array, int64, shape [num_edges].
    src, dst:      Target edge endpoints (global indices).
    num_neighbors: Per-hop neighbour limit list, e.g. [20, 10].
                   len = number of hops. -1 = no limit at that hop.
    seed:          RNG seed for sampling.

Returns:
    dict: node_ids, edge_src, edge_dst, z_drnl, hop_dists (all numpy arrays).
)doc")
        .def("extract_batch", &SEALExtractor::extract_batch,
             py::arg("row_ptr"), py::arg("col_idx"),
             py::arg("src_arr"), py::arg("dst_arr"),
             py::arg("num_neighbors"), py::arg("base_seed") = 42,
             "Batch version of extract. Edge i is seeded with base_seed ^ i.")
        .def(py::pickle(
            [](const SEALExtractor& e) { return py::make_tuple(e.num_nodes); },
            [](py::tuple t) { return SEALExtractor(t[0].cast<int64_t>()); }
        ));
}
