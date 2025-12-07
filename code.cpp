#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <queue>
#include <limits>
#include <chrono>
#include <functional>
#include <utility>
#include <iomanip>
#include <cmath>
#include <string>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ===================== Disjoint Set Union (Union-Find) =====================
struct DSU
{
    int n;
    vector<int> parent;
    vector<int> rankv;

    DSU(int n_ = 0) { init(n_); }

    void init(int n_)
    {
        n = n_;
        parent.resize(n);
        rankv.assign(n, 0);
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    // 带路径压缩的 find（只在串行区域用）
    int find(int x)
    {
        if (parent[x] == x)
            return x;
        parent[x] = find(parent[x]);
        return parent[x];
    }

    // 不带压缩的 find，用于并行只读查询，避免数据竞争
    int find_nocompress(int x) const
    {
        while (parent[x] != x)
        {
            x = parent[x];
        }
        return x;
    }

    bool unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return false;
        if (rankv[a] < rankv[b])
            std::swap(a, b);
        parent[b] = a;
        if (rankv[a] == rankv[b])
            ++rankv[a];
        return true;
    }
};

// ========================= Edge and Graph Types ============================
struct Edge
{
    int u;
    int v;
    double w;
};

struct Graph
{
    int n;
    vector<Edge> edges;
};

// ========================= Random Graph Generators =========================

// 生成随机连通图：先生成一棵随机生成树，再补齐剩余边
Graph generate_random_graph(int n, long long m, std::mt19937_64 &rng)
{
    if (m < n - 1)
    {
        m = n - 1; // 连通至少需要 n-1 条边
    }
    Graph g;
    g.n = n;
    g.edges.reserve(static_cast<size_t>(m));

    std::uniform_real_distribution<double> wdist(1.0, 1000.0);

    // 1) 随机生成树，保证整体连通
    for (int i = 1; i < n; ++i)
    {
        std::uniform_int_distribution<int> pdist(0, i - 1);
        int p = pdist(rng);
        double w = wdist(rng);
        g.edges.push_back({i, p, w});
    }

    // 2) 增加剩余的随机边（允许重复边）
    long long need = m - (n - 1);
    for (long long k = 0; k < need; ++k)
    {
        std::uniform_int_distribution<int> vdist(0, n - 1);
        int u = vdist(rng);
        int v = vdist(rng);
        if (u == v)
        {
            v = (v + 1) % n; // 避免自环
        }
        double w = wdist(rng);
        g.edges.push_back({u, v, w});
    }
    return g;
}

// 分层 / 社区图：顶点划分为多个社区，社区内稠密，社区间稀疏
Graph generate_layered_graph(int n, long long m, int num_communities, std::mt19937_64 &rng)
{
    if (num_communities < 1)
        num_communities = 1;
    if (m < n - 1)
        m = n - 1;
    Graph g;
    g.n = n;
    g.edges.reserve(static_cast<size_t>(m));

    std::uniform_real_distribution<double> wdist(1.0, 1000.0);

    vector<int> community_of(n);
    int base_size = n / num_communities;
    int extra = n % num_communities;
    int idx = 0;
    vector<pair<int, int>> community_range; // [start, end)
    community_range.reserve(num_communities);
    for (int c = 0; c < num_communities; ++c)
    {
        int size = base_size + (c < extra ? 1 : 0);
        int start = idx;
        int end = start + size;
        community_range.push_back(std::make_pair(start, end));
        for (int v = start; v < end; ++v)
        {
            community_of[v] = c;
        }
        idx = end;
    }

    // 1) 每个社区内部先生成一棵生成树
    for (size_t i = 0; i < community_range.size(); ++i)
    {
        int start = community_range[i].first;
        int end = community_range[i].second;
        for (int v = start + 1; v < end; ++v)
        {
            std::uniform_int_distribution<int> pdist(start, v - 1);
            int p = pdist(rng);
            double w = wdist(rng);
            g.edges.push_back({v, p, w});
        }
    }

    // 2) 用链式方式连接不同社区（保证整体连通）
    for (int c = 0; c < num_communities - 1; ++c)
    {
        int startA = community_range[c].first;
        int endA = community_range[c].second;
        int startB = community_range[c + 1].first;
        int endB = community_range[c + 1].second;
        std::uniform_int_distribution<int> distA(startA, endA - 1);
        std::uniform_int_distribution<int> distB(startB, endB - 1);
        int u = distA(rng);
        int v = distB(rng);
        double w = wdist(rng);
        g.edges.push_back({u, v, w});
    }

    long long current_m = static_cast<long long>(g.edges.size());
    long long need = m - current_m;
    if (need < 0)
        need = 0; // 用户设置的 m 太小时保护

    // 3) 再随机加边：70% 概率在社区内，30% 概率跨社区
    for (long long k = 0; k < need; ++k)
    {
        std::uniform_int_distribution<int> vdist(0, n - 1);
        int u = vdist(rng);
        int v;
        if (std::uniform_int_distribution<int>(0, 9)(rng) < 7)
        {
            // 70%：社区内边
            int c = community_of[u];
            int start = community_range[c].first;
            int end = community_range[c].second;
            std::uniform_int_distribution<int> distC(start, end - 1);
            v = distC(rng);
            if (v == u)
            {
                v = (v == end - 1 ? start : v + 1);
            }
        }
        else
        {
            // 30%：跨社区边
            v = vdist(rng);
            if (v == u)
            {
                v = (v + 1) % n;
            }
        }
        double w = wdist(rng);
        g.edges.push_back({u, v, w});
    }

    return g;
}

// =========================== MST Algorithms ================================

// 标准 Prim 算法：邻接表 + 最小堆（顺序）
double mst_prim(const Graph &g)
{
    int n = g.n;
    const vector<Edge> &edges = g.edges;

    vector<vector<pair<int, double>>> adj(n);
    for (const auto &e : edges)
    {
        adj[e.u].push_back(std::make_pair(e.v, e.w));
        adj[e.v].push_back(std::make_pair(e.u, e.w));
    }

    const double INF = std::numeric_limits<double>::infinity();
    vector<double> min_w(n, INF);
    vector<char> used(n, 0);

    using P = pair<double, int>; // (weight, vertex)
    priority_queue<P, vector<P>, greater<P>> pq;

    min_w[0] = 0.0;
    pq.push(std::make_pair(0.0, 0));

    double total_weight = 0.0;
    int visited_count = 0;

    while (!pq.empty())
    {
        P top = pq.top();
        pq.pop();
        double w = top.first;
        int v = top.second;
        if (used[v])
            continue;
        used[v] = 1;
        total_weight += w;
        ++visited_count;
        for (size_t i = 0; i < adj[v].size(); ++i)
        {
            int to = adj[v][i].first;
            double wt = adj[v][i].second;
            if (!used[to] && wt < min_w[to])
            {
                min_w[to] = wt;
                pq.push(std::make_pair(wt, to));
            }
        }
    }

    if (visited_count != n)
    {
        cerr << "[Prim] Warning: graph appears disconnected (visited "
             << visited_count << " of " << n << " vertices)\n";
    }
    return total_weight;
}

// 标准 Kruskal 算法（顺序）
double mst_kruskal(const Graph &g)
{
    int n = g.n;
    vector<Edge> edges = g.edges; // 复制一份，排序用

    sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b)
         { return a.w < b.w; });

    DSU dsu(n);
    double total_weight = 0.0;
    int edges_used = 0;

    for (size_t i = 0; i < edges.size(); ++i)
    {
        const Edge &e = edges[i];
        if (dsu.unite(e.u, e.v))
        {
            total_weight += e.w;
            ++edges_used;
            if (edges_used == n - 1)
                break;
        }
    }
    if (edges_used != n - 1)
    {
        cerr << "[Kruskal] Warning: MST uses " << edges_used
             << " edges, expected " << (n - 1) << " (graph may be disconnected)\n";
    }
    return total_weight;
}

// 自适应并行 MST：Borůvka + 边过滤 + Kruskal 收尾，多线程版
double mst_adaptive_parallel(const Graph &g)
{
    int n = g.n;
    if (n == 0)
        return 0.0;

    // 工作边集
    vector<Edge> edges = g.edges;
    DSU dsu(n);
    double total_weight = 0.0;
    int components = n;

    const double INF = std::numeric_limits<double>::infinity();
    const int MAX_BORUVKA_ROUNDS = 32;
    const int COMPONENT_THRESHOLD = std::max(1, n / 16);
    const size_t EDGE_THRESHOLD = static_cast<size_t>(2ull * (unsigned long long)n);

    int round = 0;
    while (round < MAX_BORUVKA_ROUNDS &&
           components > COMPONENT_THRESHOLD &&
           edges.size() > EDGE_THRESHOLD)
    {

        int nThreads = 1;
#ifdef _OPENMP
        nThreads = omp_get_max_threads();
#endif
        if (nThreads < 1)
            nThreads = 1;

        // 每个线程维护一份“本线程看到的每个组件最小边”
        vector<vector<int>> bestEdgeThread(nThreads, vector<int>(n, -1));
        vector<vector<double>> bestWeightThread(nThreads, vector<double>(n, INF));

        // ===== Step 1: 并行扫描边集，为每个组件找到最轻外向边 =====
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            auto &bestIdx = bestEdgeThread[tid];
            auto &bestW = bestWeightThread[tid];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (long long i = 0; i < (long long)edges.size(); ++i)
            {
                const Edge &e = edges[(size_t)i];
                int ru = dsu.find_nocompress(e.u);
                int rv = dsu.find_nocompress(e.v);
                if (ru == rv)
                    continue; // 当前已在同一组件内
                double w = e.w;
                if (w < bestW[ru])
                {
                    bestW[ru] = w;
                    bestIdx[ru] = (int)i;
                }
                if (w < bestW[rv])
                {
                    bestW[rv] = w;
                    bestIdx[rv] = (int)i;
                }
            }
        } // end parallel

        // ===== Step 2: 合并各线程局部结果，得到全局每组件最轻边 =====
        vector<int> bestEdge(n, -1);
        vector<double> bestWeight(n, INF);

        for (int t = 0; t < nThreads; ++t)
        {
            const auto &bestIdx = bestEdgeThread[t];
            const auto &bestW = bestWeightThread[t];
            for (int v = 0; v < n; ++v)
            {
                int idx = bestIdx[v];
                if (idx < 0)
                    continue;
                double w = bestW[v];
                if (w < bestWeight[v])
                {
                    bestWeight[v] = w;
                    bestEdge[v] = idx;
                }
            }
        }

        // ===== Step 3: 串行地把这些边加入 MST，并合并组件 =====
        vector<char> used(edges.size(), 0);
        int added_this_round = 0;
        double max_chosen_weight = 0.0;

        for (int v = 0; v < n; ++v)
        {
            int idx = bestEdge[v];
            if (idx < 0)
                continue;
            if (used[(size_t)idx])
                continue;
            used[(size_t)idx] = 1;

            const Edge &e = edges[(size_t)idx];
            int ru = dsu.find(e.u);
            int rv = dsu.find(e.v);
            if (ru == rv)
                continue;
            if (dsu.unite(ru, rv))
            {
                total_weight += e.w;
                ++added_this_round;
                --components;
                if (e.w > max_chosen_weight)
                    max_chosen_weight = e.w;
            }
        }

        if (added_this_round == 0)
        {
            // 没有进展，退出 Borůvka 阶段
            break;
        }

        // ===== Step 4: 并行边过滤 =====
        // 论文里说要删掉：
        //  (1) 已经变成“组件内部边”的边
        //  (2) 权重显著大于本轮已选边上界的重边（启发式阈值）【不会删除必要 MST 边】:contentReference[oaicite:5]{index=5}
        double thresholdW = std::numeric_limits<double>::infinity();
        if (max_chosen_weight > 0.0)
        {
            // 比较保守的系数，避免误删 MST 必要边
            thresholdW = max_chosen_weight * 4.0;
        }

        int nThreads2 = 1;
#ifdef _OPENMP
        nThreads2 = omp_get_max_threads();
#endif
        if (nThreads2 < 1)
            nThreads2 = 1;
        vector<vector<Edge>> local_edges(nThreads2);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            auto &bucket = local_edges[tid];
            bucket.clear();

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (long long i = 0; i < (long long)edges.size(); ++i)
            {
                const Edge &e = edges[(size_t)i];
                int ru = dsu.find_nocompress(e.u);
                int rv = dsu.find_nocompress(e.v);
                // (1) 删除已经变成组件内部边
                if (ru == rv)
                    continue;
                // (2) 删除明显比当前 MST 上界重的边（启发式）
                if (e.w > thresholdW)
                    continue;
                bucket.push_back(e);
            }
        } // end parallel

        // 合并所有线程保留的边，原地复用 edges 的内存
        size_t new_m = 0;
        for (int t = 0; t < nThreads2; ++t)
        {
            new_m += local_edges[t].size();
        }

        edges.clear();
        edges.reserve(new_m);
        for (int t = 0; t < nThreads2; ++t)
        {
            edges.insert(edges.end(), local_edges[t].begin(), local_edges[t].end());
        }

        ++round;
    } // end while Borůvka 阶段

    // ===== 后期：在缩减后的边集上做 Kruskal，完成 MST =====
    sort(edges.begin(), edges.end(),
         [](const Edge &a, const Edge &b)
         { return a.w < b.w; });

    for (const Edge &e : edges)
    {
        int ru = dsu.find(e.u);
        int rv = dsu.find(e.v);
        if (ru == rv)
            continue;
        if (dsu.unite(ru, rv))
        {
            total_weight += e.w;
        }
    }

    return total_weight;
}

// ============================= Benchmarking ================================
struct Result
{
    double mst_weight;
    double time_ms;
};

Result run_with_timing(std::function<double(const Graph &)> algo, const Graph &g, int repeats)
{
    using Clock = std::chrono::high_resolution_clock;
    double sum_time_ms = 0.0;
    double weight = 0.0;

    for (int i = 0; i < repeats; ++i)
    {
        auto t1 = Clock::now();
        double w = algo(g);
        auto t2 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        sum_time_ms += ms;
        weight = w; // MST 权重应当一致
    }
    Result r;
    r.mst_weight = weight;
    r.time_ms = sum_time_ms / repeats;
    return r;
}

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cout << "Adaptive Parallel MST Algorithm - C++ Evaluation Program\n";
    std::cout << "------------------------------------------------------\n\n";

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    int use_threads = std::min(8, max_threads); // 论文里用的是 8 线程:contentReference[oaicite:6]{index=6}
    if (use_threads < 1)
        use_threads = 1;
    omp_set_num_threads(use_threads);
    std::cout << "OpenMP enabled. Using " << use_threads << " threads for hybrid algorithm.\n\n";
#else
    std::cout << "OpenMP not enabled. Hybrid algorithm will run single-threaded.\n\n";
#endif

    // 论文中对每种图跑 10 次取平均:contentReference[oaicite:7]{index=7}
    const int REPEATS = 10;

    struct GraphSpec
    {
        std::string name;
        int n;
        long long m;
        std::string type;
        bool layered;
    };

    // 和表 1 一致的图规模：S1/S2/D1/L1:contentReference[oaicite:8]{index=8}
    std::vector<GraphSpec> specs = {
        {"S1", 100000, 1000000LL, "Random sparse", false},
        {"S2", 100000, 5000000LL, "Random medium-density", false},
        {"D1", 10000, 50000000LL, "Random dense", false},
        {"L1", 50000, 5000000LL, "Layered / community", true}};

    std::mt19937_64 rng(42); // 固定种子，保证可复现实验

    for (const auto &spec : specs)
    {
        std::cout << "==== Graph " << spec.name << " (" << spec.type << ") ====\n";
        std::cout << "Vertices: " << spec.n << ", Edges: " << spec.m << "\n";

        Graph g;
        if (spec.layered)
        {
            g = generate_layered_graph(spec.n, spec.m, 8, rng);
        }
        else
        {
            g = generate_random_graph(spec.n, spec.m, rng);
        }

        Result prim_res = run_with_timing(mst_prim, g, REPEATS);
        Result kruskal_res = run_with_timing(mst_kruskal, g, REPEATS);
        Result hybrid_res = run_with_timing(mst_adaptive_parallel, g, REPEATS);

        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(3);

        auto printLine = [](const std::string &name, const Result &r)
        {
            double sec = r.time_ms / 1000.0;
            std::cout << name << ":  MST weight = " << r.mst_weight
                      << ", avg time = " << r.time_ms << " ms  ("
                      << sec << " s)\n";
        };

        printLine("Prim   ", prim_res);
        printLine("Kruskal", kruskal_res);
        printLine("Hybrid ", hybrid_res);

        double diff_pk = std::fabs(prim_res.mst_weight - kruskal_res.mst_weight);
        double diff_ph = std::fabs(prim_res.mst_weight - hybrid_res.mst_weight);
        double diff_kh = std::fabs(kruskal_res.mst_weight - hybrid_res.mst_weight);
        double max_diff = std::max(diff_pk, std::max(diff_ph, diff_kh));

        if (max_diff > 1e-6)
        {
            std::cout << "[Warning] MST weights differ between algorithms! "
                      << "Max difference = " << max_diff << "\n";
        }
        else
        {
            std::cout << "MST weights are consistent across all algorithms.\n";
        }

        std::cout << "\n";
    }

#ifdef _OPENMP
    // ================== 额外：对 Hybrid 算法做线程数-加速比测试 ==================
    // 选一个代表性的图（这里选 D1: 10000, 50000000 边）
    GraphSpec speed_spec{"D1", 10000, 50000000LL, "Random dense", false};
    std::cout << "Speedup experiment on graph " << speed_spec.name
              << " (" << speed_spec.type << ") using Hybrid algorithm.\n";

    // 为 speedup 实验单独生成一张图（保证各线程数下图一致）
    Graph g_speed;
    g_speed = generate_random_graph(speed_spec.n, speed_spec.m, rng);

    // 准备线程数列表：1,2,4,8（不超过硬件最大线程数）
    std::vector<int> thread_list = {1, 2, 4, 8};
    int hw_max_threads = omp_get_max_threads();
    std::vector<int> threads;
    for (int t : thread_list)
    {
        if (t <= hw_max_threads)
            threads.push_back(t);
    }

    if (threads.empty())
        threads.push_back(1);

    std::ofstream fout("speedup_d1.csv");
    fout << "threads,time_ms,speedup\n";

    double baseline_ms = -1.0;

    for (int t : threads)
    {
        omp_set_num_threads(t);
        Result r = run_with_timing(mst_adaptive_parallel, g_speed, REPEATS);
        double time_ms = r.time_ms;
        if (baseline_ms < 0.0)
            baseline_ms = time_ms;
        double speedup = baseline_ms / time_ms;

        std::cout << "  Threads = " << t
                  << ", Hybrid avg time = " << std::fixed << std::setprecision(3)
                  << time_ms << " ms"
                  << ", speedup = " << speedup << "x\n";

        fout << t << "," << time_ms << "," << speedup << "\n";
    }
    fout.close();
    std::cout << "Speedup data has been written to speedup_d1.csv\n\n";
#endif

    std::cout << "Done.\n";
    return 0;
}
