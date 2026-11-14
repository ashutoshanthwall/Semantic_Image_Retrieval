#ifndef HNSW_H
#define HNSW_H

#include <string>

#include "common.h"
#include "data.h"
#include "search.h"
class HNSWIndex : public RetrievalIndex
{
public:
    HNSWIndex();

    ~HNSWIndex();

    HNSWIndex(const HNSWIndex &) = delete;
    HNSWIndex &operator=(const HNSWIndex &) = delete;

    void setup(const EmbeddingStore &store, const HnswParams &params);

    void insert(int id, const float *embedding);

    void search(const float *query, int queryDim, int k, CandidateArray &out) const;
    int size() const;
    long long memoryBytes() const;

    void setEfSearch(int ef);
    const HnswParams &params() const;
    const EmbeddingStore &store() const;

    int entryPoint() const;
    int maxLevel() const;
    bool contains(int id) const;
    int levelOf(int id) const;

    const IntArray &neighbors(int id, int layer) const;
    int inDegreeOf(int id, int layer) const;
    int maxDegree(int layer) const;

    long long distanceComputations() const;
    void resetDistanceComputations() const;

    void getRandomState(uint64_t *words, int &position) const;

    void restoreNode(int id, int level);
    IntArray &restoreLinks(int id, int layer);

    void finishRestore(int entry, int topLevel, const uint64_t *rngWords, int rngPosition);

private:
    struct Node
    {
        int level;
        IntArray *links;
        int *inDegree;
    };

    float dist(const float *q, int id) const;
    int randomLevel();
    Neighbor greedyClosest(const float *q, Neighbor entry, int layer) const;

    void searchLayer(const float *q, const NeighborArray &entryPoints, int ef, int layer,
                     NeighborArray &out) const;

    void selectNeighbors(const NeighborArray &candidates, int m, NeighborArray &out) const;

    void shrinkNeighbors(int id, int layer);
    void nextVisitEpoch() const;
    void freeGraph();

    const EmbeddingStore *vectors;
    const float *firstRow;
    int dims;
    HnswParams settings;
    double mL;
    Random64 rng;
    Node *nodes;
    int numNodes;
    BitArray present;
    int count;
    int entry;
    int topLevel;

    mutable unsigned int *visited;
    mutable unsigned int epoch;
    mutable long long distCount;
};

struct LatencyStats
{
    double medianMs = 0, p95Ms = 0, meanMs = 0, minMs = 0, maxMs = 0;
    int samples = 0;
};

LatencyStats summarize(double *values, int n);

void timeQueries(const RetrievalIndex &index, const EmbeddingStore &queries, int k, int warmup,
                 int repetitions, double *perQueryMs);

double recallAtK(const CandidateArray &approx, const CandidateArray &exact, int k);

bool hitAtK(const CandidateArray &results, int target, int k);

struct SystemInfo
{
    std::string cpuModel;
    int logicalCpus = 0;
    int physicalCores = 0;
    long long ramBytes = 0;
    std::string os, kernel, compiler, buildType, cxxFlags, gitCommit, cpuGovernor;
    std::string cpuAffinity;
    std::string loadAverage;
    std::string timestampUtc;
};

SystemInfo collectSystemInfo(const std::string &repoRoot);
long long peakRssBytes();
long long currentRssBytes();

#endif
