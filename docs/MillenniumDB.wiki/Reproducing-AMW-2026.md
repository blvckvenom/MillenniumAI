# Reproducing the ogbn-papers100M results of the AMW 2026 paper

This guide explains, step by step, how to obtain again the ogbn-papers100M figures of the paper *Native Graph Neural
Network Training in MillenniumDB* (AMW 2026). It gives the order of the steps, the commands, the exact text of every
query, and the results to expect.

## 1. What is reproduced

The paper trains a GraphSAGE model inside MillenniumDB on ogbn-papers100M, a citation graph between scientific papers
with 111,059,956 nodes and 1,615,685,872 edges. The task is to predict the subject area of each paper among 172
classes.

The work has four stages, each one a query to the server:

1. **Graph projection** (`graph_project`). Copies the graph into an on-disk structure built for fast traversal,
   together with the node features, the labels and the train, validation and test split.
2. **Sampling** (`gnn_offline_sample`). For every batch of 1,024 seed nodes it picks, hop by hop, a fixed number of
   random neighbours, and stores the samples on disk.
3. **Feature store** (`gnn_build_feature_store`). Places the features of the sampled nodes in GPU memory, in RAM or on
   disk, according to how often each node appears in the samples.
4. **Training** (`gnn_train`). Trains the model for 50 epochs and measures the accuracy on the test set.

## 2. The exact code of the paper

The paper's binary was built from commit `1ff6b6602bbd48faa1057c222ff68a55b4390ef7` of 1 July 2026. That code is
published as the **amw2026** release of this repository, in the file `MillenniumDB-amw2026.tar.gz`. The archive holds
the complete source of that commit, except internal working notes that take no part in the build.

The `main` branch carries later code, with fixes from August and September. To obtain the paper's figures as they were
published, use the code of the release.

## 3. Hardware used

- NVIDIA RTX 5070 Ti GPU with 16 GB of memory.
- 30 GiB of RAM and 20 CPU cores.
- NVMe disk.
- On different hardware the times change, but the expected accuracy does not.

## 4. Build

```bash
tar xzf MillenniumDB-amw2026.tar.gz
cd MillenniumDB-amw2026
# Four dependencies are not in the archive and go inside third_party/, as a directory or a link:
#   third_party/boost_1_82   Boost 1.82
#   third_party/libtorch     LibTorch, the C++ distribution of PyTorch
#   third_party/libnpy       libnpy, header-only reader of .npy files
#   third_party/ska_sort     ska_sort, header-only radix sort
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release -D ENABLE_GNN=ON
cmake --build build/Release -j "$(nproc)"
build/Release/bin/mdb help
```

`third_party/DEPENDENCIES.md`, inside the archive, gives where to download each of the four and how to place it.
`ENABLE_GNN=ON` builds the graph neural network training. CMake finds CUDA by itself and builds the GPU part. The
general setup is in `docs/MillenniumDB.wiki/Setup.md`.

## 5. Prepare the database

1. **Download the dataset** from the Open Graph Benchmark (ogbn-papers100M, about 50 GB compressed), with
   `python scripts/gnn_datasets/download_ogb.py` or directly from the OGB site.
2. **Convert it to the MillenniumDB format.** The converter reads the OGB archive in chunks, so its memory stays around
   500 MB:

   ```bash
   python scripts/gnn_datasets/stream_convert_ogb.py \
       --raw-dir ogb_data/ogbn_papers100M/raw/ \
       --split-dir ogb_data/ogbn_papers100M/split/time/ \
       --output data/example/gql/ogbn-papers100M/ \
       --dataset-name papers100M \
       --directed
   ```

   It writes the graph as text (`papers100M.gql`), the feature matrix (`papers100M_features.npy`, 128 values per node)
   and the labels.
3. **Import it.** The database must end up with the feature property and with the `label` and `split` properties,
   which carry the official OGB split:

   ```bash
   build/Release/bin/mdb import data.gql data/dbs/gql/papers100M --with-tensors node_features.npy
   ```

   This is the command of `REPRODUCE.md`, with the file names that document uses. The queries below call the feature
   property `node_features`.

The detailed procedure, with its caveats, is in `docs/research/2026-07-01-papers100m-repro/REPRODUCE.md` inside the
release (in Spanish). The October 2026 rerun that confirms these figures (section 8) used a database imported
earlier: the download, the conversion and the import are described here from the code of the release, without having
been run again.

## 6. Start the server and send queries

```bash
build/Release/bin/mdb server data/dbs/gql/papers100M -p 7952 -t 14400
```

`-p` is the port and `-t` the longest time a query may run, in seconds; four hours cover the longest stage. Each query
is sent over HTTP as plain text:

```bash
curl -s --fail-with-body -X POST "http://localhost:7952/gql" \
     -H "Content-Type: text/plain" --data-binary "CALL ... RETURN *"
```

With `--fail-with-body`, a server error makes the command exit with a non-zero code and still prints the message.

## 7. The queries, in order

These are the four calls of the procedure, taken from the query log of the October 2026 rerun with the paper's
binary, with their complete `YIELD` lists. They are laid out over several lines to be readable; line breaks and
indentation do not change a query, so each block can be sent as it is. Send each one after the previous one has
finished. In the calls, `'Node'` is the node label, `'CITES'` the edge label and `'node_features'` the name of the
feature property the import created.

**Step 1. Graph projection.** Builds the projection `brazo_proyeccion` with the minimal set of indexes the training
reads (`GNN_MINIMAL`), the features, and the label and split properties.

```sql
CALL graph_project(
    'brazo_proyeccion',
    'Node',
    'CITES',
    {
        orientation: 'NATURAL',
        indexSet: 'GNN_MINIMAL',
        buildTopologySnapshot: false,
        includeFeatures: 'node_features',
        labelProperty: 'label',
        splitProperty: 'split'
    }
)
YIELD
    graphName, nodeCount, relationshipCount, projectMillis, featureDim, numClasses,
    topologySnapshotBytes
RETURN *
```

**Step 2. Sampling.** Batches of 1,024 seeds with the official split, the graph treated as undirected (`UNDIRECTED`:
each paper is linked both to the papers it cites and to the papers that cite it), and random seed 42. The sampler runs
on the GPU over an undirected copy of the graph's topology; on a fresh projection, this call first writes that copy to
disk (about 8 minutes on the machine above) and then samples.

In this commit the list `[20,15,10]` is read from the seed outwards: 20 neighbours in the first hop, 15 in the second
and 10 in the third. In the order used by DGL and DiskGNN, the same configuration is written `[10,15,20]`. The code on
`main` uses the DGL order since 6 July 2026.

```sql
CALL gnn_offline_sample(
    'brazo_proyeccion',
    'brazo_muestra',
    [20,15,10],
    {
        randomSeed: 42,
        batchSize: 1024,
        orientation: 'UNDIRECTED',
        usePredefinedSplits: true,
        useFourLevelTopologyStore: true,
        useAdjacencyCache: true,
        useL3MmapSidecar: true,
        l1CacheMb: 512,
        l2CacheMb: 6144,
        numWorkers: 16,
        force: true
    }
)
YIELD
    sampleName, projectionName, totalBatches, trainBatches, validationBatches, testBatches,
    uniqueNodes, storagePath, computeMillis, phase0Triggered, phase0Succeeded, phase0WalksDone,
    phase0LookupsDone, phase0Millis, numWorkersUsed, sampleContentFp, samplingBackend,
    samplingDirections, samplingPlanReason, symmetricUsed, symmetricBuiltOk, symmetricMs,
    symmetricRamBytes, directionalRamFreedBytes, rssPeakBeforeFreeBytes, rssPeakAfterFreeBytes,
    rssCurrentEndBytes
RETURN *
```

**Step 3. Feature store.** Places the features according to the memory it finds (`autoCache`) and prepares every batch
on disk, ready to be read (`bakeBlocks`).

```sql
CALL gnn_build_feature_store(
    'brazo_muestra',
    'node_features',
    {
        autoCache: true,
        force: true,
        bakeBlocks: true
    }
)
YIELD
    sampleName, featureName, l1Nodes, l2Nodes, l3Nodes, l4Nodes, gpuAvailable, buildTimeMs, slimMb,
    reorderedMb, gpuCacheMb, cpuCacheMb, totalDiskMb, overBudget, addrTablesMb, addrTablesBuiltOk,
    blocksMb, blocksBuiltOk, packedFullMb
RETURN *
```

**Step 4. Training.** GraphSAGE with a hidden layer of 256, 50 epochs, learning rate 0.001, dropout 0.2, seed 42, and
batches prepared ahead of time by 8 threads.

```sql
CALL gnn_train(
    'brazo_muestra',
    'node_features',
    {
        model: 'graphsage',
        hiddenDim: 256,
        epochs: 50,
        lr: 0.001,
        dropout: 0.2,
        normalize: false,
        patience: 999,
        tolerance: 0,
        randomSeed: 42,
        useAsyncPrefetcher: true,
        prefetchNumWorkers: 8,
        prefetchQueueSize: 6,
        sampleCacheMb: 2048,
        saveOnBestVal: true,
        saveFinal: true,
        exportEmbeddings: false,
        trackTestAtBestVal: true,
        outputDir: 'brazo_t50'
    }
)
YIELD
    modelName, ranEpochs, didConverge, bestValAccuracy, testAccuracy, testAccuracyAtBestVal,
    bestValEpoch, trainSeconds, assembleSeconds, forwardSeconds, backwardSeconds, l1HitRatio,
    l2HitRatio, l3Reads, l4Reads, l3BytesDisk, l4BytesDisk, totalBytesDisk, l3ReadAmplification,
    nodesWritten, nodesInferred, inferenceMillis, writeMillis, bestCheckpointPath,
    finalCheckpointPath, resumedFromEpoch, effectivePrefetchWorkers, peakVramMb,
    recommendedL1CacheMb, useAddrTablesEffective, addrTableLoadUs
RETURN *
```

## 8. Results to expect

| measure | paper | October 2026 rerun, same binary |
|---|---|---|
| projection nodes and edges | 111,059,956 and 1,615,685,872 | same |
| batches (train, validation, test) | 1,512 (1,179, 123, 210) | same |
| projection | 44.5 min | 36.1 min |
| sampling | 13.3 min | 16.2 min |
| feature store | 37.9 min | 35.2 min |
| training, 50 epochs | 41.2 min | 40.2 min |
| best validation accuracy | 0.6907 ± 0.0011 | 0.6895 |
| test accuracy at that epoch | 0.6579 ± 0.0009 | 0.6580 |

The paper's figures are means over several runs. The October figures come from a single run of the whole chain on one
server.

The accuracy repeats within the paper's variation. The times move by a few minutes per stage, because they depend on
the state of the machine. Both columns were measured with the undirected copy of the topology already on disk, so the
first sampling run on a fresh projection takes those extra minutes once.

"Test accuracy at that epoch" is measured at the epoch with the best validation accuracy (`testAccuracyAtBestVal`). It
is the figure the paper reports.
