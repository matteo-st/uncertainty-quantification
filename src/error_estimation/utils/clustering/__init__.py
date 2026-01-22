from .my_soft_kmeans import SoftKMeans as TorchSoftKMeans
from .kmeans import KMeans as TorchKMeans
from .decision_tree import TreeQuantizer as TreeQuantizer
from .supervised_partition import SupervisedPartition

quantizers = {
    "soft-kmeans_torch": TorchSoftKMeans,
    "kmeans_torch": TorchKMeans,
    "decision-tree": TreeQuantizer,
    "supervised-partition": SupervisedPartition,
}
