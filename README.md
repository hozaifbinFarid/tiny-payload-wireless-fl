Tiny-Payload Wireless Federated Learning



What this is

Random sparsification reduces the communication cost of federated learning, but it also makes the length of each client's update depend on the sampled gradient. Under a fixed transmission deadline, longer updates are less likely to arrive, so successful delivery becomes informative about the update itself and simply dropping late updates biases the aggregate. This repository contains a single, resumable Google Colab notebook that simulates orthogonal wireless uplinks with such random-length payloads and compares six aggregation rules: uncorrected dropped updates, an oracle channel correction at the mean count, a past-delivery EMA heuristic, an oracle length-conditioned inverse-probability-weighted (IPW) correction, a probe-estimated IPW correction, and a fixed-size importance-sampling baseline. It runs three studies (UCI HAR with subject clients, UCI HAR with Dirichlet clients, and PAMAP2 with subject clients) with 12 seeds each. The notebook also produces the bias diagnostics, learning curves, matched-cost comparisons, held-out-subject test metrics, paired comparisons, and figures reported in the paper. It makes no claim about a deployed 6G system, differential privacy, or a faithful reproduction of JCDO or CA-Fed.
