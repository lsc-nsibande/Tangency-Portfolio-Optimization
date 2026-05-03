# GNN Tangency Portfolio — Time-Varying Correlation Graph Extension

> Replication and extension of:
> Liu, B., Li, H., \& Kang, L. (2026). **Tangency portfolios using graph neural networks.** Neural Networks, 193, 108043. (https://www.sciencedirect.com/science/article/pii/S0893608025009232?via%3Dihub)

## Overview
This project extends Tangency Portfolios Using Graph Neural Networks by improving how relationships between stocks are modeled in portfolio construction. The original study replaces traditional mean–variance optimisation with a Graph Neural Network (GNN), where stocks are represented as nodes and connected through a fixed industry-based graph, allowing the model to learn portfolio weights that maximise the Sharpe ratio without explicitly estimating covariance matrices.

In this extension, the key modification is that the adjacency matrix is no longer static. Instead, it is rebuilt at each rolling window using correlations between stock returns, creating a time-varying, data-driven graph. This makes the approach fully reproducible using publicly available price data and allows the model to adapt to changing market conditions.

All model components from the original paper are preserved unchanged:

* GNN embedding module (Equation 4)
* Mean and precision matrix fitting layer (Equations 5–6)
* Long-short portfolio weight prediction (Equations 3, 7)
* Combined loss function: `exp(-SR) + α₁·modu + α₂·ranking` (Equation 16)


## What Changes vs the Original Paper

|Component|Original paper|This extension|
|-|-|-|
|Graph source|ChinaScope supply-demand data|Rolling Pearson correlation / Graphical LASSO|
|Adjacency matrix|Fixed `A` — same every window|Dynamic `A(t)` — rebuilt every 40-day window|
|Edge meaning|Company `i` supplies company `j`|`\|corr(i, j)\|` exceeds threshold|
|Edge direction|Directed (supplier → customer)|Undirected (symmetric correlation)|
|Node features|65 factors (CSI 300) / 161 (A-share)|7 factors from OHLCV only|
|Data required|Proprietary ChinaScope database|Publicly available prices and volumes|
|GNN architectures|GCN, A-DGNs, SSGC, Chebynet, SAGE|GCN, GraphSAGE, GAT|
|Loss function|Equation 16 (unchanged)|Equation 16 (identical)|
|Hyperparameters|lr=5e-3, α₁=1e-3, α₂=1e-1|Identical — paper values preserved|

### GNN Architecture Guide

|Architecture|Best for|Edge weights used?|
|-|-|-|
|`GCN`|Closest to paper baseline; fast|No — degree normalisation only|
|`SAGE`|Robust to noisy graphs; stable|No — mean aggregation|
|`GAT`|**Recommended for this extension**|Yes — learned attention α\_ij per edge|

GAT is recommended because its attention mechanism learns to down-weight spurious correlation edges and amplify structurally stable ones, which is the dominant challenge in rolling correlation graphs.

### Graph Method Guide

|Method|Speed|Sparsity|Notes|
|-|-|-|-|
|`pearson`|Fast (\~0.001s per window)|Controlled by `threshold`|Default; use `threshold=0.35`|
|`glasso`|Slow (\~5s per window)|High — L1 penalty|More principled; use lower `threshold` (\~0.05)|



## Node Features (7 factors)

All features are computed from adjusted close prices and trading volume — no proprietary data required.

|Index|Name|Formula|Economic meaning|
|-|-|-|-|
|F0|`ret\_1d`|`P\_t / P\_{t-1} - 1`|1-day return — short-term momentum|
|F1|`ret\_5d`|`P\_t / P\_{t-5} - 1`|5-day cumulative return — weekly trend|
|F2|`ret\_20d`|`P\_t / P\_{t-20} - 1`|20-day cumulative return — monthly trend|
|F3|`vol\_20d`|`std(r\_{t-20:t})`|20-day realised volatility — risk level|
|F4|`volume\_z`|`zscore(log(V\_t))` cross-sectional|Relative trading activity|
|F5|`mom\_60d`|`P\_{t-5} / P\_{t-60} - 1`|60-day momentum (skip last 5 days)|
|F6|`rsi\_14`|`(RSI(14) / 50) - 1`|14-day RSI rescaled to \[-1, 1]|

The feature tensor has shape `(T, N, 7)` and serves as `H⁰` — the initial node feature matrix fed into the GNN.



## Outputs

After a successful training run, the following files are saved to `/content/`:

|File|Contents|
|-|-|
|`best\_gnn.pt`|Best GNN model weights (by validation Sharpe ratio)|
|`best\_fitter.pt`|Best fitting layer weights|
|`training\_log.csv`|Per-epoch: total loss, Sharpe loss, modularity loss, ranking loss, validation SR|
|`results.json`|Final test metrics: SR, return, volatility, MDD, Calmar, win rate|
|`diagnostics.png`|6-panel figure: loss curves, validation SR, cumulative returns, drawdown, edge density, return distribution|
|`prices.csv`|Cached price and volume data (avoids re-downloading on repeat runs)|


## Expected Results

The table below shows the paper's reported results for CSI 300 (Table 3, Liu et al. 2026) alongside placeholder values for this extension. Replace the extension column with your actual trained results.

|Model|Sharpe (annual)|Return|Volatility|
|-|-|-|-|
|Fama-3 (baseline)|0.11|2.2e-4|0.085|
|OLS (baseline)|0.16|6.6e-4|0.017|
|OAS (baseline)|0.56|9.1e-4|0.033|
|LSTM (baseline)|0.19|7.1e-4|0.074|
|**Ours GCN (paper)**|**0.86**|**3.2e-3**|**0.297**|
|**Ours A-DGNs (paper)**|**1.10**|**4.8e-3**|**0.078**|
|Extension GCN|0.91|0.34|0.37|
|Extension SAGE|0.46|0.16|0.34|
|Extension GAT|0.76|0.29|0.3847|



## System Requirements

|Dataset|GPU VRAM|System RAM|Storage|Colab tier|
|-|-|-|-|-|
|Proxy tickers (N≈50)|8 GB|16 GB|\~2 GB|Free (T4)|
|CSI 300 (N=94)|8 GB|16 GB|\~10 GB|Pro (G4)|

For the A-share dataset, set `rank=128` in `MeanPrecisionFitter` to use a low-rank approximation of the precision matrix and reduce VRAM usage.


## Paper Figures

`paper\_figures.py` generates 7 figures at 300 DPI:

|Figure|Description|
|-|-|
|`fig1\_model\_architecture.png`|Full pipeline from OHLCV to loss function|
|`fig2\_rolling\_window.png`|Rolling window construction: time series, correlation matrix, graph|
|`fig3\_gnn\_comparison.png`|GCN vs GraphSAGE vs GAT message-passing comparison|
|`fig4\_adjacency\_matrix.png`|Static industry chain vs dynamic correlation adjacency|
|`fig5\_loss\_diagram.png`|Equation 16 decomposed into its three components|
|`fig6\_results\_comparison.png`|Sharpe ratio bar chart and return vs volatility scatter|
|`fig7\_feature\_diagram.png`|All 7 features with formulas and economic meanings|

(Note:still to be incorporated!)




## References

* Liu, B., Li, H., \& Kang, L. (2026). Tangency portfolios using graph neural networks. *Neural Networks*, 193, 108043.




