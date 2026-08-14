<h2 align="center">DGC-Net: Dual-Granularity Cross-Stage
Calibration for Reactive Lymphocyte
Recognition (PRCV 2026)</h2>

## Overview
<p align='center'>
    <img src="figures/fig.2.png" width="86%" height="86%">
</p>

**Figure 1. Flow diagram of the DGC-Net model.**

**_Abstract -_** Reactive lymphocytes serve as key cytological indicators of immune activation, infection, inflammation, and hematological disorders. Accurate recognition of these cells in peripheral blood smear images is critical for morphology-based screening and auxiliary diagnosis. However, automated identification remains challenging due to substantial intra-class morphological variability and high inter-class similarity with lymphocytes, monocytes, blasts, and other atypical cells, which results in ambiguous decision boundaries. To address this challenge, we propose DGC-Net, a Dual-Granularity Cross-Stage Calibration Network for fine-grained reactive lymphocyte classification. Specifically, we first introduce a Dual-Granularity Representation Encoding (DGRE) module to extract complementary structural- and detail-level representations from single-cell images. We then design a Progressive Stage-Coupled Fusion (PSCF) module to progressively propagate fused features across hierarchical stages, thereby preserving low-level discriminative cues while integrating high-level semantic information. Finally, a Bilateral Gated Semantic Calibration (BGSC) module performs bidirectional feature interaction and gated semantic refinement to adaptively enhance discriminative representations for classification. We evaluate our framework on an in-house six-class blood cell dataset as well as two public blood cell datasets, demonstrating that DGC-Net achieves competitive or improved performance with a macro-F1 score of 88.24±0.95% and an accuracy of 90.58±0.89%. Compared with representative vision models, our method learns more effective representations for distinguishing reactive lymphocytes from morphologically similar cell types.Our code is available at https://github.com/Jin-k123/DGC-Net

## Experimental Results
<p align='center'>
    <img src="figures/fig.3.png" width="86%" height="86%">
</p>

**Figure 1. Experimental results on the six-class blood cell classification task. (a) ROC curves, (b) precision–recall curves, (c) t-SNE visualization, and (d) normalized confusion matrix.**

### Dataset Preparation
Put the dataset as follows:
```text
Dataset
├── 0
│   ├── 1.png
│   ├── 2.png
│   ├── ...
├── 1
│   ├── 1.png
│   ├── 2.png
│   ├── ...
...
```

### Dataset Access
Due to patient privacy protections, access to the **diagnosis dataset** can be requested by contacting the corresponding author with a reasonable research justification. For any questions or collaborations, please contact [Liye Mei](mailto:liyemei@whu.edu.cn), [Ziheng Jin](mailto:jinziheng@hubt.edu.cn).



## Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@inproceedings{jin2026dgcnet,
  title={DGC-Net: Dual-Granularity Cross-Stage Calibration for Reactive Lymphocyte Recognition},
  author={Mei, Liye and Jin, Ziheng and Song, Xiaofang and He, Jing and Ye, Zhiwei and Lei, Cheng and Shen, Hui and Xiong, Bei},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision},
  year={2026},
  organization={Springer Nature}
}
```
