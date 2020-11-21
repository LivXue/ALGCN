<h1 align = "center">Adaptive Label-aware Graph Convolutional Networks for Cross-Modal Retrieval</h1>

<h1 align = "center">Abstract</h1> 

<p style="text-align:justify">The cross-modal retrieval task has raised continuous attention in recent years with the increasing scale 
of multi-modal data, which has broad application prospects including multimedia data management and intelligent search 
engine. Most existing methods mainly project data of different modalities into a common representation space where label
 information is often exploited to distinguish samples from different semantic categories. However, they typically treat 
 each label as an independent individual and ignore the underlying semantic structure of labels. In this paper, we 
 propose an end-to-end adaptive label-aware graph convolutional network (ALGCN) by designing both the instance 
 representation learning branch and the label representation learning branch, which can obtain modality-invariant and 
 discriminative representations for cross-modal retrieval. Firstly, we construct an instance representation learning 
 branch to transform instances of different modalities into a common representation space. Secondly, we adopt Graph 
 Convolutional Network (GCN) to learn inter-dependent classifiers in the label representation learning branch. In 
 addition, a novel adaptive correlation matrix is proposed to efficiently explore and preserve the semantic structure 
 of labels in a data-driven manner. Comprehensive experimental results on three benchmark datasets, 
 NUS-WIDE, MIRFlickr and MS-COCO, demonstrate the superiority of ALGCN, compared with the state-of-the-art methods in 
 cross-modal retrieval.</p>

<h1 align = "center">Motivation</h1>

<p style="text-align:justify">Despite that deep learning methods have shown their great success on cross-modal retrieval tasks, we have following observations: </p>

<p style="text-align:justify">O1: The existing supervised methods treat each label as an independent individual in data structure level, while ignoring the label correlations of the cross-modal data.</p>

<p style="text-align:justify">O2: Most of existing supervised cross-modal retrieval methods merely explicitly utilize labels in the loss calculation step.</p>

<p style="text-align:justify">Motivated by the above observations, we propose a novel end-to-end adaptive label-aware graph convolutional network (ALGCN) to capture the label correlations for cross-modal retrieval, which can absorb information from the label graph to guide the model learning discriminative features. </p>

<h1 align = "center">Overview</h1>

<div align="center">
    <img src="images/framework.jpg" width="100%" height ="100%" alt="ALGCN.jpg" />
</div>
<p align = 'center'>
<small> Figure 1. The overall structure of our ALGCN. </small>
</p>

<h1 align = "center">Experiments</h1>

## Cross-modal Retrieval

<p align = 'center'>
<small> Table 1. The mAP result comparisons on NUS-WIDE and MIRFlikcr datasets. </small>
</p>

<div align="center">
    <img src="images/results.jpg" width="100%" height ="100%" alt="results.jpg" />
</div>


## Visualization

<p style="text-align:justify">To illustrate the effectiveness of the label representation learning branch, a recognized visualization tool t-SNE is utilized. Compared with the method which abandons GCN, the distribution of the learned classifiers in a same cluster is more concentrated. Meanwhile, different clusters are more separated.</p>

<div align="center">
    <img src="images/tsne_gcn.png" width="70%" height ="70%" alt="tsne_gcn.png" />
</div>
<p align = 'center'>
<div align="center">
    <img src="images/tsne.png" width="70%" height ="70%" alt="tsne.png" />
</div>
<p align = 'center'>
<small> Figure 4. t-SNE visualization for the learned classifiers on the MIRFlickr dataset: with GCN (above) and without GCN (below). Node colors denote clusters. </small>
</p>


<h1 align = "center">Conclusion</h1>

<p style="text-align:justify">In this paper, we propose an end-to-end Adaptive Label-aware Graph Convolutional Networks (ALGCN) to 
learn the common representations for cross-modal retrieval. In ALGCN, for better preserving the underlying semantic 
structure and bridging the modality gap, we introduce a novel joint-training framework constructed by an instance 
representation learning branch and a label representation learning branch, which is able to preserve the cross-modal 
semantic correlations among instances and mine the latent semantic structure of labels. In addition, an adaptive 
correlation matrix is proposed to dynamically exploit the rich semantic relations between labels in a data-driven 
manner, which can bring more generality to adapt to cross-modal data. Extensive experiments on three widely-used benchmark
 datasets have demonstrated that ALGCN can achieve superior cross-modal retrieval performance compared to 
 state-of-the-art methods. </p>

