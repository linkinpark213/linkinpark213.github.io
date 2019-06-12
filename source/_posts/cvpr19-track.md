---
title: A Summary of CVPR19 Visual Tracking Papers
tags:
  - Deep Learning
  - Visual Tracking
date: 2019-06-11 17:27:10
---

<style type="text/css" rel="stylesheet">
.markdown-body p {
    text-indent: 0
}   
</style>

Here's my brief summary of all CVPR19 papers in the field of visual tracking. Abbreviations without parentheses are part of the paper title, and those with parentheses are added by me according to the paper.

<!-- more -->
# RGB-based
## Single-Object Tracking
__(UDT): Unsupervised Deep Tracking__
__Authors__: Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, Houqiang Li
__arXiv Link__: https://arxiv.org/abs/1904.01828
__Project Link__: https://github.com/594422814/UDT
__Summary__: Train a robust siamese network on large-scale unlabeled videos in an unsupervised manner - forward-and-backward, i.e., the tracker can forward localize the target object in successive frames and backtrace to its initial position in the first frame.
__Highlights__: Unsupervised learning


__(TADT): Target-Aware Deep Tracking__
__Authors__: Xin Li, Chao Ma, Baoyuan Wu, Zhenyu He, Ming-Hsuan Yang
__arXiv Link__: https://arxiv.org/abs/1904.01772
__Project Link__: https://xinli-zn.github.io/TADT-project-page/
__Summary__: Targets of interest can be arbitrary object class with arbitrary forms, while pre-trained deep features are less effective in modeling these targets of arbitrary forms for distinguishing them from the background. TADT learns target-aware features, thus can better recognize the targets undergoing significant appearance variations than pre-trained deep features.
__Highlights__: Target-aware features, better discrimination


__(SiamMask): Fast Online Object Tracking and Segmentation: A Unifying Approach__
__Authors__: Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr
__arXiv Link__: https://arxiv.org/abs/1812.05050
__Project Link__: https://github.com/foolwood/SiamMask
Zhihu Link: https://zhuanlan.zhihu.com/p/58154634
__Summary__: Perform both visual object tracking and semi-supervised video object segmentation, in real-time, with a single simple approach.
__Highlights__: Mask prediction in tracking


__SiamRPN++: Evolution of Siamese Visual Tracking With Very Deep Networks__
__Authors__: Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan
__arXiv Link__: https://arxiv.org/abs/1812.11703
__Project Link__: http://bo-li.info/SiamRPN++/
__Summary__: SiamRPN++ breaks the translation invariance restriction through a simple yet effective spatial-aware sampling strategy. SiamRPN++ performs depth-wise and layer-wise aggregations, improving the accuracy but also reduces the model size. Current state-of-the-art in OTB2015, VOT2018, UAV123, LaSOT, and TrackingNet.
__Highlights__: Deep backbones, state-of-the-art


__(CIR/SiamDW): Deeper and Wider Siamese Networks for Real-Time Visual Tracking__
__Authors__: Zhipeng Zhang, Houwen Peng
__arXiv Link__: https://arxiv.org/abs/1901.01660
__Project Link__: https://github.com/researchmm/SiamDW
__Summary__: SiamDW explores utilizing deeper and wider network backbones in another aspect - careful designs of residual units, considering receptive field, stride, output feature size - to eliminate the negative impact of padding in deep network backbones.
__Highlights__: Cropping-Inside-Residual, eliminating the negative impact of padding


__(SiamC-RPN): Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking__
__Authors__: Heng Fan, Haibin Ling
__arXiv Link__: https://arxiv.org/abs/1812.06148
__Project Link__: None
__Summary__: Previously proposed one-stage Siamese-RPN trackers degenerate in presence of similar distractors and large scale variation. Advantages: 1) Each RPN in Siamese C-RPN is trained using outputs of the previous RPN, thus simulating hard negative sampling. 2) Feature transfer blocks (FTB) further improving the discriminability. 3) The location and shape of the target in each RPN is progressively refined, resulting in better localization.
__Highlights__: Cascaded RPN, excellent accuracy


__SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking__
__Authors__: Guangting Wang, Chong Luo, Zhiwei Xiong, Wenjun Zeng
__arXiv Link__: https://arxiv.org/abs/1904.04452
__Project Link__: None
__Summary__: To overcome the simultaneous requirements on robustness and discrimination power, SPM-Tracker tackle the challenge by connecting a coarse matching stage and a fine matching stage, taking advantage of both stages, resulting in superior performance, and exceeding other real-time trackers by a notable margin.
__Highlights__: Coarse matching & fine matching


__ATOM: Accurate Tracking by Overlap Maximization__
__Authors__: Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg
__arXiv Link__: https://arxiv.org/abs/1811.07628
__Project Link__: https://github.com/visionml/pytracking
__Summary__: Target estimation is a complex task, requiring highlevel knowledge about the object, while most trackers only resort to a simple multi-scale search. In comparison, ATOM estimate target states by predicting the overlap between the target object and an estimated bounding box. Besides, a classification component that is trained online to guarantee high discriminative power in the presence of distractors.
__Highlights__: Overlap IoU prediction


__(GCT): Graph Convolutional Tracking__
__Authors__: Junyu Gao, Tianzhu Zhang, Changsheng Xu
__arXiv Link__: None
__PDF Link__: http://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Graph_Convolutional_Tracking_CVPR_2019_paper.pdf
__Project Link__: http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html
__Summary__: Spatial-temporal information can provide diverse features to enhance the target representation. GCT incorporates 1) a spatial-temporal GCN to model the structured representation of historical target exemplars, and 2) a context GCN to utilize the context of the current frame to learn adaptive features for target localization.
__Highlights__: Graph convolution networks, spatial-temporal information


__(ASRCF): Visual Tracking via Adaptive Spatially-Regularized Correlation Filters__
__Authors__: Kenan Dai, Dong Wang, Huchuan Lu, Chong Sun, Jianhua Li
__arXiv Link__: None
__Project Link__: https://github.com/Daikenan/ASRCF (To be updated)
__Summary__: ASRCF simultaneously optimize the filter coefficients and the spatial regularization weight. ASRCF applies two correlation filters (CFs) to estimate the location and scale respectively - 1) location CF model, which exploits ensembles of shallow and deep features to determine the optimal position accurately, and 2) scale CF model, which works on multi-scale shallow features to estimate the optimal scale efficiently.
__Highlights__: Estimate location and scale respectively


__(RPCF): RoI Pooled Correlation Filters for Visual Tracking__
__Authors__: Yuxuan Sun, Chong Sun, Dong Wang, You He, Huchuan Lu
__arXiv Link__: None
__Project Link__: None
PDF Link: http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_ROI_Pooled_Correlation_Filters_for_Visual_Tracking_CVPR_2019_paper.pdf
__Summary__: RoI-based pooling can be equivalently achieved by enforcing additional constraints on the learned filter weights and thus becomes feasible on the virtual circular samples. Considering RoI pooling in the correlation filter formula, the RPCF performs favourably against other state-of-the-art trackers.
__Highlights__: RoI pooling in correlation filters


## Multi-Object Tracking
__(TBA): Tracking by Animation: Unsupervised Learning of Multi-Object Attentive Trackers__
__Authors__: Zhen He, Jian Li, Daxue Liu, Hangen He, David Barber
__arXiv Link__: https://arxiv.org/abs/1809.03137
__Project Link__: https://github.com/zhen-he/tracking-by-animation
__Summary__: The common Tracking-by-Detection (TBD) paradigm use supervised learning and treat detection and tracking separately. Instead, TBA is a differentiable neural model that first tracks objects from input frames, animates these objects into reconstructed frames, and learns by the reconstruction error through backpropagation. Besides, a Reprioritized Attentive Tracking is proposed to improve the robustness of data association.
__Highlights__: Label-free, end-to-end MOT learning


__Eliminating Exposure Bias and Metric Mismatch in Multiple Object Tracking__
__Authors__: Andrii Maksai, Pascal Fua
__arXiv Link__: https://arxiv.org/abs/1811.10984
__Project Link__: None
__Summary__: Many state-of-the-art MOT approaches now use sequence models to solve identity switches but their training can be affected by biases. An iterative scheme of building a rich training set is proposed and used to learn a scoring function that is an explicit proxy for the target tracking metric.
__Highlights__: Eliminating loss-evaluation mismatch


## Pose Tracking
__Multi-Person Articulated Tracking With Spatial and Temporal Embeddings__
__Authors__: Sheng Jin, Wentao Liu, Wanli Ouyang, Chen Qian
__arXiv Link__: https://arxiv.org/abs/1903.09214
__Project Link__: None
__Summary__: The framework consists of a SpatialNet and a TemporalNet, predicting (body part detection heatmaps + Keypoint Embedding (KE) + Spatial Instance Embedding (SIE)) and (Human Embedding (HE) + Temporal Instance Embedding (TIE)). Besides, a differentiable Pose-Guided Grouping (PGG) module to make the whole part detection and grouping pipeline fully end-to-end trainable.
__Highlights__: Spatial & temporal embeddings, end-to-end learning "detection and grouping" pipeline


__(STAF): Efficient Online Multi-Person 2D Pose Tracking With Recurrent Spatio-Temporal Affinity Fields__
__Authors__: Yaadhav Raaj, Haroon Idrees, Gines Hidalgo, Yaser Sheikh
__arXiv Link__: https://arxiv.org/abs/1811.11975
__Project Link__: None
__Summary__: Upon Part Affinity Field (PAF) representation designed for static images, an architecture encoding ans predicting Spatio-Temporal Affinity Fields (STAF) across a video sequence is proposed - a novel temporal topology cross-linked across limbs which can consistently handle body motions of a wide range of magnitudes. The network ingests STAF heatmaps from previous frames and estimates those for the current frame.
__Highlights__: Online, fastest and the most accurate bottom-up approach


# RGBD-based
__(OTR)__: Object Tracking by Reconstruction With View-Specific Discriminative Correlation Filters
__Authors__: Ugur Kart, Alan Lukezic, Matej Kristan, Joni-Kristian Kamarainen, Jiri Matas
__arXiv Link__: https://arxiv.org/abs/1811.10863
__Summary__: Perform online 3D target reconstruction to facilitate robust learning of a set of view-specific discriminative correlation filters (DCFs). State-of-the-art on Princeton RGB-D tracking and STC Benchmarks.
__Highlights__:


# Pointcloud-based
I'm not experienced in point clouds so I couldn't make a summary for the following papers. The abstracts are given below. Check them out at arXiv to learn more if you're interested.

__VITAMIN-E__: VIsual Tracking and MappINg With Extremely Dense Feature Points
__Authors__: Masashi Yokozuka, Shuji Oishi, Simon Thompson, Atsuhiko Banno
__arXiv Link__: https://arxiv.org/abs/1904.10324
__Project Link__: None
__Abstract__: In this paper, we propose a novel indirect monocular SLAM algorithm called "VITAMIN-E," which is highly accurate and robust as a result of tracking extremely dense feature points. Typical indirect methods have difficulty in reconstructing dense geometry because of their careful feature point selection for accurate matching. Unlike conventional methods, the proposed method processes an enormous number of feature points by tracking the local extrema of curvature informed by dominant flow estimation. Because this may lead to high computational cost during bundle adjustment, we propose a novel optimization technique, the "subspace Gauss--Newton method", that significantly improves the computational efficiency of bundle adjustment by partially updating the variables. We concurrently generate meshes from the reconstructed points and merge them for an entire 3D model. The experimental results on the SLAM benchmark dataset EuRoC demonstrated that the proposed method outperformed state-of-the-art SLAM methods, such as DSO, ORB-SLAM, and LSD-SLAM, both in terms of accuracy and robustness in trajectory estimation. The proposed method simultaneously generated significantly detailed 3D geometry from the dense feature points in real time using only a CPU.


__Leveraging Shape Completion for 3D Siamese Tracking__
__Authors__: Silvio Giancola*, Jesus Zarzar*, and Bernard Ghanem
__arXiv Link__: https://arxiv.org/abs/1903.01784
__Project Link__: https://github.com/SilvioGiancola/ShapeCompletion3DTracking
__Abstract__: Point clouds are challenging to process due to their sparsity, therefore autonomous vehicles rely more on appearance attributes than pure geometric features. However, 3D LIDAR perception can provide crucial information for urban navigation in challenging light or weather conditions. In this paper, we investigate the versatility of Shape Completion for 3D Object Tracking in LIDAR point clouds. We design a Siamese tracker that encodes model and candidate shapes into a compact latent representation. We regularize the encoding by enforcing the latent representation to decode into an object model shape. We observe that 3D object tracking and 3D shape completion complement each other. Learning a more meaningful latent representation shows better discriminatory capabilities, leading to improved tracking performance. We test our method on the KITTI Tracking set using car 3D bounding boxes. Our model reaches a 76.94% Success rate and 81.38% Precision for 3D Object Tracking, with the shape completion regularization leading to an improvement of 3% in both metrics.


# Datasets
__LaSOT__: A High-Quality Benchmark for Large-Scale Single Object Tracking
__Authors__: Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling
__arXiv Link__: https://arxiv.org/abs/1809.07845
__Project Link__: https://cis.temple.edu/lasot/
__Summary__: A high-quality benchmark for __La__rge-scale __S__ingle __O__bject __T__racking, consisting of 1,400 sequences with more than 3.5M frames.


__CityFlow__: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification
Zheng Tang, Milind Naphade, Ming-Yu Liu, Xiaodong Yang, Stan Birchfield, Shuo Wang, Ratnesh Kumar, David Anastasiu, Jenq-Neng Hwang
__arXiv Link__: https://arxiv.org/abs/1903.09254
__Project Link__: https://www.aicitychallenge.org/
__Summary__: The largest-scale dataset in terms of spatial coverage and the number of cameras/videos in an urban environment,  consisting of more than 3 hours of synchronized HD videos from 40 cameras across 10 intersections, with the longest distance between two simultaneous cameras being 2.5 km.


__MOTS__: Multi-Object Tracking and Segmentation
__Authors__: Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, Bastian Leibe
__arXiv Link__: https://arxiv.org/abs/1902.03604
__Project Link__: https://www.vision.rwth-aachen.de/page/mots
__Summary__: Going beyond 2D bounding boxes and extending the popular task of multi-object tracking to multi-object tracking and segmentation, in tasks and metrics.
__Highlights__: Extend MOT with segmentation


__Argoverse__: 3D Tracking and Forecasting With Rich Maps
Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, James Hays
__arXiv Link__: None
__PDF Link__: http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf
__Project Link__: Argoverse.org (Not working?)
__Summary__: A dataset designed to support autonomous vehicle perception tasks including 3D tracking and motion forecasting.
