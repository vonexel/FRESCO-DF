<p align="center">
<h1 align="center"><strong>Forensic Response-Curve Evaluation of Spline-Head Classifiers On Faces</strong></h1>
</p>

## Overview: KAN and MLP heads under forensic interventions

FRESCO compares spline-based Kolmogorov–Arnold network (KAN) heads with MLPs on analytical regression and deepfake detection, and audits whether their explanations track predictions under actual image degradation.

## What was measured

For FaceForensics++ (FF++, c23), 3,600 videos were used for training, 700 for validation, and 840 for testing, including 140 FaceShifter videos held out as an unseen manipulation. The external Celeb-DF++ official test list contained 5,418 videos (178 real, 5,240 fake). The experiment used 32 requested frames per video, face crops, a frozen OpenAI CLIP ViT-L/14 encoder (1,024-dimensional post-layer-normalization, pre-projection CLS), L2 normalization and PCA fitted on source training data. Four image-quality covariates may be added. JPEG, blur and resizing are applied to the face crop; CLIP features and covariates are recomputed. Video scores are means of frame probabilities. Ten random seeds were used in the reported series.

The controlled benchmark has five known functions, eight project-implemented heads and two budgets. Spline, RBF, rational and wavelet variants are implementations inspired by the literature; they are not numerical reproductions of the official FastKAN, KAT or Wav-KAN libraries. Baselines include linear and MLP heads. Parameter matching is approximate, and the linear baseline is not parameter matched. Explanation fidelity is assessed against the model's *actual* prediction change after intervention.