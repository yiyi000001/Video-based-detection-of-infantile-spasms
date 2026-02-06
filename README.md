# Infantile Spasm Detection Based on Spatio-temporal Attention Graphconv UNet

Infantile Spasm Detection Based on Spatio-temporal Attention Graphconv UNet [[paper](https://ieeexplore.ieee.org/document/11180141)]

Accurate detection of infantile spasms is crucial
for timely intervention and treatment. The widespread use of
electroencephalography (EEG) is limited, hindering the popu-
larization of EEG-based epilepsy analysis in under-resourced
medical settings. In response to these challenges, this study
introduces a novel approach called Spatio-temporal Attention
Graphconv U-Net (STA-GUNet) for detecting infantile spasms
solely based on video data. The dataset comprises video clips
collected from clinical records at China Medical University,
consisting of 559 five-second clips depicting infantile spasms
and 1031 five-second clips without infantile spasms. The videos
are analyzed using the YOLOv7-Infant detection method, which
tracks the dynamic changes in infants’ skeletal movements by
examining key features such as position, velocity, and accel-
eration in spatio-temporal context. The proposed model au-
tonomously learns spatial and temporal patterns, leveraging
attention mechanisms, graph convolutional networks, and a U-
Net structure to capture intricate spatio-temporal dependencies,
thereby enhancing the classification accuracy and robustness.
To assess the model’s generalizability, a ten-fold cross-validation
was conducted, demonstrating an overall accuracy of 93.14%,
precision of 90.04%, F1 score of 90.28%, AUROC of 91.84%
and recall of 90.52%.Furthermore, an independent evaluation on
a YouTube video dataset of infantile spasms demonstrates that
STA-GUNet maintains strong detection performance, confirming
its robustness and potential applicability in diverse real world
scenarios. To enhance the predictive performance of the model
and validate its robustness, we completely separate the training
set and the test set, and then proceed with the validation. These
results support that the proposed detection method not only
accurately identifies infantile spasms but also maintains a low
false-positive rate, positioning it as a valuable adjunctive tool for
clinical diagnosis.


<img width="785" height="344" alt="d2ce6470-f935-45a9-b1b7-e56028c236ee" src="https://github.com/user-attachments/assets/bf975a70-a421-441d-be31-d22814faae8c" />


## Datasets requirements
json
{
  "features": [
    {
      "coordinate": [x, y],
      "speed": [vx, vy],
      "acceleration": [ax, ay]
    }
  ]
}


## Quick Start
```python
import os
import json
python tr.py






