# FlowAttention: Lightweight Cross-Covariance Attention for Resource-Efficient Semi-Supervised Semantic Segmentation

[[Paper](#)] [[Project Page](#)]

단일 GPU(RTX 3090, 24GB) 환경에서 학습 가능한 경량 semi-supervised semantic segmentation 프레임워크입니다. ImageNet pretrained 없이 ResNet-50을 backbone으로 사용하고, Cross-Covariance Attention(XCA)을 통해 레이블 피처의 정보를 언레이블 피처로 전달하여, 멀티 GPU가 필요했던 [CorrMatch (CVPR 2024)](https://github.com/BBBBchan/CorrMatch) 대비 훨씬 적은 자원으로 유사한 성능을 목표로 합니다.

<p align="center">
  <img src="assets/framework.png" width="800">
  <br>
  <em>Fig. FlowAttention 전체 구조 (그림 추가 예정)</em>
</p>

## News

- **[YYYY.MM.DD]** 코드 공개

## Results

### Cityscapes

| Method | Backbone | Pretrained | Resolution | 1/8 | 1/4 | 1/2 | Checkpoint |
|---|---|---|---|---|---|---|---|
| CorrMatch (official) | ResNet-101 | ✅ | 801×801 | - | - | - | [link](#) |
| CorrMatch (reproduced) | ResNet-50 | ❌ | 448×448 | - | - | - | [link](#) |
| **FlowAttention (ours)** | ResNet-50 | ❌ | 448×448 | - | - | - | [link](#) |

> 학습/추론에 필요한 GPU, VRAM, 학습 시간 등 자원 비교는 논문 Table X 참고.

## Getting Started

### Installation

```bash
conda create -n flowattention python=3.10
conda activate flowattention
pip install -r requirements.txt
```

`requirements.txt` 예시:

```
torch>=2.4
torchvision
numpy
pillow
pyyaml
tqdm
einops
```

### Pretrained Backbone

본 방법은 ImageNet pretrained weight를 사용하지 않습니다. CorrMatch 재현 실험(pretrained 조건)에 필요한 backbone은 아래에서 받으세요.

| Backbone | 다운로드 |
|---|---|
| ResNet-50 | [link](#) |
| ResNet-101 | [link](#) |

받은 파일은 `./pretrained/` 아래에 위치시킵니다.

### Data Preparation

```
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```

데이터 경로는 `configs/cityscapes.yaml`의 `data_root`에 설정합니다.

```
├── FlowAttention
    ├── pretrained
    │   └── resnet50.pth
    ├── configs
    │   └── cityscapes.yaml
    ├── splits
    │   └── cityscapes
    │       ├── 1_8
    │       ├── 1_4
    │       └── 1_2
    ├── model
    │   ├── backbone
    │   │   └── resnet.py
    │   └── xca.py
    ├── scripts
    │   ├── train.sh
    │   └── val.sh
    ├── flowattention.py
    └── val.py
```

## Training

```bash
sh scripts/train.sh <num_gpu> <port>
```

예시 (GPU 1장):

```bash
sh scripts/train.sh 1 12345
```

주요 config 항목 (`configs/cityscapes.yaml`):

| 항목 | 값 | 설명 |
|---|---|---|
| `batch_size` | 4 | micro-batch (GPU당) |
| `accumulation_steps` | 4 | gradient accumulation (effective batch = batch_size × accumulation_steps) |
| `lr` | 0.00125 | effective batch 16 기준 |
| `crop_size` | 448 | |
| `epochs` | - | |

> 24GB 미만 GPU를 사용하는 경우 `batch_size`를 줄이고 `accumulation_steps`를 늘려 동일한 effective batch를 유지하세요.

## Evaluation

```bash
sh scripts/val.sh
```

`scripts/val.sh` 내 `checkpoint_path`를 평가할 체크포인트 경로로 수정한 뒤 실행합니다.

## Citation

본 코드를 사용하신다면 아래를 인용해주세요.

```bibtex
@article{flowattention2026,
  title={FlowAttention: Lightweight Cross-Covariance Attention for Resource-Efficient Semi-Supervised Semantic Segmentation},
  author={[Your Name]},
  journal={[Venue]},
  year={2026}
}
```

## Acknowledgement

본 프로젝트는 [CorrMatch](https://github.com/BBBBchan/CorrMatch)의 코드와 학습 프로토콜을 기반으로 하며, [UniMatch](https://github.com/LiheYoung/UniMatch), [CPS](https://github.com/charlesCXK/TorchSemiSeg) 등 semi-supervised segmentation 코드베이스의 관례를 따릅니다. XCA 모듈은 [XCiT (NeurIPS 2021)](https://arxiv.org/abs/2106.09681)의 cross-covariance attention을 기반으로 합니다.

```bibtex
@article{sun2023corrmatch,
  title={CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation},
  author={Sun, Boyuan and Yang, Yuqi and Zhang, Le and Cheng, Ming-Ming and Hou, Qibin},
  journal={IEEE Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@inproceedings{elnouby2021xcit,
  title={XCiT: Cross-Covariance Image Transformers},
  author={El-Nouby, Alaaeldin and Touvron, Hugo and Caron, Mathilde and Bojanowski, Piotr and Douze, Matthijs and Joulin, Armand and Laptev, Ivan and Neverova, Natalia and Synnaeve, Gabriel and Verbeek, Jakob and J{\'e}gou, Herv{\'e}},
  booktitle={NeurIPS},
  year={2021}
}
```

## License

CorrMatch의 코드를 기반으로 하는 부분은 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (비상업적 용도) 라이선스를 따릅니다. 본 저장소의 나머지 부분에 대한 라이선스는 `LICENSE` 파일을 참고하세요.