# Multi-modal Text Recognition Networks: Interactive Enhancements between Visual and Semantic Features (ECCV 2022)

| [paper](https://arxiv.org/pdf/2111.15263) |


Official PyTorch implementation for Multi-modal Text Recognition Networks: Interactive Enhancements between Visual and Semantic Features (MATRN) in [ECCV 2022](https://eccv2022.ecva.net/).

**[Byeonghu Na](https://github.com/wp03052), [Yoonsik Kim](https://github.com/terryoo), and [Sungrae Park](https://github.com/sungraepark)**

This paper introduces a novel method, called Multi-modAl Text Recognition Network (MATRN), that enables interactions between visual and semantic features for better recognition performances.

<img src="./figures/overview.png" width="1000" title="overview" alt="An overview of MATRN. A visual feature extractor and an LM extract visual and semantic features, respectively. By utilizing the attention map, representing relations between visual features and character positions, MATRNs encode spatial information into the semantic features and hide visual features related to a randomly selected character. Through the multi-modal feature enhancement module, visual and semantic features interact with each other and the enhanced features in two modalities are fused to finalize the output sequence.">

## Datasets

We use lmdb dataset for training and evaluation dataset.
The datasets can be downloaded in [clova (for validation and evaluation)](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) and [ABINet (for training and evaluation)](https://github.com/FangShancheng/ABINet#datasets).

* Training datasets
    * [MJSynth (MJ)](https://www.robots.ox.ac.uk/~vgg/data/text/)
    * [SynthText (ST)](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
    * [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)
* Validation datasets
    * The union of the training set of [ICDAR2013](https://rrc.cvc.uab.es/?ch=2), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4), [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [Street View Text](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
* Evaluation datasets
    * Regular datasets 
        * [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) (IIIT)
        * [Street View Text](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) (SVT)
        * [ICDAR2013](https://rrc.cvc.uab.es/?ch=2): IC13<sub>S</sub> with 857 images, IC13<sub>L</sub> with 1015 images
    * Irregular dataset
        * [ICDAR2015](https://rrc.cvc.uab.es/?ch=4): IC15<sub>S</sub> with 1811 images, IC15<sub>L</sub> with 2077 images
        * [Street View Text Perspective](https://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf) (SVTP)
        * [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html) (CUTE)
* Tree structure of `data` directory
    ```
    data
    ├── charset_36.txt
    ├── evaluation
    │   ├── CUTE80
    │   ├── IC13_857
    │   ├── IC13_1015
    │   ├── IC15_1811
    │   ├── IC15_2077
    │   ├── IIIT5k_3000
    │   ├── SVT
    │   └── SVTP
    ├── training
    │   ├── MJ
    │   │   ├── MJ_test
    │   │   ├── MJ_train
    │   │   └── MJ_valid
    │   └── ST
    ├── validation
    ├── WikiText-103.csv
    └── WikiText-103_eval_d1.csv
    ```

## Requirements

```
pip install torch==1.7.1 torchvision==0.8.2 fastai==1.0.60 lmdb pillow opencv-python
```

### Pretrained Models

* Download pretrained model of MATRN from this [link](https://www.dropbox.com/s/pjcarm73cqwbxh4/best-train-matrn.pth?dl=0). Performances of the pretrained models are:

|Model|IIIT|SVT|IC13<sub>S</sub>|IC13<sub>L</sub>|IC15<sub>S</sub>|IC15<sub>L</sub>|SVTP|CUTE|
|-|-|-|-|-|-|-|-|-|
|MATRN|96.7|94.9|97.9|95.8|86.6|82.9|90.5|94.1|

* If you want to train with pretrained visioan and language model, download pretrained model of vision and language model from [ABINet](https://github.com/FangShancheng/ABINet#pretrained-models).


## Training and Evaluation

* Training
```
python main.py --config=configs/train_matrn.yaml
```

* Evaluation
```
python main.py --config=configs/train_matrn.yaml --phase test --image_only
```
Additional flags:
- `--checkpoint /path/to/checkpoint` set the path of evaluation model 
- `--test_root /path/to/dataset` set the path of evaluation dataset
- `--model_eval [alignment|vision|language]` which sub-model to evaluate
- `--image_only` disable dumping visualization of attention masks

## Acknowledgements

This implementation has been based on [ABINet](https://github.com/FangShancheng/ABINet).

## Citation
Please cite this work in your publications if it helps your research.
```bash 
@inproceedings{na2022multi,
  title={Multi-modal Text Recognition Networks: Interactive Enhancements between Visual and Semantic Features},
  author={Na, Byeonghu and Kim, Yoonsik and Park, Sungrae},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
 ```
