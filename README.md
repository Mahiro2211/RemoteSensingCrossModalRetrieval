This repo aims to provide a common baseline for remote sensing cross-modal retrieval. The split of the dataset follows Harma(see citation below), thanks to the author!

## Setup

```shell
conda create -n rsbs python=3.10
pip install -r requirements.txt
```

## TODO-List

* add more Parameter-Efficient Method
* add wandb support
* add more visualization

<hr>

## Details

* training 50 epochs per experiment.
* embeding size is 512 by default.
* learning rate is 4e-6 on clip model and 1e-5 on adatper.
* all experiment's loss function use CLIPLOSS (implemented in open_clip.loss library)

## Results

### RSITMD

| Methods                | Backbone (image/text)       | Trainable Params | I2T |I2T       |I2T        | T2I | T2I       | T2I         |      mR      |
|------------------------|-----------------------------|------------------|---------------|--------|---------|---------------|--------|---------|--------------|
|                        |                             |                  | R@1           | R@5    | R@10    | R@1           | R@5    | R@10    |              |
| **CLIP BaseMethod methods**|                             |                  |               |        |         |               |        |         |              |
| Full-FT CLIP           | CLIP(ViT-B-32)              | 151M             | 20.4          | 51.59  | 71.5    | 25.44         | 48.67  | 61.28   | 46.48        |
| Full-FT GeoRSCLIP      | GeoRSCLIP(ViT-B-32-RET-2)   | 151M             | 24.87         | 60.75  | 77.43   | 29.42         | 54.20  | 65.71   | 52.06 |
|CLIP Adatpter| CLIP(ViT-B-32)| 0.26M| 10.40|25.88|39.16|9.87|30.88|46.73|27.19|
|CLIP Adatpter| GeoRSCLIP(ViT-B-32-RET-2) | 0.26M|27.88|51.33|65.71|23.45|55.75|74.56|49.78|


# CITATION

```bibtex
@article{huang2024efficient,
  title={Efficient Remote Sensing with Harmonized Transfer Learning and Modality Alignment},
  author={Huang, Tengjun},
  journal={arXiv preprint arXiv:2404.18253},
  year={2024}
}
```