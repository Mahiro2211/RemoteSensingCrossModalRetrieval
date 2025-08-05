This repo aims to provide a common baseline for remote sensing cross-modal retrieval. The split of the dataset follows Harma(see citation below), thanks to the author!

## TODO-List
* add more Parameter-Efficient Method
* add wandb support
* add more visualization

## Introduction



<hr>

| Methods                | Backbone (image/text)       | Trainable Params | Image-to-text |        |         | Text-to-image |        |         |      mR      |
|------------------------|-----------------------------|------------------|---------------|--------|---------|---------------|--------|---------|--------------|
|                        |                             |                  | R@1           | R@5    | R@10    | R@1           | R@5    | R@10    |              |
| **Traditional methods**|                             |                  |               |        |         |               |        |         |              |
| Full-FT CLIP           | CLIP(ViT-B-32)              | 151M             | 20.4          | 51.59  | 71.5    | 25.44         | 48.67  | 61.28   | 46.48        |
| Full-FT GeoRSCLIP      | GeoRSCLIP(ViT-B-32-RET-2)   | 151M             | 24.87         | 60.75  | 77.43   | 29.42         | 54.20  | 65.71   | 52.06 |
