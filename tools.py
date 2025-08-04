import PIL
from data.re_dataset import re_eval_dataset, re_train_dataset
import open_clip
def get_ds_path(args):
    if args.dataset == 'rsitmd':
        return {"train_ann_file": './data/finetune_json/rsitmd_train.json',
                "test_ann_file": './data/finetune_json/rsitmd_test.json',
                "image":'./dataset/rsitmd'}
    else:
        raise NotImplementedError
if __name__ == '__main__':

    CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
        device="cuda:0",
        cache_dir='cache/weights/open_clip'
    )
    train_set = re_train_dataset(ann_file=['./data/finetune_json/rsitmd_train.json'],
                                  transform=preprocess_train,
                                  image_root='./dataset/rsitmd')
    