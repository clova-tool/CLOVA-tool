import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import json 

class COCO2014Dataset(Dataset):
    IMAGE_PATH = {
        "train": ("train2014", "annotations_trainval2014/annotations/captions_train2014.json"),
        "val": ("val2014", "annotations_trainval2014/annotations/captions_val2014.json"),
        "test": ("test2015", "image_info_test2015/annotations/image_info_test2015.json"),
        "testdev": ("test2015", "image_info_test2015/annotations/image_info_test-dev2015.json"),
        "test2014": ("test2014")}

    def __init__(self, split, data_path="",
                 image_transforms=None, caption_transforms=None, tokenize=None,
                 verbose=True, testing=False):
        """
        split train, val, test
        balanced True, False
        image_transforms
        caption_transforms
        """
        start_time = time.time()
        self.split = split
        self.testing = testing
        assert split in ["train", "val", "test", "testdev"], 'Use a split of ["train", "val", "test", "testdev"]'
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.caption_transforms = caption_transforms
        self.tokenize = tokenize
        path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][1]))

        if verbose:
            print(f"Start loading COCO2014 Dataset from {path}", flush=True)

        # Questions
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data["images"])

        # Annotations
        if not testing:
            df_annotations = pd.DataFrame(data["annotations"])
            df_annotations["file_name"] = df_annotations["image_id"].apply(lambda x: f"COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")

            #path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][2]))
            #with open(path, 'r') as f:
            #    data = json.load(f)
            #df_annotations = pd.DataFrame(data["annotations"])
            df = pd.merge(df_annotations, df, left_on='file_name', right_on='file_name', how='left')
        self.df = df
        self.n_samples = self.df.shape[0]
        if verbose:
            print(
                f"Loading COCO2014 Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")

    def __getitem__(self, index):
        # image input
        image_path = self.df.iloc[index]["file_name"]
        split = self.split

        if not self.testing:
            image_id = self.df.iloc[index]["image_id"]
            caption_id = self.df.iloc[index]["id_x"]
            caption = self.df.iloc[index]["caption"]

        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, self.IMAGE_PATH[split][0], image_path))
        with open(image_path, "rb") as f:
            img = Image.open(f)
        if self.image_transforms:
            img = self.image_transforms(img)

        # Load, transform and tokenize caption
        if self.caption_transforms:
            caption = self.question_transforms(caption)
        if self.tokenize:
            caption = self.tokenize(caption)

        # Return
        if self.testing:
            return {"img": img}
        else:
            return {"img": img, "image_id": image_id, "caption_id": caption_id, "caption": caption}

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.df.shape[0]


if __name__ == "__main__":

    # COCO2014Dataset
    for split, testing in [("train", False), ("val", False), ("test", True), ("testdev", True)]:
        print("==>", split, testing)
        dataset = COCO2014Dataset(split=split, data_path="~/Data/COCO2014", testing=testing)
        print("Length:", len(dataset), "\nData:", dataset[0], "\n")
        print("="*80)
