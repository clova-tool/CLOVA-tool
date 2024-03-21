import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import json
import numpy as np


def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)


########################################################################################
# GQA
########################################################################################

class GQADataset(Dataset):
    BALANCED_TYPE = {
        True: "balanced",
        False: "all"
    }

    def __init__(self, split, balanced=True, data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 verbose=True, testing=False):
        """
        Args:
            split (str): Data split. One of ["challenge", "submission", "test", "testdev", "train", "val"]
            balanced (bool): You balanced version or full version.
            image_transforms:
            question_transforms:
            tokenize (fct):
            verbose (bool): Print some infos. Default=True
            testing (bool): Set to true for data splits without targets. Default=False.
        """
        start_time = time.time()
        self.split = split
        self.testing = testing
        assert split in ["challenge", "submission", "test", "testdev", "train", "val"]
        self.balanced = balanced
        self.balanced_type = self.BALANCED_TYPE[balanced]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize

        if not balanced and split == "train":
            raise NotImplementedError
        else:
            self.file_name = f"questions1.2/{self.split}_{self.balanced_type}_questions.json"
            path = os.path.expanduser(os.path.join(data_path, self.file_name))
            if verbose:
                print(f"Start loading GQA Dataset from {path}", flush=True)
            self.df = pd.read_json(path, orient="index")
            self.json=self.loadFile(path)

        self.n_samples = self.df.shape[0]
        if verbose:
            print(
                f"Loading GQA Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")

    def loadFile(self,name):
        # load standard json file
        if os.path.isfile(name):
            with open(name) as file:
                data = json.load(file)
        # load file chunks if too big 
        elif os.path.isdir(name.split(".")[0]):
            data = {}
            chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir = name.split(".")[0], ext = name.split(".")[1]))
            for chunk in chunks:
                with open(chunk) as file: 
                    data.update(json.load(file))
        else:
            raise Exception("Can't find {}".format(name))
        return data

    def __getitem__(self, index):
        # image input
        sample_id = self.df.iloc[index].name
        image_id = self.df.iloc[index]["imageId"]
        question = self.df.iloc[index]["question"]
        # split = self.split[index]
        if not self.testing:
            answer = self.df.iloc[index]["answer"]
            question_type = self.df.iloc[index]["groups"]["global"]

        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, "images", f"{image_id}.jpg"))
        # print ('image_path',image_path)
        with open(image_path, "rb") as f:
            img = Image.open(f)

        if self.image_transforms:
            img = self.image_transforms(img)

        # Load, transform and tokenize question
        if self.question_transforms:
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)

        # Return
        # if self.testing:
        #     return {"sample_id": sample_id, "image_id": image_id, "answer": None, "img": img, "question": question, "question_type": None}
        # else:
        #     return {"sample_id": sample_id, "image_id": image_id, "answer": answer, "img": img, "question": question,
        #             "question_type": question_type}

        if self.testing:
            return {"sample_id": sample_id, "image_id": image_id, "answer": None, "img": image_path, "question": question}
        else:
            return {"sample_id": sample_id, "image_id": image_id, "answer": answer, "img": image_path, "question": question}

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.df.shape[0]
    

if __name__ == "__main__":

    from Datasets.loaders import GQADataset, TextVQADataset, VQAv2Dataset, OKVQADataset


    for split, testing in [("train", True), ("val", True), ("testdev", True), ("test", True),
                           ("challenge", True)]:  # , ("submission", True)]:
        dataset = GQADataset(split=split, balanced=True, data_path="~/Data/GQA", testing=testing)
        print("Length:", len(dataset), "\nData:", dataset[0], "\n")