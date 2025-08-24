import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class ACDC(data.Dataset):
    def __init__(
            self,
            acdc_root = '/mnt/data/asus/tyh/sfda/cma-main/DATA_DIR/ACDC',
            data_list = 'ACDC_val.txt',
            stage=None,
            max_iters=None,  # None    62500*4
            num_classes=19,  # 19
            split="train",  # train   test  val
            transform=None,  # （1536 768）
            ignore_label=255,  # 255
            cfg=None,
            pseudo=False,  # False   3000iters后True
            debug=False,  # False
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.acdc_root = acdc_root
        self.data_list = []
        self.pseudo = pseudo
        self.stage = stage

        with open(data_list, "r") as handle:
            content = handle.readlines()
        #GP030176_frame_000569   _gt_labelTrainIds.png
        for fname in content:  # snow/val/GP030176/GP030176_frame_000569  _rgb_anon.png
            name = fname.strip()
            """
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.acdc_root, "rgb_anon/%s" % (name)
                    ),
                    "label": os.path.join(
                        self.acdc_root, "gt/%s" % (name.split('_rgb_anon.png')[0] + "_gt_labelTrainIds.png"),
                    ),
                    "name": name,
                }
            )
            """
            if pseudo:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.acdc_root, "rgb_anon/%s" % (name)
                        ),
                        "label": os.path.join(
                            cfg.OUTPUT_DIR, "CTR_O/%s"
                                            % (
                                                name.replace('/','_'),
                                            ),
                        ),
                        "name": name,
                    }
                )
            else:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.acdc_root, "rgb_anon/%s" % (name)
                        ),

                        "label": os.path.join(
                            self.acdc_root, "gt/%s" % (name.split('_rgb_anon.png')[0] + "_gt_labelTrainIds.png"),
                        ),
                        "name": name,
                    }
                )


        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
        name = datafiles["name"]

        if self.transform is not None:
            image, label, _ = self.transform(image, label)
        return image, label, name
