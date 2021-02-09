from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
root_info = '/home/xutengfei/meisai2021'
root_img = '/home/xutengfei/meisai2021/imgs'


def default_loader(path):

    # ANTIALIAS:high quality
    return Image.open(path).resize((224, 224), Image.ANTIALIAS).convert('RGB')


class MyDataset(Dataset):
    """MyDataset for read garbage_cls data.
    """

    def __init__(self, root_info=root_info, root_img=root_img, transform=None, target_transform=None,
                 loader=default_loader, negative_num=7):

        self.root_info = root_info
        self.root_img = root_img
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        imgs = []
        df_filename_status = pd.read_csv(os.path.join(root_info, 'info.csv'))
        df_filename_status.sample(frac=1) # shuffle the dataframe ,so every time we get different negatives
        nid = 0
        for index, row in df_filename_status.iterrows():
            if df_filename_status.loc[index, 'Lab Status'] == 'Positive ID':
                imgs.append((df_filename_status.loc[index, 'FileName'], 1))
            elif df_filename_status.loc[index, 'Lab Status'] == 'Negative ID' and df_filename_status.loc[index, 'FileName'].endswith(('.jpg', '.png')) and nid < negative_num:
                imgs.append((df_filename_status.loc[index, 'FileName'], 0))
                nid = nid+1
        self.imgs = imgs

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        # the returned "path" is a relative one
        img = self.loader(os.path.join(root_img, path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
