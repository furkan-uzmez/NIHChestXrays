from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


class ChestXrayDataset(Dataset):
    def __init__(self, DATASET_PATH,image_paths,transform=None):
        self.df = pd.read_excel(DATASET_PATH)
        path_dict = {os.path.basename(path): path for path in image_paths}
        self.df['image_path'] = self.df['Image Index'].map(path_dict) 
        self.df.loc[self.df['View Position'] == 'AP', 'View Position'] = 0
        self.df.loc[self.df['View Position'] == 'PA', 'View Position'] = 1
        self.transform = transform    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['View Position']

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_path(self,idx):
        return self.df.loc[idx]['image_path']