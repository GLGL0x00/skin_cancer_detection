import h5py
import numpy as np 
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader

class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )
        target = self.targets[index]
        target  = torch.tensor(target, dtype=torch.int64)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]

        img = torch.tensor(img, dtype=torch.float32)
            
        return img, target

    
def prepare_loaders(df_train, h5_file, augmentations, CONFIG, num_workers=10):
    
    train_dataset = ISICDataset(df_train, h5_file, transforms=augmentations)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)
    
    return train_loader




# class ISICDataset(Dataset):
#     def __init__(self, hdf5_file, csv_file):
#         # Load image data from hdf5
#         self.hdf5_data = h5py.File(hdf5_file, 'r')
#         self.isic_ids = list(self.hdf5_data.keys())
        
#         # Load metadata from csv
#         self.metadata_df = pd.read_csv(csv_file)
        
#         # Align metadata with image data based on 'isic_id'
#         self.metadata_df.set_index('isic_id', inplace=True)
        
#     def __len__(self):
#         return len(self.isic_ids)
    
#     def __getitem__(self, idx):
#         # Get image by isic_id
#         isic_id = self.isic_ids[idx]
#         image = self.hdf5_data[isic_id][:]
        
#         # Get corresponding metadata
#         metadata = self.metadata_df.loc[isic_id].values
        
#         # Convert to torch tensors
#         image = torch.tensor(image, dtype=torch.float32)
#         metadata = torch.tensor(metadata, dtype=torch.float32)
        
#         return image, metadata


