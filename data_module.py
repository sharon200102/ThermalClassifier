import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.download_dataset import download_funcs
from datasets.get_dataset import datasets_dict

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, 
                dataset_name: str,
                class2idx: dict,
                root_dir: str,
                class_mapper: dict,
                train_batch_size: int = 32, 
                val_batch_size: int = 32,
                test_batch_size: int = 32,
                train_num_workers: int = 2,
                val_num_workers: int = 2,
                test_num_workers: int = 2) -> None:

        super().__init__()

        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)

        self.class2idx = class2idx
        self.class_mapper = class_mapper

        # dataloader params
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers


    def prepare_data(self):
        download_funcs[self.dataset_name](self.root_dir)
    
    def setup(self, stage: str) -> None:
        
        if stage == 'fit':
            self.train_dataset = datasets_dict[self.dataset_name](data_root_dir=self.root_dir,
                                               split="train",
                                               class2idx=self.class2idx,
                                               class_mapper=self.class_mapper)
            
            self.val_dataset = datasets_dict[self.dataset_name](data_root_dir=self.root_dir,
                                                    split="val",
                                                    class2idx=self.class2idx,
                                                    class_mapper=self.class_mapper) 

        if stage == 'test':
            self.test_dataset = datasets_dict[self.dataset_name](data_root_dir=self.root_dir,
                                              split="test",
                                              class2idx=self.class2idx,
                                              class_mapper=self.class_mapper)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.train_batch_size, 
                          num_workers=self.train_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.val_batch_size, 
                          num_workers=self.val_num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.test_num_workers)