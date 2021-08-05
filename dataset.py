import os
import numpy as np
import torch
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None #Needed to open large images
import skimage.measure    as skmeasure
import skimage.morphology as skmorph


class Dataset(torch.utils.data.Dataset):
    '''Data loading and preprocessing. Converts segmentation maps into boxes and masks as required by MaskRCNN.'''
    
    def __init__(self, imagefiles, maskfiles, patchsize=1000, preprocess_mask_fn=None, augment=False):
        super().__init__()
        self.images     = imagefiles
        self.masks      = maskfiles
        self.augment    = augment
        self.patchsize  = patchsize
        self.preprocess_mask = preprocess_mask_fn
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image          = PIL.Image.open(self.images[idx]).convert('RGB') / np.float32(255)
        mask           = PIL.Image.open(self.masks[idx]).convert('L') / np.float32(255)
        mask,image     = (mask[:,::-1],   image[:,::-1])   if self.augment and np.random.random()<0.5 else (mask, image)
        mask,image     = (mask[::-1],     image[::-1])     if self.augment and np.random.random()<0.5 else (mask, image)
        mask,image     = (np.rot90(mask), np.rot90(image)) if self.augment and np.random.random()<0.5 else (mask, image)
        
        if self.preprocess_mask:
            mask      = self.preprocess_mask(mask)
            
        randomyx       = [np.random.randint(max(1,image.shape[0]-self.patchsize)), 
                          np.random.randint(max(1,image.shape[1]-self.patchsize))]
        randombox      = randomyx + [randomyx[0]+self.patchsize, randomyx[1]+self.patchsize]
        image          = image[randombox[0]:, randombox[1]:][:self.patchsize, :self.patchsize]
        mask           = mask[randombox[0]:, randombox[1]:][:self.patchsize, :self.patchsize]
        
        mask           = skmorph.remove_small_objects(mask>0, min_size=32)
        labelmask      = skmeasure.label(mask)
        regions        = skmeasure.regionprops(labelmask)
        boxes          = np.array([r.bbox for r in regions]).reshape(-1,4)
        #convert from xyxy to yxyx format
        boxes          = boxes[..., (1,0,3,2)]
        boxes          = torch.as_tensor(boxes, dtype=torch.float32)
        labels         = torch.ones((len(boxes),), dtype=torch.int64)
        masks          = torch.as_tensor( (labelmask == np.unique(labelmask)[1:,np.newaxis,np.newaxis]) )
        target         = dict(boxes=boxes, labels=labels, masks=masks)
        return image.transpose(2,0,1).copy(), target


#preprocessing functions, remove small or large vessels from the ground truth
#specifically for oak/quercus
remove_large_vessels =  lambda x: x*1 - skmorph.remove_small_objects(x>0, min_size=100000)*x.max()
remove_small_vessels =  lambda x: skmorph.remove_small_objects(x>0, min_size=25000)*x.max()


def collate_fn(batchlist):
    images    = [x for x,y in batchlist]
    targets   = [y for x,y in batchlist]
    return torch.as_tensor(images), targets

def create_dataloader(ds, shuffle, batchsize=1):
    return torch.utils.data.DataLoader(ds, pin_memory=True, shuffle=shuffle,
                                       num_workers=os.cpu_count(), collate_fn=collate_fn,
                                       worker_init_fn=lambda x: np.random.seed(torch.randint(0,1000,(1,))[0].item()+x),)