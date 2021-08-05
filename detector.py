import numpy as np
import PIL.Image
import torch, torchvision
import pytorch_lightning as pl
import util

def create_basemodel(**kwargs):
    return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=False, **kwargs)

def create_conifer_basemodel(**kwargs):
    return create_basemodel(box_detections_per_img=2000, rpn_pre_nms_top_n_test=24000, **kwargs)


class TrainingModel(pl.LightningModule):
    def __init__(self, basemodel):
        super().__init__()
        self.basemodel = basemodel
    
    def training_step(self, batch, i):
        lossdict  = self.basemodel(*batch)
        loss      = sum(loss for loss in lossdict.values())
        self.log('loss', loss)
        for lossname, lossval in lossdict.items():
            self.log(lossname, lossval, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optim    = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        sched    = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)
        return [optim], [sched]


class PrintMetricsCallback(pl.callbacks.progress.ProgressBarBase):
    '''Prints metrics after each training epoch in a compact table'''
    def on_train_epoch_end(self, trainer, pl_module, *args):
        metrics_str = ' | '.join([f'{k}:{float(v):>9.5f}' for k,v in trainer.progress_bar_dict.items()])
        print(f'[{trainer.current_epoch:04d}]', metrics_str)
    def on_train_batch_end(self, trainer, pl_module, *args):
        super().on_train_batch_end(trainer, pl_module, *args)
        percent = (self.train_batch_idx / self.total_train_batches)
        metrics_str = ' | '.join([f'{k}:{float(v):>9.5f}' for k,v in trainer.progress_bar_dict.items()])
        print(f'[{percent:.2f}] {metrics_str}', end='\r')

        
def create_trainer(epochs=10):
    return pl.Trainer(max_epochs=epochs, gpus=1, checkpoint_callback=None, logger=False, 
                      terminate_on_nan=True, gradient_clip_val=1.0, #clipping gradient because I sometimes get NaNs without it
                      callbacks=[PrintMetricsCallback()])


class FullDetector:
    '''Combines multiple basemodels (e.g. for oak) and automatically applies slicing for large images.'''
    
    def __init__(self, basemodels, patchsizes, slacks):
        assert len(basemodels) == len(patchsizes) == len(slacks)
        self.basemodels  = basemodels
        for m in self.basemodels:
            #minimum detection score: 0.5, gives a small speed boost
            m.roi_heads.score_thresh = 0.5
        self.patchsizes  = patchsizes
        self.slacks      = slacks
    
    @staticmethod
    def load_image(imgpath):
        return PIL.Image.open(imgpath) / np.float32(255)
    
    def predict_patches(self, model, patches, callback=None):
        maskpatches = []
        for i,patch in enumerate(patches):
            inputs   = patch.transpose(2,0,1)[np.newaxis]
            with torch.no_grad():
                device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                outputs  = model.to(device)(torch.as_tensor(inputs, device=device))
            outputs  = [outputs[0]['masks'].cpu().numpy()]
            masks    = outputs[0][:,0]
            maskpatches += [ masks.max(axis=0) if len(masks) else np.zeros(masks.shape[1:]) ]
            if callback:
                callback( (i+1) / len(patches) )
        model.cpu();
        return np.stack(maskpatches)
    
    def process_image(self, image, progress_callback=None):
        H,W             = image.shape[:2]
        fullresults     = []
        for model, slack, patchsize in zip(self.basemodels, self.slacks, self.patchsizes):
            paddings        = [(0,max(0,patchsize-H)), (0,max(0,patchsize-W)), (0,0)]
            padded          = np.pad(image, paddings, mode='constant')
            patches         = util.slice_into_patches_with_overlap(padded, patchsize, slack)
            resultpatches   = self.predict_patches(model, patches, callback=progress_callback)
            fullresult      = util.stitch_overlapping_patches(resultpatches, image.shape, slack=slack )[:H,:W]
            fullresults    += [fullresult]
        fullresult      = np.max(fullresults, axis=0)
        finalresult     = (fullresult > 0.5)
        return finalresult