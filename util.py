import numpy as np
import pylab as plt
import matplotlib as mpl



def slice_into_patches_with_overlap(image, patchsize=1024, slack=32):
    '''Slices a numpy array image into overlapping patches of size `patchsize`'''
    grid      = grid_for_patches(image.shape, patchsize, slack)
    patches   = [image[i0:i1, j0:j1] for i0,j0,i1,j1 in grid.reshape(-1, 4)]
    return patches

def stitch_overlapping_patches(patches, imageshape, slack=32, out=None):
    '''Merges the patches as returned by `slice_into_patches_with_overlap` back into the original shape'''
    patchsize = patches[0].shape[0]
    grid      = grid_for_patches(imageshape, patchsize, slack)
    halfslack = slack//2
    i0,i1     = (grid[grid.shape[0]-2,grid.shape[1]-2,(2,3)] - grid[-1,-1,(0,1)])//2
    d0 = np.stack( np.meshgrid( [0]+[ halfslack]*(grid.shape[0]-2)+[           i0]*(grid.shape[0]>1),
                                [0]+[ halfslack]*(grid.shape[1]-2)+[           i1]*(grid.shape[1]>1), indexing='ij' ), axis=-1 )
    d1 = np.stack( np.meshgrid(     [-halfslack]*(grid.shape[0]-1)+[imageshape[0]],      
                                    [-halfslack]*(grid.shape[1]-1)+[imageshape[1]], indexing='ij' ), axis=-1 )
    d  = np.concatenate([d0,d1], axis=-1)
    if out is None:
        out = np.zeros(imageshape[:2]+patches[0].shape[2:])
    for patch,gi,di in zip(patches, d.reshape(-1,4), (grid+d).reshape(-1,4)):
        out[di[0]:di[2], di[1]:di[3]] = patch[gi[0]:gi[2], gi[1]:gi[3]]
    return out

def grid_for_patches(imageshape, patchsize, slack):
    #helper function for slicing and stitching
    H,W       = imageshape[:2]
    stepsize  = patchsize - slack
    grid      = np.stack( np.meshgrid( np.minimum( np.arange(patchsize, H+stepsize, stepsize), H ), 
                                       np.minimum( np.arange(patchsize, W+stepsize, stepsize), W ), indexing='ij' ), axis=-1 )
    grid      = np.concatenate([grid-patchsize, grid], axis=-1)
    grid      = np.maximum(0, grid)
    return grid
        
def draw_box(box, color='w', linewidth=None):
    '''Convenience function to draw a box (shape:(4,)) on a matplotlib/pylab plot'''
    axes     = plt.gca()
    rect     = mpl.patches.Rectangle(box[:2], *(box[2:]-box[:2]), linewidth=linewidth, edgecolor=color, facecolor='none')
    axes.add_patch( rect )