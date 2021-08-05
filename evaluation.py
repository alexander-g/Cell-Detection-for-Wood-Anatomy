import io
import numpy as np
import PIL.Image
from skimage.measure import label,regionprops
from skimage.morphology import remove_small_objects
import pylab as plt
import matplotlib as mpl


def compare_to_groundtruth(groundtruth, processed, ignore_buffer_px=0):
    '''Compares a prediction result to a ground truth segmentation map.
       Returns a visualization map and error statistics in csv format.'''
    processed  = remove_small_objects(processed>0.5,   min_size=32)
    groundtruth= remove_small_objects(groundtruth>0.5, min_size=32)
    t0         = filter_border_cells(processed,  ignore_buffer_px)[0]
    l0,inv0    = filter_border_cells(groundtruth,ignore_buffer_px)
    c0         = label(t0.astype(np.uint8))
    c1         = label(l0.astype(np.uint8))
    #not all predicted cells get filtered by filter_border_cells; second pass:
    invc1      = label(inv0.astype(np.uint8))
    invr1      = regionprops(invc1, cache=False)
    r0         = regionprops(c0, cache=False)
    invmatches = match_regions(invr1, r0, threshold=0.75) #matches between predicted cells and filtered ground truth cells
    t0         = t0 - (1*np.max([c0==m1 for _,m1 in invmatches], axis=0) if len(invmatches) else 0)
    c0         = label(t0.astype(np.uint8))
    r0         = regionprops(c0, cache=False)
    r1         = regionprops(c1, cache=False)
    matches    = match_regions(r1, r0)
    precision, recall         = (len(matches) / (c1.max()-1), len(matches) / (c0.max()-1))
    matched_predictions_map   = np.max([(c0==m0) for m1,m0 in matches]+[c0*0], axis=0 )*np.uint8(1) #+[c0*0] if there are no matches
    matched_labels_map        = np.max([(c1==m1) for m1,m0 in matches]+[c0*0], axis=0 )*np.uint8(1)
    unmatched_labels          = [i for i in np.arange(1,c1.max()+1) if i not in matches[:,0]]
    unmatched_labels_map      = np.max([(c1==m1) for m1 in unmatched_labels]+[c1*0], axis=0, initial=0 )*np.uint8(1)
    unmatched_predictions     = [i for i in np.arange(1,c0.max()+1) if i not in matches[:,1]]
    unmatched_predictions_map = np.max([(c0==m0) for m0 in unmatched_predictions]+[c0*0], axis=0, initial=0 )*np.uint8(1)
    vismap  = visualize(matched_predictions_map, matched_labels_map,
                                    unmatched_predictions_map, unmatched_labels_map, inv0)
    stats   = compute_statistics(matches, r0,r1, c0,c1, unmatched_predictions)
    return vismap, stats
comapare_to_groundtruth = compare_to_groundtruth #typo, left for legacy


def compute_statistics(matches, r0,r1, c0,c1, unp):
    matched_predictions_areas    = np.array([r0[m0-1].area for m1,m0 in matches ] )
    matched_label_areas          = np.array([r1[m1-1].area for m1,m0 in matches ] )
    per_cell_areas               = np.array([r.area for r in r1] )
    per_cell_centroids           = np.array([r.centroid for r in r1]) 

    matches_dict = dict(matches)
    per_cell_prediction_area     = np.array([ ( r0[matches_dict[i]-1].area                  if i in matches_dict else 0)     for i in np.arange(1,c1.max()+1) ])
    per_cell_prediction_centroid = np.array([ ( np.argwhere( c0==matches_dict[i] ).mean(0)  if i in matches_dict else (0,0)) for i in np.arange(1,c1.max()+1) ])
    per_cell_true_positives      = np.array([ ( ((c0==matches_dict[i])*(c1==i)).sum()       if i in matches_dict else 0) for i in np.arange(1,c1.max()+1) ])
    per_cell_false_negatives     = np.array([ ( (((c1==i)*1-(c0==matches_dict[i]))>0).sum() if i in matches_dict else 0) for i in np.arange(1,c1.max()+1) ])
    per_cell_false_positives     = np.array([ ( (((c1==i)*1-(c0==matches_dict[i]))<0).sum() if i in matches_dict else 0) for i in np.arange(1,c1.max()+1) ])
    per_cell_F1                  = ((2*per_cell_true_positives) / (2*per_cell_true_positives + per_cell_false_negatives + per_cell_false_positives + 1e-6))

    false_positive_centroids     = np.array([ r0[i-1].centroid for i in unp ])
    false_positive_areas         = np.array([ r0[i-1].area     for i in unp ])

    stats    = format_stats( per_cell_centroids,       per_cell_prediction_centroid, 
                                 per_cell_areas,           per_cell_prediction_area,
                                 per_cell_true_positives,  per_cell_false_positives,
                                 per_cell_false_negatives, per_cell_F1)
    FP_stats = format_FP_stats(false_positive_centroids, false_positive_areas)
    return stats, FP_stats

def format_stats(YX_c, YX_p, A_c, A_p, TP, FP, FN, F1):
    line0 = '#cell centroid position (y x), prediction centroid position (y x), cell area, prediction area, true positive area, false positive area, false negative area, F1 score'
    lines = [line0]
    for i in range(len(YX_c)):
        line = '(%.1f %.1f), (%.1f %.1f), %i, %i, %i, %i, %i, %5f'%(YX_c[i][0], YX_c[i][1], YX_p[i][0], YX_p[i][1], A_c[i], A_p[i], TP[i], FP[i], FN[i], F1[i])
        lines.append(line)
    return '\n'.join(lines)

def format_FP_stats(YX_p, A_p):
    line0 = '#prediction centroid position (y x), prediction area'
    lines = [line0]
    for i in range(len(YX_p)):
        line = '(%.1f %.1f), %i'%(YX_p[i][0], YX_p[i][1], A_p[i])
        lines.append(line)
    return '\n'.join(lines)


def visualize(mpm, mlm, upm, ulm, inv):
    colors  = np.array([(115,250,10), (0,200,165), (255,140,25), (0,0,255), (255,0,0), 
                        (255,0,255), (255,200,165), (255,140,255), (0,0,0), (128,128,128)], np.uint8)
    vismap  = np.zeros( (mpm.shape[:2] + (3,)), np.float32 )
    vismap += (mpm * mlm)        [..., np.newaxis] * colors[0]
    vismap += ((mlm*1.0 - mpm)>0)[..., np.newaxis] * colors[1]
    vismap += ((mlm*1.0 - mpm)<0)[..., np.newaxis] * colors[2]
    vismap += ulm[..., np.newaxis]                 * colors[3]
    vismap += upm[..., np.newaxis]                 * colors[4]
    vismap  = vismap*(1-inv)[...,np.newaxis]  +inv[...,np.newaxis] * colors[-1]
    vismap  = np.clip(vismap, 0, 255).astype(np.uint8)

    fig    = plt.figure(-1, figsize=(20,20), tight_layout=True)
    plt.imshow( vismap ); plt.yticks([]); plt.xticks([]);
    colorlabels = ['True Positive', 'False Negative Cell Area', 'False Positve Cell Area', 
                   'False Negative Cell Instance', 'False Positive Cell Instance', 'Unmatched Positive', 
                   'Disconnected Positive', 'Merged Negative', 'True Negative', 'Ignored (Border)']
    patches     = [ mpl.patches.Patch(color=colors[i]/255, label=l ) for i,l in enumerate(colorlabels) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 0.01), loc=3, borderaxespad=0., fontsize=20)

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', bbox_inches='tight')
    io_buf.seek(0)
    img_arr = np.array( PIL.Image.open(io_buf) )[5:-10, 10:]
    io_buf.close()

    plt.close(fig)
    return img_arr

def coords_IoU(xyset0, xyset1):
    return 2*len(xyset0.intersection(xyset1)) / (len(xyset0) + len(xyset1))

def match_regions(regions0, regions1, threshold=0.01):
    coords0 = [ set( (r.coords * [100000,1]).sum(-1) ) for r in regions0 ]
    coords1 = [ set( (r.coords * [100000,1]).sum(-1) ) for r in regions1 ]
    return np.array( match_predictions_labels(coords0, coords1, coords_IoU, threshold) ).reshape(-1,2)+1

def match_predictions_labels(labels, predictions, similarity_f, similarity_threshold=0.5):
    #calculate the similarity metric for all combinations of labels and predictions
    try:
        #try vectorized if the similarity function supports it
        similarities = np.array([similarity_f(l,predictions) for l in labels])
        assert similarities.shape == (len(labels), len(predictions))
    except:
        #otherwise in a nested loop
        similarities = np.array([[similarity_f(l,p) for p in predictions] for l in labels])

    #match labels and predictions that are most similar to each other
    matched_predictions = set()
    matched_pairs       = list()
    for j,label_similarities in enumerate(similarities):
        for idx in reversed(label_similarities.argsort()):
            if label_similarities[idx] < similarity_threshold: break         #since the similarities are sorted, we can stop here
            if idx in matched_predictions:                     continue      #this prediction has already a matched label
            matched_predictions.add(idx)
            matched_pairs += [(j, idx)]
            break
    return matched_pairs

def filter_border_cells(x, buffer=0):
    components = label(x)
    regions    = regionprops(components, cache=False)
    okregions  = [not (buffer>=r.bbox[0] or buffer>=r.bbox[1] or x.shape[0]-buffer<=r.bbox[2] or x.shape[1]-buffer<=r.bbox[3]) for r in regions ]
    valid      = np.max(  [np.zeros(x.shape, np.bool)] + [ (components==r.label) for i,r in enumerate(regions) if okregions[i]],  axis=0  )
    invalid    = np.max(  [np.zeros(x.shape, np.bool)] + [ (components==r.label) for i,r in enumerate(regions) if not okregions[i]],  axis=0  )
    return valid, invalid

