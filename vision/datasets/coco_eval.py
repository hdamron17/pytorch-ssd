# Overlay dataset class for use with COCO evaluation

from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

def numpy_to_coco(data, categories):
    img_ids_list = list(np.unique(data[:,0].astype(np.uint32)))
    img_ids = [{"id": k} for k in img_ids_list]
    cat_ids_list = list(np.unique(data[:,6].astype(np.uint32)))
    cat_ids = [{"id": k, "name": categories[k]} for k in cat_ids_list]

    base = COCO()
    base.dataset["images"] = img_ids
    base.dataset["categories"] = cat_ids
    base.createIndex()
    return base.loadRes(data)

def dataset_to_coco(dataset):
    data = []
    for i in range(len(dataset)):
        boxes, labels, _ = dataset.get_annotation(i)[1]
        for box, label in zip(boxes, labels):
            data.append(np.concatenate(([float(i)], box, [1.0, float(label)])))
    numpy_data = np.vstack(data)
    return numpy_to_coco(numpy_data, dataset.class_names)

def coco_pr_curve(coco_eval, classes, name, file_prefix=None):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

    for cls_ind, cls in enumerate(classes[1:]):
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
        x = np.linspace(0, 1, 101)
        for i in range(ind_hi+1 - ind_lo):
            plt.plot(x, precision[i, :], label="IOU = %f" % (i*0.05 + 0.5))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.title(name + " (%s) IOU = %f" % (cls, i*0.05 + 0.5))
            if file_prefix is None:
                plt.show()
            else:
                plt.savefig(file_prefix + '-%s-%s.png' % (cls,int((i*0.05 + 0.5)*100)))
                plt.close()
