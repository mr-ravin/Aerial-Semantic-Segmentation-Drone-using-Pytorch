import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import generate_group_mask

color_dict = {
    0: [155,38,182],   # obstacles
    1: [14,135,204],   # water
    2: [124,252,0],    # nature
    3: [255,20,147],   # moving
    4: [169,169,169],  # landable
}


def get_grouped_class_color_map():
    return color_dict

def add_tag_on_image(image, text):
    xdim = image.shape[1]
    notice_pad = np.ones((40,xdim,3))*255
    notice_pad = np.uint8(notice_pad)
    cv2.putText(notice_pad, text, (12,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    out = cv2.vconcat([notice_pad, image])
    return out


def generate_rgb_mask(label_mask, total_channels, individual_class_count):
    label_mask = generate_group_mask(label_mask, individual_class_count)
    label_mask_shape = label_mask.shape
    img = np.zeros((label_mask_shape[0], label_mask_shape[1], 3), dtype = np.uint8)
    for idx in range(total_channels):
        class_mask = (label_mask == idx)*1
        class_mask_r = class_mask * color_dict[idx][0]
        class_mask_g = class_mask * color_dict[idx][1]
        class_mask_b = class_mask * color_dict[idx][2]
        class_mask_bgr = cv2.merge([class_mask_b, class_mask_g, class_mask_r])
        img = img + class_mask_bgr
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def generate_mask_from_model_output(numpy_output, total_channels):
    output_shape = numpy_output.shape
    img = np.zeros((output_shape[-2], output_shape[-1], 3), dtype = np.uint8)
    for idx in range(total_channels):
        class_id = idx
        class_mask = numpy_output[idx]
        class_mask_r = class_mask * color_dict[class_id][0]
        class_mask_g = class_mask * color_dict[class_id][1]
        class_mask_b = class_mask * color_dict[class_id][2]
        class_mask_bgr = cv2.merge([class_mask_b, class_mask_g, class_mask_r])
        img = img + class_mask_bgr
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def plot_graph(root, total_epoch, epoch_tr_loss, epoch_vl_loss, epoch_tr_iou, epoch_vl_iou):
    x_data = [i for i in range(1,total_epoch+1)]
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(60,30))    
    
    ax1.set_title("Train and Val Losses")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(x_data)
    ax1.plot(epoch_tr_loss, "r-+", label=" Train Loss")
    ax1.plot(epoch_vl_loss, "g--", label=" Val Loss")
    ax1.legend(["Train Loss","Val Loss"], loc='upper right')

    ax2.set_title("IOU Scores")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("mean IOU Score")
    ax2.set_xticks(x_data)
    ax2.plot(epoch_tr_iou, "r-+", label="Train mean IOU")
    ax2.plot(epoch_vl_iou, "g--", label="Val mean IOU")
    ax2.legend(["Train mean IOU","Val mean IOU"], loc='upper right')

    f.tight_layout(pad=2.0)
    plt.savefig(root+'/results/overall_analysis.png')