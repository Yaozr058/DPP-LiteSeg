import time
import numpy as np
import cv2
from DPPLiteSeg_D2STDCNet import *


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb = labelmap_rgb + (labelmap == label)[:, :, np.newaxis] * \
                       np.tile(colors[label],
                               (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def main():
    base_size = 960
    height_size = 720
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # fix colors
    fixed_colors = np.array([
        [128, 64, 128],
        [192, 0, 0],
        [128, 128, 128],
        [128, 128, 0],
        [0, 0, 192],
        [64, 0, 128],
        [192, 192, 128],
        [64, 64, 128],
        [64, 64, 0],
        [0, 128, 192],
        [128, 128, 128],
        [128, 0, 192]
    ], dtype=np.uint8)

    # 记录模型加载开始时间
    model_load_start = time.time()
    backbone = DeSTDC2() #or DeSTDC1
    model = ppliteseg_test_ddr(
        num_classes=12, backbone=backbone,
    )
    model.eval()
    ckpt = torch.load("your.pth", weights_only=False)
    model = model.cuda()
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    # 记录模型加载结束时间，并计算耗时
    model_load_end = time.time()
    model_load_time = model_load_end - model_load_start
    print(f"model load Timing: {model_load_time:.4f} 秒")

    img = cv2.imread("11.png")

    img_warmup = img.copy()
    img_warmup = cv2.resize(img_warmup, (base_size, height_size))  # Resize image to 960*720
    image_warmup = img_warmup.astype(np.float32)[:, :, ::-1]
    image_warmup = image_warmup / 255.0
    image_warmup -= mean
    image_warmup /= std
    image_warmup = image_warmup.transpose((2, 0, 1))
    image_warmup = torch.from_numpy(image_warmup)
    image_warmup = image_warmup.unsqueeze(0)
    image_warmup = image_warmup.cuda()
    _ = model(image_warmup)

    imgor = img.copy()
    img = cv2.resize(img, (base_size, height_size))  # Resize image to 960*720
    image = img.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std

    image = image.transpose((2, 0, 1))
    image1 = image.copy()
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.cuda()

    out = model(image)
    out = out[0].squeeze(dim=0)
    outadd = F.softmax(out, dim=0)
    outadd = torch.argmax(outadd, dim=0)
    predadd = outadd.detach().cpu().numpy()
    pred = np.int32(predadd)

    pred_color = colorEncode(pred, fixed_colors).astype(np.uint8)
    pred_color = cv2.resize(pred_color, (imgor.shape[1], imgor.shape[0]))
    #im_vis = cv2.addWeighted(imgor, 0.7, pred_color, 0.3, 0)
    cv2.imwrite("results.jpg", pred_color) #or im_vis

if __name__ == '__main__':
    main()