

import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import torch
import torchvision



def torch_to_cv2_image(image):
    pil_image = torchvision.transforms.ToPILImage()(image)
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./')
    parser.add_argument('--name', type=str, default='vid.avi')
    parser.add_argument('--ext_a', type=str, default='.jpg')
    parser.add_argument('--ext_b', type=str, default='.jpg')
    parser.add_argument('--out', type=str, default='./')
    parser.add_argument('--prefix_a', type=str, default='a_')
    parser.add_argument('--prefix_b', type=str, default='b_')
    parser.add_argument('--prefix_c', type=str, default='a_')
    parser.add_argument('--prefix_d', type=str, default='b_')

    parser.add_argument('--start_a', type=int, default=0)
    parser.add_argument('--end_a', type=int, default=10)

    parser.add_argument('--start_b', type=int, default=0)
    parser.add_argument('--end_b', type=int, default=10)

    parser.add_argument('--stride_a', type=int, default=1)
    parser.add_argument('--stride_b', type=int, default=1)

    parser.add_argument('--w', type=int, default=256)
    parser.add_argument('--h', type=int, default=256)

    parser.add_argument('--fps', type=float, default=15.0)

    parser.add_argument('--resize', dest='resize', action='store_true')
    parser.set_defaults(resize=False)

    parser.add_argument('--same_length', dest='same_length', action='store_true')
    parser.set_defaults(same_length=False)

    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.set_defaults(crop=False)

    parser.add_argument('--crop_w', type=int, default=256)
    parser.add_argument('--crop_h', type=int, default=256)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    to_tensor = torchvision.transforms.ToTensor()

    if args.same_length:
        args.start_b = args.start_a
        args.end_b = args.end_a

    frame_array = []

    for i, j in zip(range(args.start_a, args.end_a, args.stride_a), range(args.start_b, args.end_b, args.stride_b)):

        print(i,j)

        name_a = os.path.join(args.img_path, "%s%0d%s" % (args.prefix_a, i, args.ext_a))
        name_b = os.path.join(args.img_path, "%s%0d%s" % (args.prefix_b, i, args.ext_a))

        name_c = os.path.join(args.img_path, "%s%0d%s" % (args.prefix_c, j, args.ext_b))
        name_d = os.path.join(args.img_path, "%s%0d%s" % (args.prefix_d, j, args.ext_b))


        try:
            img_a = Image.open(name_a)
            img_b = Image.open(name_b)
            img_c = Image.open(name_c)
            img_d = Image.open(name_d)
        except FileNotFoundError:
            print("ERROR! ")
            print(name_a, name_b, name_c, name_d)
            continue

        if args.crop:
            width, height = img_a.size  # Get dimensions

            left = (width - args.crop_w) / 2
            top = (height - args.crop_h) / 2
            right = (width + args.crop_w) / 2
            bottom = (height + args.crop_h) / 2

            img_a = img_a.crop((left, top, right, bottom))
            img_b = img_b.crop((left, top, right, bottom))

            img_c = img_c.crop((left, top, right, bottom))
            img_d = img_d.crop((left, top, right, bottom))

        if args.resize:
            img_a = img_a.resize((args.w, args.h))
            img_b = img_b.resize((args.w, args.h))

            img_c = img_c.resize((args.w, args.h))
            img_d = img_d.resize((args.w, args.h))

        img_a = to_tensor(img_a)
        img_b = to_tensor(img_b)
        img_c = to_tensor(img_c)
        img_d = to_tensor(img_d)

        frame1 = vutils.make_grid([img_a, img_b], normalize=False, nrow=1, pad_value=1, padding=0)
        frame2 = vutils.make_grid([img_c, img_d], normalize=False, nrow=1, pad_value=1, padding=0)

        frame = vutils.make_grid([frame1, frame2], normalize=False, pad_value=1, padding=24)
        frame = torch_to_cv2_image(frame)
        frame = cv2.copyMakeBorder(frame, 0, 0, 60, 0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

        cv2.putText(frame, 'Input', (15,int((frame.shape[0]) / 4 + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
        cv2.putText(frame, 'Ours', (15, int((frame.shape[0] * 3) / 4 )), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        frame_array.append(frame)

    height, width, layers = frame.shape
    size = (width, height)
    out = cv2.VideoWriter(os.path.join(args.out, "%0d_%s" % (int(args.fps), args.name)), cv2.VideoWriter_fourcc(*'DIVX'), args.fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()



