
import os
import sys
import torch
import torchvision
import argparse
import torch.utils.data as Data
from PIL import Image
import models.networks as nets
import torchvision.utils as vutils
import os.path as osp
import numpy as np
import cv2
from modules.keypoint_detector_heatmap import KPDetector
from util import visualizer_kp
import imageio
from modules.util import make_coordinate_grid
from modules.keypoint_detector_strong import KPDetector_strong



def load_model(load_path, G_A, R_A, R_B, KP):
    state = torch.load(load_path)
    G_A.load_state_dict(state['G_A'])
    R_A.load_state_dict(state['R_A'])
    R_B.load_state_dict(state['R_B'])
    KP.load_state_dict(state['KP'])

    return state['epoch'], state['learned_t']


def norm(var):
    var = var.cpu().detach()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    return var

def vis_points(viz, img, kpoints):
    source = norm(img.data)
    kp_source = kpoints.data.cpu().numpy()
    source = np.transpose(source, [0, 2, 3, 1])

    return viz.create_image_column_with_kp(source, kp_source)


def transform_kp(coordinates, theta, bs):
    theta = theta.repeat(bs, 1, 1)

    theta = theta.unsqueeze(1)
    transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
    transformed = transformed.squeeze(-1)

    return transformed


def inverse_transform_kp(coordinates, theta, bs):

    inverse = torch.inverse(theta[:, :, :2])
    theta = theta.repeat(bs, 1, 1)
    theta = theta.unsqueeze(1)
    inverse = inverse.repeat(bs, 1, 1)
    inverse = inverse.unsqueeze(1)
    transformed = coordinates.unsqueeze(-1) - theta[:, :, :, 2:]
    transformed = torch.matmul(inverse, transformed)
    transformed = transformed.squeeze(-1)

    return transformed


def augment(path, seg_path, pad=True, pad_factor=0.2):

    img = cv2.imread(path)

    if pad:
        seg = cv2.imread(seg_path)
        w ,h = img.shape[0], img.shape[1]
        img = cv2.copyMakeBorder(img, int(w * pad_factor), int(w * pad_factor), int(h * pad_factor), int(h * pad_factor), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        seg = cv2.copyMakeBorder(seg, int(w * pad_factor), int(w * pad_factor), int(h * pad_factor),
                                 int(h * pad_factor), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        seg = seg[:, :, :1]
    else:
        seg = cv2.imread(seg_path)[:, :, :1]

    seg = np.stack((seg[:,:,0],)*3, axis=-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    seg_pil = Image.fromarray(seg)

    return im_pil, seg_pil


class Aug_dataset(Data.Dataset):
    def __init__(self, root_seg, transform, transform_seg, args, train=True, hflip=False, ext='.jpg', prefix='', pad_factor=0.2):
        self.transform = transform
        self.transform_seg = transform_seg
        self.hflip = hflip
        self.train = train
        self.ext = ext
        self.prefix = prefix
        self.pad_factor = pad_factor

        self.root_dir = root_seg.replace("_seg", "")
        self.seg_dir = root_seg

        dir_imgs = [f for f in os.listdir(self.seg_dir) if f.endswith(self.ext)]

        for i in range(0, len(dir_imgs)):
            if os.path.isfile(os.path.join(self.seg_dir,self.prefix + "%0d%s" % (i, self.ext))):
                start = i
                break

        assert start >= 0
        print("start " + str(start))

        self.imgs = [self.prefix +  "%0d%s" % (i,self.ext) for i in range(start, start+len(dir_imgs))]

        if args.data_size > 0:
            self.size = args.data_size
        else:
            self.size = len(self.imgs)

        self.real_size = len(self.imgs)
        print("Data size is " + str(self.size))


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        name1 = self.imgs[idx % self.real_size]
        img1_path = os.path.join(self.root_dir, name1)
        seg1_path = os.path.join(self.seg_dir, name1)

        print(img1_path, seg1_path)

        img1, seg1 = augment(img1_path, seg1_path, pad_factor=self.pad_factor)
        seg1 = seg1.convert('RGB')

        if self.hflip and self.train:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            seg1 = seg1.transpose(Image.FLIP_LEFT_RIGHT)

        img1 = self.transform(img1)
        seg1 = self.transform_seg(seg1)

        seg1 = (seg1 > 0.5).float()

        img1 = img1 * seg1

        return img1, seg1, name1


def kp_to_heatmap(x, spatial_size=256, std=0.2):
    """

    :param kp: bs X num_kp X 2
    :param spatial_size: int
    :param std: float
    :return: bs X num_kp X spatial_size X spatial_size
    """
    kp = x.unsqueeze(2).unsqueeze(2)
    #print(kp.size())
    ss = spatial_size
    bs, num_kp = kp.size(0),  kp.size(1)
    grid = make_coordinate_grid((ss, ss), torch.float).unsqueeze(0).unsqueeze(0).repeat(bs, num_kp,1 ,1,1).cuda()  # Range -1, 1
    #kp = (kp / float(ss)) * 2 - 1
    #print(kp.size())
    #print(grid.size())
    y = torch.abs(grid - kp)
    y = torch.exp(-y / (std ** 2))
    z = y[:, :, :, :, 0] * y[:, :, :, :, 1]
    z = z / torch.max(z)

    assert bs == z.size(0) and num_kp == z.size(1) and ss == z.size(2) and ss == z.size(3)

    return z

def resize(img, w, h):
   img_PIL = torchvision.transforms.ToPILImage()(img[0])
   img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
   new_img = torchvision.transforms.ToTensor()(img_PIL)
   new_img = new_img.unsqueeze(0)
   return new_img


def eval(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    tran_list = []

    tran_list.append(torchvision.transforms.Resize((args.resize_w, args.resize_h)))

    tran_list.append(torchvision.transforms.ToTensor())
    tran_list.append(torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = torchvision.transforms.Compose(tran_list)
    transform_seg = torchvision.transforms.Compose(tran_list[:-1])


    dataset_a = Aug_dataset(args.root_a, transform, transform_seg, args, hflip=args.hflip, ext=args.ext_a, prefix=args.prefix_a, pad_factor=args.pad_factor_a)
    data_loader_a = torch.utils.data.DataLoader(dataset_a, shuffle=False, batch_size=args.bs, drop_last=False)

    dataset_b = Aug_dataset(args.root_b, transform, transform_seg, args, ext=args.ext_b, prefix=args.prefix_b, pad_factor=args.pad_factor_b)
    data_loader_b = torch.utils.data.DataLoader(dataset_b, shuffle=False, batch_size=args.bs, drop_last=False)

    if not args.strong_kp:
        KP = KPDetector(block_expansion=32, num_kp=args.num_kp, num_channels=3, max_features=1024,
                 num_blocks=5, temperature=0.1, estimate_jacobian=False, scale_factor=args.scale_kp)
    else:
        KP = KPDetector_strong(block_expansion=32, num_kp=args.num_kp, num_channels=3, max_features=1024,
                 num_blocks=5, temperature=0.1, estimate_jacobian=False, scale_factor=args.scale_kp, args=args)

    ch_size =args.num_kp

    G_A = nets.define_G(ch_size, 2, args.ngf, args.netG, args.norm,
                        not args.no_dropout, args.init_type, args.init_gain, not_mask=False, only_mask=True)

    R_A = nets.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.new_norm,
                        not args.no_dropout, args.init_type, args.init_gain)

    R_B = nets.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.new_norm,
                        not args.no_dropout, args.init_type, args.init_gain)

    G_A = G_A.cuda()
    KP = KP.cuda()
    R_A = R_A.cuda()
    R_B = R_B.cuda()

    viz = visualizer_kp.Visualizer(kp_size=args.num_kp)

    load_epoch, learned_t = load_model(args.load, G_A, R_A, R_B, KP)
    print("Loaded successfully, epoch=" + str(load_epoch))

    KP = KP.train()
    G_A = G_A.eval()
    R_A = R_A.eval()
    R_B = R_B.eval()

    iter_cnt = 0

    print('Started Inference...')

    print(f'lengths: {len(dataset_a), len(dataset_b)}')
    for data_a, data_b in zip(data_loader_a, data_loader_b):

        print(iter_cnt)

        img_a, seg_a, name_a = data_a
        img_b, seg_b, name_b = data_b

        img_a = img_a.cuda()
        img_b = img_b.cuda()

        seg_a = seg_a.cuda()
        seg_b = seg_b.cuda()

        with torch.no_grad():

            kpoints_a, heatmap_a = KP(img_a)
            new_heatmap_a = kp_to_heatmap(kpoints_a, img_a.size(-1))

            kpoints_b, heatmap_b = KP(img_b)
            new_heatmap_b = kp_to_heatmap(kpoints_b, img_b.size(-1))


            if args.affine:
                kpoints_b_transformed = transform_kp(kpoints_b, learned_t, args.bs)
                kpoints_a_transformed = inverse_transform_kp(kpoints_a, learned_t, args.bs)
                new_heatmap_b_transformed = kp_to_heatmap(kpoints_b_transformed, img_b.size(-1))
                new_heatmap_a_transformed = kp_to_heatmap(kpoints_a_transformed, img_a.size(-1))

                decoded_a, _ = G_A(new_heatmap_a)
                _, decoded_b = G_A(new_heatmap_b)

                new_heatmap_a = new_heatmap_a_transformed
                new_heatmap_b = new_heatmap_b_transformed

                _, decoded_ab = G_A(new_heatmap_a)
                decoded_ba, _ = G_A(new_heatmap_b)
            else:
                decoded_a, decoded_ab = G_A(new_heatmap_a)
                decoded_ba, decoded_b = G_A(new_heatmap_b)

            refined_a = R_A(decoded_a)
            refined_b = R_B(decoded_b)

            refined_ab = R_B(decoded_ab)
            refined_ba = R_A(decoded_ba)

        if not args.splitted:

            exps = torch.cat([img_a, seg_a, decoded_a, refined_a], 0)
            vutils.save_image(exps, osp.join(args.out, "recon_a_" + str(name_a[0])), normalize=True, nrow=args.bs)

            exps = torch.cat([img_b, seg_b, decoded_b, refined_b], 0)
            vutils.save_image(exps, osp.join(args.out, "recon_b_" + str(name_b[0])), normalize=True, nrow=args.bs)

            exps = torch.cat([img_a, refined_ab], 0)
            vutils.save_image(exps, osp.join(args.out, "bab_" + str(name_a[0])), normalize=True)

            exps = torch.cat([img_b, refined_ba], 0)
            vutils.save_image(exps, osp.join(args.out, "aba_" + str(name_b[0])), normalize=True)

            to_print = []
            to_print.append(vis_points(viz, img_a, kpoints_a))
            to_print.append(vis_points(viz, img_b, kpoints_b))
            if args.affine:
                to_print.append(vis_points(viz, img_a, kpoints_b_transformed))
                to_print.append((vis_points(viz, img_a, kpoints_b_transformed) + vis_points(viz, torch.zeros(img_a.size()), kpoints_b)) / 2)
                to_print.append(vis_points(viz, img_b, kpoints_a_transformed))
                to_print.append((vis_points(viz, img_b, kpoints_a_transformed) + vis_points(viz, torch.zeros(img_a.size()), kpoints_a)) / 2)

            to_print = np.concatenate(to_print, axis=1)
            to_print = (255 * to_print).astype(np.uint8)
            imageio.imsave(osp.join(args.out, "kp_" + str(name_a[0])), to_print)

            to_print = []
            to_print.append(vis_points(viz, img_a, kpoints_a))
            to_print.append(vis_points(viz, decoded_ab, kpoints_a))
            to_print.append(vis_points(viz, refined_ab, kpoints_a))
            to_print = np.concatenate(to_print, axis=1)
            to_print = (255 * to_print).astype(np.uint8)
            imageio.imsave(osp.join(args.out, "kp_test_a" + str(name_a[0])), to_print)

            to_print = []
            to_print.append(vis_points(viz, img_b, kpoints_b))
            to_print.append(vis_points(viz, decoded_ba, kpoints_b))
            to_print.append(vis_points(viz, refined_ba, kpoints_b))
            to_print = np.concatenate(to_print, axis=1)
            to_print = (255 * to_print).astype(np.uint8)
            imageio.imsave(osp.join(args.out, "kp_test_b" + str(name_b[0])), to_print)
        else:
            if args.w > 0 and args.h > 0:
                img_a = resize(norm(img_a), args.w, args.h)
                seg_a = resize(seg_a, args.w, args.h)
                img_b = resize(norm(img_b), args.w, args.h)
                seg_b = resize(seg_b, args.w, args.h)
                refined_ab = resize(norm(refined_ab), args.w, args.h)
                refined_ba = resize(norm(refined_ba), args.w, args.h)
                decoded_ba = resize(decoded_ba, args.w, args.h)
                decoded_ab = resize(decoded_ab, args.w, args.h)
            else:
                img_a = norm(img_a)
                img_b = norm(img_b)
                refined_ab = norm(refined_ab)
                refined_ba = norm(refined_ba)

            vutils.save_image(img_a, osp.join(args.out, "a_" + str(name_a[0])), normalize=False)

            vutils.save_image(seg_a, osp.join(args.out, "seg_a_" + str(name_a[0])), normalize=False)

            vutils.save_image(refined_ab, osp.join(args.out, "refined_ab_" + str(name_a[0])), normalize=False)
            vutils.save_image(decoded_ab, osp.join(args.out, "decoded_ab_" + str(name_a[0])), normalize=True)

            vutils.save_image(img_b, osp.join(args.out, "b_" + str(name_b[0])), normalize=False)

            vutils.save_image(seg_b, osp.join(args.out, "seg_b_" + str(name_b[0])), normalize=False)

            vutils.save_image(refined_ba, osp.join(args.out, "refined_ba_" + str(name_b[0])), normalize=False)
            vutils.save_image(decoded_ba, osp.join(args.out, "decoded_ba_" + str(name_b[0])), normalize=True)

        iter_cnt += 1

        print(name_a[0], name_b[0])

    print("Inference is done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=-1)
    parser.add_argument('--root_a', default='')
    parser.add_argument('--root_b', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--ext_a', default='.jpg')
    parser.add_argument('--ext_b', default='.jpg')
    parser.add_argument('--prefix_a', default='')
    parser.add_argument('--prefix_b', default='')
    parser.add_argument('--w', type=int, default=0)
    parser.add_argument('--h', type=int, default=0)
    parser.add_argument('--pad_factor_a', type=float, default=0.2)
    parser.add_argument('--pad_factor_b', type=float, default=0.2)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--resize_w', type=int, default=256)
    parser.add_argument('--resize_h', type=int, default=256)
    parser.add_argument('--num_kp', type=int, default=10)
    parser.add_argument('--scale_kp', type=float, default=0.25)
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netG', type=str, default='resnet_9blocks_double',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--new_norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--load', default='')
    parser.add_argument('--no_mask', type=bool, default=True)
    parser.add_argument('--hflip', dest='hflip', action='store_true')
    parser.add_argument('--no_hflip', dest='hflip', action='store_false')
    parser.set_defaults(hflip=False)
    parser.add_argument('--resize', dest='resize', action='store_true')
    parser.add_argument('--no_resize', dest='resize', action='store_false')
    parser.set_defaults(resize=False)

    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--bottleneck', type=int, default=512)

    parser.add_argument('--affine', dest='affine', action='store_true')
    parser.set_defaults(affine=False)

    parser.add_argument('--strong_kp', dest='strong_kp', action='store_true')
    parser.set_defaults(strong_kp=False)

    parser.add_argument('--splitted', dest='splitted', action='store_true')
    parser.set_defaults(splitted=False)

    args = parser.parse_args()

    eval(args)
