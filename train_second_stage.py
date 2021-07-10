
import os
import sys
import torch
import lpips
from torch import optim
import torchvision
import argparse
import torch.utils.data as Data
from PIL import Image
import models.networks as nets
import torchvision.utils as vutils
import os.path as osp
import random
import numpy as np
import imgaug.augmenters as iaa
import cv2
from modules.keypoint_detector_heatmap import KPDetector
from util import visualizer_kp
import imageio
#from modules.util import make_coordinate_grid
from torch import nn, autograd
from modules.keypoint_detector_strong import KPDetector_strong


def save_model(out_file, G_A, R_A, R_B, KP, g_opt2, learned_t, epoch):
    state = {
        'G_A': G_A.state_dict(),
        'R_A': R_A.state_dict(),
        'R_B': R_B.state_dict(),
        'KP': KP.state_dict(),
        'g_opt2': g_opt2.state_dict(),
        'learned_t': learned_t,
        'epoch': epoch
    }
    torch.save(state, out_file)
    return


def load_first_stage(load_path, G_A, KP):
    state = torch.load(load_path)
    G_A.load_state_dict(state['G_A'])
    KP.load_state_dict(state['KP'])
    return state['epoch'], state['learned_t']


def load_model(load_path, G_A, R_A, R_B, KP, g_opt2):
    state = torch.load(load_path)
    G_A.load_state_dict(state['G_A'])
    R_A.load_state_dict(state['R_A'])
    R_B.load_state_dict(state['R_B'])
    KP.load_state_dict(state['KP'])
    g_opt2.load_state_dict(state['g_opt2'])
    return state['epoch'], state['learned_t']

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def norm(var):
    var = var.cpu().detach()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    return var


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

def augment(path, path2, seg_path, seg_path2, aug, pad=True, pad_factor=0.2):

    img = cv2.imread(path)
    img2 = cv2.imread(path2)


    if pad:
        seg = cv2.imread(seg_path)
        seg2 = cv2.imread(seg_path2)
        w ,h = img.shape[0], img.shape[1]
        img = cv2.copyMakeBorder(img, int(w * pad_factor), int(w * pad_factor), int(h * pad_factor), int(h * pad_factor), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img2 = cv2.copyMakeBorder(img2, int(w * pad_factor), int(w * pad_factor), int(h * pad_factor), int(h * pad_factor), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        seg = cv2.copyMakeBorder(seg, int(w * pad_factor), int(w * pad_factor), int(h * pad_factor),
                                 int(h * pad_factor), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        seg2 = cv2.copyMakeBorder(seg2, int(w * pad_factor), int(w * pad_factor), int(h * pad_factor),
                                 int(h * pad_factor), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        seg = seg[:, :, :1]
        seg2 = seg2[:, :, :1]
    else:
        seg = cv2.imread(seg_path)[:, :, :1]
        seg2 = cv2.imread(seg_path2)[:, :, :1]


    aug_i = aug.to_deterministic()
    img, seg = aug_i(images=np.expand_dims(img, axis=0), segmentation_maps=np.expand_dims(seg, axis=0))
    img2, seg2 = aug_i(images=np.expand_dims(img2, axis=0), segmentation_maps=np.expand_dims(seg2, axis=0))

    img = img[0]
    seg = seg[0]
    img2 = img2[0]
    seg2 = seg2[0]

    seg = np.stack((seg[:,:,0],)*3, axis=-1)
    seg2 = np.stack((seg2[:, :, 0],) * 3, axis=-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    im_pil2 = Image.fromarray(img2)

    seg_pil = Image.fromarray(seg)
    seg_pil2 = Image.fromarray(seg2)

    return im_pil, im_pil2, seg_pil, seg_pil2



class Aug_dataset(Data.Dataset):

    def __init__(self, root_seg, transform, transform_seg, train=True, hflip=False, ext='.jpg', prefix='', pad_factor=0.2):
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
            if os.path.isfile(os.path.join(self.seg_dir, self.prefix + "%0d%s" % (i, self.ext))):
                start = i
                break

        assert start >= 0
        print("start " + str(start))
        end = len(dir_imgs)
        print("end " + str(end))
        self.imgs = [self.prefix + "%0d%s" % (i,self.ext) for i in range(start, start+end)]

        self.size = len(self.imgs) -1
        print("Data size is " + str(self.size))

        self.seq = iaa.Sequential([
            iaa.Resize((0.8, 1.1)),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)}),
            iaa.Affine(shear={"x": (-15, 15), "y": (-15, 15)}),
        ])

        self.seq2 = iaa.Sequential([
            #iaa.PerspectiveTransform(scale=(0.01, 0.05)),  # CHange max to 0.1? Could be bit excessive
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)}),
            iaa.Affine(shear={"x": (-20, 20), "y": (-20, 20)}),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Crop(percent=(0.0001, 0.05))
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        name1 = self.imgs[idx]
        name2 = self.imgs[idx+1]
        img1_path = os.path.join(self.root_dir, name1)
        img2_path = os.path.join(self.root_dir, name2)
        seg1_path = os.path.join(self.seg_dir, name1)
        seg2_path = os.path.join(self.seg_dir, name2)

        r_ = random.random()
        if r_ > 0.6:
            pair1, pair2, seg1, seg2 = augment(img1_path, img2_path, seg1_path, seg2_path, self.seq2, pad_factor=self.pad_factor)
        else:
            pair1, pair2, seg1, seg2 = augment(img1_path, img2_path, seg1_path, seg2_path, self.seq, pad_factor=self.pad_factor)

        seg1 = seg1.convert('RGB')
        seg2 = seg2.convert('RGB')

        if self.hflip and self.train:
            pair1 = pair1.transpose(Image.FLIP_LEFT_RIGHT)
            seg1 = seg1.transpose(Image.FLIP_LEFT_RIGHT)
            pair2 = pair2.transpose(Image.FLIP_LEFT_RIGHT)
            seg2 = seg2.transpose(Image.FLIP_LEFT_RIGHT)

        pair1 = self.transform(pair1)
        pair2 = self.transform(pair2)
        seg1 = self.transform_seg(seg1)
        seg2 = self.transform_seg(seg2)

        seg1 = (seg1 > 0.5).float()
        seg2 = (seg2 > 0.5).float()

        pair1 = pair1 * seg1
        pair2 = pair2 * seg2

        return pair1, pair2, seg1, seg2


def vis_points(viz, img, kpoints):
    source = norm(img.data)
    kp_source = kpoints.data.cpu().numpy()
    source = np.transpose(source, [0, 2, 3, 1])

    return viz.create_image_column_with_kp(source, kp_source)




def train(args, opt=None):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(sys.argv)

    tran_list = []

    tran_list.append(torchvision.transforms.Resize((args.resize_w, args.resize_h)))

    tran_list.append(torchvision.transforms.ToTensor())
    tran_list.append(torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = torchvision.transforms.Compose(tran_list)
    transform_seg = torchvision.transforms.Compose(tran_list[:-1])

    dataset_a = Aug_dataset(args.root_a, transform, transform_seg, hflip=args.hflip, ext=args.ext_a, prefix=args.prefix_a, pad_factor=args.pad_factor_a)
    data_loader_a = torch.utils.data.DataLoader(dataset_a, shuffle=True, batch_size=args.bs, drop_last=True)

    dataset_b = Aug_dataset(args.root_b, transform, transform_seg, ext=args.ext_b, prefix=args.prefix_b, pad_factor=args.pad_factor_b)
    data_loader_b = torch.utils.data.DataLoader(dataset_b, shuffle=True, batch_size=args.bs, drop_last=True)

    if not args.strong_kp:
        KP = KPDetector(block_expansion=32, num_kp=args.num_kp, num_channels=3, max_features=1024,
                 num_blocks=5, temperature=0.1, estimate_jacobian=False, scale_factor=args.scale_kp)
    else:
        KP = KPDetector_strong(block_expansion=32, num_kp=args.num_kp, num_channels=3, max_features=1024,
                 num_blocks=5, temperature=0.1, estimate_jacobian=False, scale_factor=args.scale_kp, args=args)

    KP.requires_grad_(False)

    ch_size =args.num_kp

    G_A = nets.define_G(ch_size, 2, args.ngf, args.netG, args.norm,
                        not args.no_dropout, args.init_type, args.init_gain, not_mask=False, only_mask=True)
    G_A.requires_grad_(False)
    R_A = nets.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.new_norm,
                        not args.no_dropout, args.init_type, args.init_gain)

    R_B = nets.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.new_norm,
                        not args.no_dropout, args.init_type, args.init_gain)


    l1 = nn.L1Loss().cuda()
    #l2 = nn.MSELoss().cuda()
    criterionVGG = lpips.LPIPS(net='vgg').cuda()
    criterionVGG.requires_grad_(False)

    G_A = G_A.cuda()
    KP = KP.cuda()
    R_A = R_A.cuda()
    R_B = R_B.cuda()

    g_params2 = list(R_A.parameters()) + list(R_B.parameters())
    g_opt2 = optim.Adam(g_params2, lr=args.g_lr, betas=(0.5, 0.999))

    viz = visualizer_kp.Visualizer(kp_size=args.num_kp)

    assert args.load != ''

    if args.load_second_stage:
        load_epoch, learned_t = load_model(args.load, G_A, G_B, R_A, R_B, Disc, KP, g_opt, d_opt, g_opt2)
        print("Loaded successfully, epoch=" + str(load_epoch))
    else:
        load_epoch, learned_t = load_first_stage(args.load, G_A, KP)


    KP = KP.train()
    G_A = G_A.eval()
    R_A = R_A.train()
    R_B = R_B.train()

    iter_cnt = 0

    print('Started training...')
    for epoch in range(0, args.epoch):

        if iter_cnt > args.iters:
            break

        for data_a, data_b in zip(data_loader_a, data_loader_b):

            img_a, pair_a, seg_a, seg_pair_a = data_a
            img_b, pair_b, seg_b, seg_pair_b = data_b

            img_a = img_a.cuda()
            img_b = img_b.cuda()

            seg_a = seg_a.cuda()
            seg_b = seg_b.cuda()

            with torch.no_grad():

                kpoints_a, heatmap_a = KP(img_a)
                kpoints_b, heatmap_b = KP(img_b)

                new_heatmap_a = kp_to_heatmap(kpoints_a, img_a.size(-1))
                new_heatmap_b = kp_to_heatmap(kpoints_b, img_b.size(-1))

                decoded_a, decoded_ab = G_A(new_heatmap_a)
                decoded_ba, decoded_b = G_A(new_heatmap_b)

            g_opt2.zero_grad()

            refined_a = R_A(decoded_a)
            refined_b = R_B(decoded_b)

            l1_rec = args.lambda_l1 * (l1(img_a, refined_a) + l1(img_b, refined_b))

            vgg_rec = args.lambda_vgg * (criterionVGG(img_a, refined_a).mean() + criterionVGG(img_b, refined_b).mean())

            loss_g = l1_rec + vgg_rec

            loss_g.backward()
            g_opt2.step()


            if iter_cnt % args.print_loss == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, iter_cnt))
                print('<<<< vgg_rec=%f <<<< l1_rec=%f' % (float(vgg_rec), float(l1_rec)))
                sys.stdout.flush()

            if iter_cnt % args.save_img == 0:
                print("Saving imgs")
                sys.stdout.flush()

                exps = torch.cat([seg_a, decoded_a, img_a, refined_a, seg_b, decoded_b, img_b, refined_b], 0)
                vutils.save_image(exps, osp.join(args.out, "reconstruction_" + str(iter_cnt) + ".png"), normalize=True, nrow=args.bs)

                to_print = []

                to_print.append(vis_points(viz, img_a, kpoints_a))
                to_print.append(vis_points(viz, img_b, kpoints_b))

                to_print = np.concatenate(to_print, axis=1)
                to_print = (255 * to_print).astype(np.uint8)

                imageio.imsave(osp.join(args.out, "%s-kp.png" % str(iter_cnt)), to_print)

            if iter_cnt % args.eval_test == 0:

                with torch.no_grad():
                    # kpoints_a, heatmap_a = KP(img_a)
                    # new_heatmap_a = kp_to_heatmap(kpoints_a, img_a.size(-1))
                    #
                    # kpoints_b, heatmap_b = KP(img_b)
                    # new_heatmap_b = kp_to_heatmap(kpoints_b, img_b.size(-1))

                    kpoints_b_transformed = transform_kp(kpoints_b, learned_t, args.bs)
                    kpoints_a_transformed = inverse_transform_kp(kpoints_a, learned_t, args.bs)
                    new_heatmap_b_transformed = kp_to_heatmap(kpoints_b_transformed, img_b.size(-1))
                    new_heatmap_a_transformed = kp_to_heatmap(kpoints_a_transformed, img_a.size(-1))

                    new_heatmap_a = new_heatmap_a_transformed
                    new_heatmap_b = new_heatmap_b_transformed

                    _, decoded_ab = G_A(new_heatmap_a)
                    decoded_ba, _ = G_A(new_heatmap_b)

                    refined_ab = R_B(decoded_ab)
                    refined_ba = R_A(decoded_ba)

                exps = torch.cat([img_a, decoded_ab, refined_ab, img_b, decoded_ba, refined_ba], 0)
                vutils.save_image(exps, osp.join(args.out, "test_" + str(iter_cnt) + ".png"), normalize=True, nrow=args.bs)

                to_print = []

                to_print.append(vis_points(viz, decoded_ab, kpoints_a))
                to_print.append(vis_points(viz, decoded_ba, kpoints_b))

                to_print = np.concatenate(to_print, axis=1)
                to_print = (255 * to_print).astype(np.uint8)

                imageio.imsave(osp.join(args.out, "%s-kp_test_.png" % str(iter_cnt)), to_print)

            if iter_cnt % args.save_check == 0:
                save_file = os.path.join(args.out, 'checkpoint_' + str(iter_cnt))
                save_model(save_file, G_A, R_A, R_B, KP, g_opt2, learned_t, epoch)
                print("Checkpoint saved")

            iter_cnt += 1
            if iter_cnt > args.iters:
                break


    print("Training is done")
    save_file = os.path.join(args.out, 'checkpoint')
    save_model(save_file, G_A, R_A, R_B, KP, g_opt2, learned_t, epoch)
    print("Final checkpoint saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_a', default='')
    parser.add_argument('--root_b', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--ext_a', default='.jpg')
    parser.add_argument('--ext_b', default='.jpg')
    parser.add_argument('--prefix_a', default='')
    parser.add_argument('--prefix_b', default='')
    parser.add_argument('--pad_factor_a', type=float, default=0.2)
    parser.add_argument('--pad_factor_b', type=float, default=0.2)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=200000)
    parser.add_argument('--iters', type=int, default=30001)
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
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--save_img', type=int, default=2500)
    parser.add_argument('--print_loss', type=int, default=200)
    parser.add_argument('--save_test_img', type=int, default=10)
    parser.add_argument('--eval_test', type=int, default=2500)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--lambda_sill', type=float, default=10.0)
    parser.add_argument('--save_check', type=int, default=15000)
    parser.add_argument('--ext', default='.jpg')

    parser.add_argument('--affine', dest='affine', action='store_true')
    parser.set_defaults(affine=False)

    parser.add_argument('--new_norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')

    parser.add_argument('--strong_kp', dest='strong_kp', action='store_true')
    parser.set_defaults(strong_kp=False)

    parser.add_argument('--load_second_stage', dest='load_second_stage', action='store_true')
    parser.set_defaults(load_second_stage=False)

    args = parser.parse_args()

    train(args)
