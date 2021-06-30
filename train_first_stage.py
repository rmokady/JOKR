
import os
import sys
import torch
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
from modules.keypoint_detector_strong import KPDetector_strong
from util import visualizer_kp
import imageio
#from modules.util import make_coordinate_grid
from torch.autograd import grad
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, autograd



class kp_disc(nn.Module):
    def __init__(self, kp_num, bottleneck=512):
        super(kp_disc, self).__init__()
        self.kp_num = kp_num
        self.bottleneck = bottleneck

        self.classify = nn.Sequential(
            nn.Linear(self.kp_num * 2, self.bottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.bottleneck, self.bottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.bottleneck, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, self.kp_num * 2)
        net = self.classify(net)
        net = net.view(-1)
        return net

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


def save_model(out_file, G_A, Disc, KP, g_opt, d_opt, learned_t, epoch):
    state = {
        'G_A': G_A.state_dict(),
        'Disc': Disc.state_dict(),
        'KP': KP.state_dict(),
        'g_opt': g_opt.state_dict(),
        'd_opt': d_opt.state_dict(),
        'learned_t': learned_t,
        'epoch': epoch
    }
    torch.save(state, out_file)
    return


def load_model(load_path, G_A, Disc, KP, g_opt, d_opt):
    state = torch.load(load_path)
    G_A.load_state_dict(state['G_A'])
    Disc.load_state_dict(state['Disc'])
    KP.load_state_dict(state['KP'])
    g_opt.load_state_dict(state['g_opt'])
    d_opt.load_state_dict(state['d_opt'])
    return state['epoch'], state['learned_t']



def norm(var):
    var = var.cpu().detach()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    return var


def equi_loss(frame, kp, kp_extractor):
    transform = Transform(frame.shape[0], sigma_affine=0.1, points_tps=None)
    transformed_frame = transform.transform_frame(frame)
    transformed_kp, _ = kp_extractor(transformed_frame)

    value = torch.abs(kp - transform.warp_coordinates(transformed_kp)).mean()
    return value, transformed_frame, transformed_kp


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

class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if sigma_tps and points_tps:
            self.tps = True
            self.control_points = make_coordinate_grid((points_tps, points_tps), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=sigma_tps * torch.ones([bs, 1, points_tps ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


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

        #Augmentations
        self.seq = iaa.Sequential([
            iaa.Resize((0.8, 1.1)),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)}),
            iaa.Affine(shear={"x": (-15, 15), "y": (-15, 15)}),
        ])

        self.seq2 = iaa.Sequential([
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


def separation_loss_p(x, delta):
    """Computes the separation loss.
    Args:
      xyz: [batch, num_kp, 3] Input keypoints.
      delta: A separation threshold. Incur 0 cost if the distance >= delta.
    Returns:
      The seperation loss.
    """

    bs = x.size(0)
    num_kp_p = x.size(1)
    t1_p = x.repeat(1, num_kp_p, 1)

    t2_p = x.repeat(1, 1, num_kp_p).view(t1_p.size())
    diffsq_p = (t1_p - t2_p) ** 2

    # -> [batch, num_kp ^ 2]
    lensqr_p = torch.sum(diffsq_p, dim=2)

    return torch.sum(torch.max(delta-lensqr_p, torch.zeros(lensqr_p.size()).float().cuda())) / (float(num_kp_p * bs * 2))


def sill_loss(heatmap, seg):

    hm_size = heatmap.size(2)

    with torch.no_grad():
        seg = torch.nn.functional.interpolate(seg, (hm_size,hm_size), mode='bilinear')
        seg = (seg > 0.5).float()

    mul = heatmap * seg[:,:1]
    sum = torch.sum(mul, dim=[2,3])
    log_ = -torch.log(sum + 1e-12)
    res = torch.mean(log_)

    return res


def vis_points(viz, img, kpoints):
    source = norm(img.data)
    kp_source = kpoints.data.cpu().numpy()
    source = np.transpose(source, [0, 2, 3, 1])

    return viz.create_image_column_with_kp(source, kp_source)


def temp_loss(kp1, kp2, alpha):

    kp_diff = torch.abs(kp2 - kp1)
    kp_diff = kp_diff ** 2
    kp_diff = torch.sqrt(torch.sum(kp_diff, dim=2))
    kp_diff = torch.mean(kp_diff, dim=1)

    return torch.sum(torch.max(alpha * kp_diff, torch.zeros(kp_diff.size()).float().cuda())) / (float(kp_diff.size(0)))


def train(args):
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

    ch_size = args.num_kp


    G_A = nets.define_G(ch_size,2, args.ngf, args.netG, args.norm,
                                    not args.no_dropout, args.init_type, args.init_gain, not_mask=False, only_mask=True)

    Disc = kp_disc(kp_num=args.num_kp)

    if args.affine:
        noise = torch.normal(mean=0, std=0.05 * torch.ones([1, 2, 3]))
        learned_t = noise + torch.eye(2, 3).view(1, 2, 3)
        learned_t = learned_t.cuda()
        learned_t = Variable(learned_t, requires_grad=True)
    else:
        learned_t = None

    #l1 = nn.L1Loss().cuda()
    l2 = nn.MSELoss().cuda()
    bce = nn.BCELoss().cuda()

    G_A = G_A.cuda()
    KP = KP.cuda()
    Disc = Disc.cuda()

    g_params = list(G_A.parameters()) + list(KP.parameters())
    if args.affine:
        g_params += [learned_t]
    g_opt = optim.Adam(g_params, lr=args.g_lr, betas=(0.5, 0.999))

    d_params = list(Disc.parameters())
    d_opt = optim.Adam(d_params, lr=args.d_lr, betas=(0.5, 0.999))

    viz = visualizer_kp.Visualizer(kp_size=args.num_kp)

    if args.load != '':
        load_epoch = load_model(args.load, G_A, Disc, KP, g_opt, d_opt)
        print("Loaded successfully, epoch=" + str(load_epoch))
    else:
        load_epoch = 0

    KP = KP.train()
    G_A = G_A.train()
    Disc = Disc.train()

    A_label = torch.full((args.bs,), 1.0).cuda()
    B_label = torch.full((args.bs,), 0.0).cuda()

    iter_cnt = 0

    print('Started training...')
    for epoch in range(load_epoch, args.epoch):

        for data_a, data_b in zip(data_loader_a, data_loader_b):

            img_a, pair_a, seg_a, seg_pair_a = data_a
            img_b, pair_b, seg_b, seg_pair_b = data_b

            img_a = img_a.cuda()
            img_b = img_b.cuda()

            seg_a = seg_a.cuda()
            seg_b = seg_b.cuda()

            pair_a = pair_a.cuda()
            pair_b = pair_b.cuda()

            #
            # Generators
            #

            g_opt.zero_grad()

            kp_a, heatmap_a = KP(torch.cat([img_a, pair_a], dim=0))
            kp_b, heatmap_b = KP(torch.cat([img_b, pair_b], dim=0))

            kpoints_a, kpoints_pair_a = kp_a[:args.bs], kp_a[args.bs:]
            kpoints_b, kpoints_pair_b = kp_b[:args.bs], kp_b[args.bs:]

            new_heatmap_a = kp_to_heatmap(kpoints_a, img_a.size(-1))
            new_heatmap_b = kp_to_heatmap(kpoints_b, img_b.size(-1))

            decoded_a, decoded_ab = G_A(new_heatmap_a)
            decoded_ba, decoded_b = G_A(new_heatmap_b)

            loss_sill = (sill_loss(heatmap_a[:args.bs], seg_a) + sill_loss(heatmap_b[:args.bs], seg_b)) * args.lambda_sill

            l2_rec = args.lambda_l2 * (l2(seg_a, decoded_a) + l2(seg_b, decoded_b))

            loss_eq_a, transformed_frame_a, transformed_kp_a = equi_loss(img_a, kpoints_a, KP)
            loss_eq_a *= args.lambda_eq

            loss_eq_b, transformed_frame_b, transformed_kp_b = equi_loss(img_b, kpoints_b, KP)
            loss_eq_b *= args.lambda_eq

            loss_sep = args.lambda_sep * (separation_loss_p(kpoints_a[:,:10], args.delta) + separation_loss_p(kpoints_b[:,:10], args.delta))

            preds_A = Disc(kpoints_a)
            if args.affine:
                kpoints_b_transformed = transform_kp(kpoints_b, learned_t, args.bs)
                preds_B = Disc(kpoints_b_transformed)
            else:
                preds_B = Disc(kpoints_b)

            loss_disc = args.lambda_disc * (bce(preds_A, A_label) + bce(preds_B, A_label))

            loss_pred_a = args.lambda_pred * temp_loss(kpoints_a, kpoints_pair_a, args.new_alpha)
            loss_pred_b = args.lambda_pred * temp_loss(kpoints_b, kpoints_pair_b, args.new_alpha)

            loss_g = l2_rec + loss_eq_a + loss_eq_b + loss_sep + loss_disc + loss_sill + loss_pred_a + loss_pred_b

            loss_g.backward()
            g_opt.step()


            #
            # Discriminator
            #

            d_opt.zero_grad()

            disc_A = Disc(kpoints_a.detach())
            if args.affine:
                disc_B = Disc(kpoints_b_transformed.detach())
            else:
                disc_B = Disc(kpoints_b.detach())

            loss_d_a = bce(disc_A, A_label)
            loss_d_b = bce(disc_B, B_label)

            loss_d = loss_d_a + loss_d_b

            loss_d.backward()

            d_opt.step()


            if iter_cnt % args.print_loss == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, iter_cnt))
                print('loss_pred_a=%f, loss_pred_b=%f' % (float(loss_pred_a), float(loss_pred_b)))
                print('<<< l2_rec=%f <<<< loss_eq_a=%f <<<< loss_eq_b=%f <<<< loss_sep=%f' % (float(l2_rec), float(loss_eq_a), float(loss_eq_b), float(loss_sep)))
                print('G: loss_disc=%f, loss_sill=%f' % (float(loss_disc), float(loss_sill)))
                print('D_confusion: loss_d_a=%f, loss_d_b=%f' % (float(loss_d_a), float(loss_d_b)))
                print('DEBUG: disc_A=%f, disc_B=%f' % (float(disc_A[0]), float(disc_B[0])))
                print(learned_t)
                sys.stdout.flush()

            if iter_cnt % args.save_img == 0:
                print("Saving imgs")
                sys.stdout.flush()

                exps = torch.cat([img_a, seg_a, decoded_a, img_b, seg_b, decoded_b], 0)
                vutils.save_image(exps, osp.join(args.out, "reconstruction_" + str(iter_cnt) + ".png"), normalize=True, nrow=args.bs)

                to_print = []

                to_print.append(vis_points(viz, img_a, kpoints_a))
                to_print.append(vis_points(viz, pair_a, kpoints_pair_a))
                to_print.append(vis_points(viz, transformed_frame_a, transformed_kp_a))
                to_print.append(vis_points(viz, img_b, kpoints_b))
                to_print.append(vis_points(viz, pair_b, kpoints_pair_b))
                to_print.append(vis_points(viz, transformed_frame_b, transformed_kp_b))
                if args.affine:
                    to_print.append(vis_points(viz, img_b, kpoints_b_transformed))
                    to_print.append((vis_points(viz, img_b, kpoints_b_transformed) + vis_points(viz, torch.zeros(img_b.size()), kpoints_b)) / 2)

                to_print = np.concatenate(to_print, axis=1)
                to_print = (255 * to_print).astype(np.uint8)
                imageio.imsave(osp.join(args.out, "%s-kp.png" % str(iter_cnt)), to_print)

                if args.affine:
                    with torch.no_grad():
                        kpoints_b_transformed = transform_kp(kpoints_b, learned_t, args.bs)
                        kpoints_a_transformed = inverse_transform_kp(kpoints_a, learned_t, args.bs)
                        new_heatmap_b_transformed = kp_to_heatmap(kpoints_b_transformed, img_b.size(-1))
                        new_heatmap_a_transformed = kp_to_heatmap(kpoints_a_transformed, img_a.size(-1))

                        new_heatmap_a = new_heatmap_a_transformed
                        new_heatmap_b = new_heatmap_b_transformed

                        _, decoded_ab = G_A(new_heatmap_a)
                        decoded_ba, _ = G_A(new_heatmap_b)


                exps = torch.cat([img_a, decoded_ab, img_b, decoded_ba], 0)
                vutils.save_image(exps, osp.join(args.out, "test_" + str(iter_cnt) + ".png"), normalize=True, nrow=args.bs)

                to_print = []
                to_print.append(vis_points(viz, decoded_ab, kpoints_a))
                to_print.append(vis_points(viz, decoded_ba, kpoints_b))
                to_print = np.concatenate(to_print, axis=1)
                to_print = (255 * to_print).astype(np.uint8)
                imageio.imsave(osp.join(args.out, "%s-kp_test.png" % str(iter_cnt)), to_print)

            if iter_cnt % args.save_check == 0:
                save_file = os.path.join(args.out, 'checkpoint_' + str(iter_cnt))
                save_model(save_file, G_A, Disc, KP, g_opt, d_opt, learned_t, epoch)
                print("Checkpoint saved")

            iter_cnt += 1

    print("Training is done")
    save_file = os.path.join(args.out, 'checkpoint')
    save_model(save_file, G_A, Disc, KP, g_opt, d_opt, learned_t, epoch)
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
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=45000)
    parser.add_argument('--resize_w', type=int, default=256)
    parser.add_argument('--resize_h', type=int, default=256)
    parser.add_argument('--num_kp', type=int, default=10)
    parser.add_argument('--scale_kp', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=0.1)
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
    parser.add_argument('--lambda_l2', type=float, default=20.0)
    parser.add_argument('--lambda_eq', type=float, default=2.0)
    parser.add_argument('--lambda_sep', type=float, default=1.0)
    parser.add_argument('--lambda_pred', type=float, default=10.0)
    parser.add_argument('--save_img', type=int, default=2500)
    parser.add_argument('--print_loss', type=int, default=200)
    parser.add_argument('--eval_test', type=int, default=2500)
    parser.add_argument('--lambda_disc', type=float, default=0.005)
    #parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--lambda_sill', type=float, default=10.0)
    parser.add_argument('--save_check', type=int, default=15000)

    #Add affine invariant to domain confusion
    parser.add_argument('--affine', dest='affine', action='store_true')
    parser.set_defaults(affine=False)

    parser.add_argument('--new_alpha', type=float, default=13.0)

    #Stronger kp extractor
    parser.add_argument('--strong_kp', dest='strong_kp', action='store_true')
    parser.set_defaults(strong_kp=False)

    args = parser.parse_args()

    train(args)
