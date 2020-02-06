import os
import argparse
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets.KITTI_DepthCompletion import KITTI_DepthCompletion
import Trainer

batch_size = 4
num_epochs = 100

kitti_dir = "./KITTI"
data_depth_annotated_dir = "./KITTI/data_dir/train/data_depth_annotated_train.txt"
data_depth_velodyne_dir = "./KITTI//data_dir/train/data_depth_velodyne_train.txt"
data_RGB_dir = "./KITTI/data_dir/data_RGB_train.txt"
size = (512, 512) #input and output size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = KITTI_DepthCompletion(kitti_dir, data_depth_annotated_dir, data_depth_velodyne_dir, data_RGB_dir, size, test=False)
dataloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size,
                shuffle=True, num_workers=4
             )

trainer = Trainer(device)
writer = SummaryWriter()

def log_img(image, semantic, it):
    input_img = vutils.make_grid(image)
    input_array = input_img.to('cpu').numpy()

    semantic = F.softmax(semantic, dim=1)
    semantic = torch.argmax(semantic, dim=1, keepdim=True)
    semantic = vutils.make_grid(semantic)
    semantic = semantic.permute(1, 2, 0)
    semantic_array = semantic.to('cpu').numpy()[:,:,0]
    semantic_img = utils.color_semantic_label(semantic_array)
    semantic_img = np.transpose(semantic_img, (2, 0, 1))

    writer.add_image('input', input_array, it)
    writer.add_image('semantic', semantic_img, it)

def main(args):
    if args.resume:
        assert os.path.isfile(args.resume), \
            "%s is not a file" % args.resume

        checkpoint = torch.load(args.resume)
        trainer.load(checkpoint)
        it = checkpoint["iterations"] + 1

        print("Loaded checkpoint '{}' (iterations {})"
            .format(args.resume, checkpoint['iterations']))
    else:
        it = 1

    for e in range(1, num_epochs):
        print("Epoch %d" % e)

        sum_loss = 0
        t = tqdm(dataloader)
        for i, data in enumerate(t):
            # Input
            depth_map_gt = data[0].to(device)
            lidar = data[1].to(device)
            rgb = data[2].to(device)

            # Train
            output = trainer.update(depth_map_gt, lidar, rgb)

            # Log
            metrics = trainer.get_metrics()
            for k, v in metrics.items():
                writer.add_scalar(k, v, it)

            #if it % 500 == 0:
            #log_img(rgb, output, it)

            t.set_postfix(loss=metrics['loss/total_loss']/batch_size)

            sum_loss += metrics['loss/total_loss']

            # Save
            if it % 1000 == 0:
                print('Saving checkpoint...')
                trainer.save(weight_dir, it)


            it += 1

        print('Average loss @ epoch: {}'.format((sum_loss / i * batch_size)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Load weight from file")

    args = parser.parse_args()

    main(args)

