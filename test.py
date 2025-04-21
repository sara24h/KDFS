import os
import time
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import Dataset_hardfakevsreal  # Your custom dataset
from model.student.ResNet_sparse import ResNet_50_sparse_imagenet
from utils import utils, meter
from get_flops_and_params import get_flops_and_params

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch  # Should be 'ResNet_50'
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

    def dataload(self):
        _, self.val_loader = Dataset_hardfakevsreal.get_loaders(
            data_dir=self.dataset_dir,
            csv_file=os.path.join(self.dataset_dir, 'dataset.csv'),  # Adjust as needed
            train_batch_size=self.test_batch_size,  # Not used for testing
            eval_batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=False
        )
        print("Dataset has been loaded!")

    def build_model(self):
        print("==> Building student model..")
        print("Loading sparse student model")
        self.student = ResNet_50_sparse_imagenet()  # Initialize without Gumbel params for testing
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        self.student.load_state_dict(ckpt_student["student"])

    def test(self):
        if self.device == "cuda":
            self.student = self.student.cuda()

        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        self.student.eval()
        self.student.ticket = True
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                for images, targets in self.val_loader:
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits_student, _ = self.student(images)
                    prec1 = utils.get_accuracy(logits_student, targets, topk=(1,))[0]
                    n = images.size(0)
                    meter_top1.update(prec1.item(), n)

                    _tqdm.set_postfix(top1="{:.4f}".format(meter_top1.avg))
                    _tqdm.update(1)
                    time.sleep(0.01)

        print(
            f"[Test] Prec@1 {meter_top1.avg:.2f}"
        )

        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(args=self.args)
        print(
            f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, Params reduction: {Params_reduction:.2f}%"
        )
        print(
            f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, Flops reduction: {Flops_reduction:.2f}%"
        )

    def main(self):
        self.dataload()
        self.build_model()
        self.test()
