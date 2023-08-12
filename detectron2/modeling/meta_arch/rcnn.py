# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
from torchvision import models, transforms
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

# mask rcnn的通用框架
# 你所看到的mask rcnn的整体执行代码都在这里了。

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    # 定义了rcnn的主要流程
    # 这里你所看到的就是在train函数中的model(data)的执行输出。输出是dict，loss.update实际上是triple的更新。
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], my_center_loss_fun=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        #
        # print('images size ' + str(len(images)))

        # backbone
        # back bone出来的结果就是特征图。 backbone是FPN架构
        # 如果要可视化特征图，直接看这里就好了。
        # 这里要把所有的image训练完成后，才能输出特征
        # notice： 这里的feature 是 [n,m,w,h]格式， n是batch图片数量，也就是输入的图片数量。 m通常就是256，输出通道。 w，h就是多尺度图像。
        features = self.backbone(images.tensor)
        # print('features keys:', features.keys())
        # print('features p2.shape:', features['p2'].shape)
        # print('features p3.shape:', features['p3'].shape)
        # print('features p4.shape:', features['p4'].shape)
        # print('features p5.shape:', features['p5'].shape)
        # print('features p6.shape:', features['p6'].shape)

        # 256个输出通道？ 能不能卷积一下。 1*1卷积。
        # 1*1 卷积，融合256个输出通道
        # my_conv1 = Conv2d(
        #     256,
        #     1,
        #     kernel_size=1,
        #     stride=1,
        #     bias=False
        # )
        # features_6_out = my_conv1(features['p2'].cpu().detach())
        # print('features_6_out shape', features_6_out.shape)
        #

        # 融合？
        # 输出这么个通道怎么处理呢？ why？
        # feature_map = features['p6'][1].squeeze(0).cpu() # 压缩成torch.Size([64, 55, 55])
        #
        # # 以下4行，通过双线性插值的方式改变保存图像的大小
        # feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
        #                                feature_map.shape[2])  # (1,64,55,55)
        # upsample = torch.nn.UpsamplingBilinear2d(size=(28, 28))  # 这里进行调整大小
        # feature_map = upsample(feature_map)
        # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        #
        # feature_map_num = feature_map.shape[0]  # 返回通道数
        # row_num = np.ceil(np.sqrt(feature_map_num))  # 8
        #
        # for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
        #     plt.figure()
        #     # plt.imshow(feature_map[index - 1].detach(), cmap='rgb')  # feature_map[0].shape=torch.Size([55, 55])
        #     # 将上行代码替换成，可显示彩色
        #     plt.imshow(transforms.ToPILImage()(feature_map[index - 1].detach()))  # feature_map[0].shape=torch.Size([55, 55])
        #     plt.axis('off')
        #     plt.savefig('/content/drive/My Drive/Colab Notebooks/feature_map_save/' + str(index) + ".png")

        # 一下代码先注释
        # feature_map = features_6_out[0]  # 压缩成torch.Size([64, 55, 55])
        #
        # # 以下4行，通过双线性插值的方式改变保存图像的大小
        # feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
        #                                feature_map.shape[2])  # (1,64,55,55)
        # upsample = torch.nn.UpsamplingBilinear2d(size=(184, 184))  # 这里进行调整大小
        # feature_map = upsample(feature_map)
        # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        #
        # feature_map_num = feature_map.shape[0]  # 返回通道数
        # row_num = np.ceil(np.sqrt(feature_map_num))  # 8
        #
        # for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
        #     plt.figure()
        #     # plt.imshow(feature_map[index - 1].detach(), cmap='rgb')  # feature_map[0].shape=torch.Size([55, 55])
        #     # 将上行代码替换成，可显示彩色
        #     plt.imshow(
        #         transforms.ToPILImage()(feature_map[index - 1]))  # feature_map[0].shape=torch.Size([55, 55])
        #     plt.axis('off')
        #     plt.savefig('/content/drive/My Drive/Colab Notebooks/feature_map_save/' + str(index) + ".png")

        # module2： RPN
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # 参数：特征图，
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, my_center_loss_fun)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        # center loss ??
        # print('detector_losses losses', repr(detector_losses))

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

       #  print('batched_inputs ==>', repr(batched_inputs))

        if isinstance (batched_inputs, List) and isinstance(batched_inputs[0], dict):
          #   print('a')
            images = self.preprocess_image(batched_inputs)
        else:
            # 这个分支是用于cam的，目前应该废弃了
         #   print('b')
            images = self.preprocess_image_cam(batched_inputs)

       #  print('images -->', repr(images.tensor.shape))

        features = self.backbone(images.tensor)

        # my_conv1 = Conv2d(
        #     256,
        #     1,
        #     kernel_size=1,
        #     stride=1,
        #     bias=False
        # )
        # features_6_out = my_conv1(features['p3'].cpu().detach())
        # print('features_6_out shape', features_6_out.shape)
        #
        # feature_map = features_6_out[0]  # 压缩成torch.Size([64, 55, 55])
        #
        # # 以下4行，通过双线性插值的方式改变保存图像的大小
        # feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
        #                                feature_map.shape[2])  # (1,64,55,55)
        # # upsample = torch.nn.UpsamplingBilinear2d(size=(200, 200))  # 这里进行调整大小
        # # feature_map = upsample(feature_map)
        # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        #
        # feature_map_num = feature_map.shape[0]  # 返回通道数
        # row_num = np.ceil(np.sqrt(feature_map_num))  # 8
        #
        # for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
        #     plt.figure()
        #     # plt.imshow(feature_map[index - 1].detach(), cmap='rgb')  # feature_map[0].shape=torch.Size([55, 55])
        #     # 将上行代码替换成，可显示彩色
        #     plt.imshow(
        #         transforms.ToPILImage()(feature_map[index - 1]))  # feature_map[0].shape=torch.Size([55, 55])
        #     plt.axis('off')
        #     plt.savefig('/content/drive/My Drive/Colab Notebooks/feature_map_save/' + str(index) + ".png")

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_image_cam(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [batched_inputs["image"].to(self.device)]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


    def show_feature_map(self, feature_map):
        # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
        # feature_map[2].shape     out of bounds

        feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

        # 以下4行，通过双线性插值的方式改变保存图像的大小
        feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
                                       feature_map.shape[2])  # (1,64,55,55)
        upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
        feature_map = upsample(feature_map)
        feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

        feature_map_num = feature_map.shape[0]  # 返回通道数
        row_num = np.ceil(np.sqrt(feature_map_num))  # 8
        plt.figure()
        for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
            plt.subplot(row_num, row_num, index)
            plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
            # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
            plt.axis('off')
            scipy.misc.imsave('feature_map_save//' + str(index) + ".png", feature_map[index - 1])
        plt.show()


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
