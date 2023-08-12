import matplotlib
import matplotlib.pyplot as plt
from detectron2_gradcam import Detectron2GradCAM

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from gradcam import GradCAM, GradCamPlusPlus


plt.rcParams["figure.figsize"] = (30,10)

img_path = "/content/drive/My Drive/Colab Notebooks/ys_multi_class_pod_data/real-world test dataset/real-world test dataset/218-1-5.png"
config_file =  '/content/drive/My Drive/Colab Notebooks/transfiner-opt/multi_pod_output_101_3x_14_14_seg_pre_train_cls_14_14/config.yaml' # model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
model_file = "/content/drive/My Drive/Colab Notebooks/transfiner-opt/multi_pod_output_101_3x_14_14_seg_pre_train_cls_14_14/model_0049999.pth"

config_list = [
"MODEL.WEIGHTS", model_file,
"MODEL.DEVICE", 'cpu'
]

# 这里可视化的是卷积层
layer_name = "backbone.bottom_up.res5.2.conv3"
instance = 25  #CAM is generated per object instance, not per class!


def main():
    cam_extractor = Detectron2GradCAM(config_file, config_list, img_path=img_path)
    grad_cam = GradCAM

    image_dict, cam_orig = cam_extractor.get_cam(target_instance=instance, layer_name=layer_name, grad_cam_instance=grad_cam)

    v = Visualizer(image_dict["image"], MetadataCatalog.get(cam_extractor.cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(image_dict["output"]["instances"][instance].to("cpu"))

    plt.imshow(out.get_image(), interpolation='none')
    plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)
    plt.title(f"CAM for Instance {instance} (class {image_dict['label']})")
    plt.savefig(f"instance_{instance}_cam.jpg", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
