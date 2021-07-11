import torch
import torch.onnx
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize

import matplotlib.pyplot as plt
import onnxruntime as rt
from PIL import Image
import numpy as np
import time
import onnx
import cv2
import os


def compare_outputs(onnx_model_path: str, pytorch_model_path: str):
    # ----- Compare outputs of Pytorch vs Onnx -----

    test_dir_path = "H:\Projects\SpaceNet2\Images"

    image_path = os.path.join(
        test_dir_path, "SN6_Train_AOI_11_Rotterdam_PS-RGB_20190823091132_20190823091448_tile_7924.tif")
    image = Image.open(image_path)
    image = np.array(image)
    image = image[:512, :512, :]

    # show picture
    plt.imshow(image)
    plt.show()

    # Pytorch ToTensor()
    image = image.transpose((2, 0, 1))
    image = image / 255

    # Pytorch Normalize()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for channel in range(image.shape[0]):
        image[channel, :, :] = (image[channel, :, :] -
                                mean[channel]) / std[channel]

    image = np.array([image])
    print(image.shape)
    print(image[0, 0, :10, 0])

    # ----- Run Onnx model

    sess = rt.InferenceSession(onnx_model_path)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    print(input_name)
    print(label_name)

    pred = sess.run([label_name], {input_name: image.astype(np.float32)})[0]
    output = pred.squeeze(0)

    mask = output.argmax(axis=0).astype(np.uint8)
    print("mask shape: ", mask.shape)


def convert_model(pytorch_model: nn.Module, onnx_save_path: str):
    '''
    Converts pytorch model to onnx format

    \param pytorch_model_path: path to the trained pyrorch model
    \param onnx_save_path: onnx model save path
    '''

    pass


def convert_esrgan_model(pytorch_model_path: str, onnx_save_path: str, device: str = 'cpu'):
    from core.archs.rrdbnet_arch import RRDBNet
    '''
    Converts Esrgan
    '''

    rrdb = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32)
    state = torch.load(pytorch_model_path, map_location=device)
    rrdb.load_state_dict(state["params"])
    rrdb.eval()
    rrdb = rrdb.to(device)

    # ----- Convert -----
    IM_SZ = 84
    channel = 3
    height = width = IM_SZ
    dummy_input = torch.randn(1, channel, height, width)

    # no need for grad during inference
    with torch.no_grad():
        torch.onnx.export(rrdb, dummy_input,
                          onnx_save_path, opset_version=11)

    # Check ONNX model
    onnx_model = onnx.load(onnx_save_path)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)


def test_esrgan_model(pytorch_model_path: str, onnx_model_path: str, result_dirname: str, device: str = 'cpu'):
    from core.archs.rrdbnet_arch import RRDBNet
    # read image
    image_path = r"H:\Datasets\increase-image-resolution-using-superresolution\dataset_30sm\val\low_res\000014809.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    #
    # Test pytorch model
    #

    torch_model = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32)
    torch_model_state = torch.load(pytorch_model_path, map_location=device)
    torch_model.load_state_dict(torch_model_state["params"])
    torch_model.eval()
    torch_model = torch_model.to(device)

    img = torch.from_numpy(image).float()
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        t = time.time()
        pytorch_result = torch_model(img)
        print("Pytorch inference time: ", time.time() - t)

    pytorch_result = pytorch_result.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    pytorch_result = pytorch_result.transpose((1, 2, 0))
    pytorch_result = cv2.cvtColor(pytorch_result, cv2.COLOR_RGB2BGR)
    pytorch_result = (pytorch_result * 255.0).round().astype(np.uint8)

    #
    # Test onnx model
    #

    img = np.array([image])
    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    t = time.time()
    pred = sess.run([label_name], {input_name: img.astype(np.float32)})[0]
    print("Onnx inference time: ", time.time() - t)

    onnx_result = pred.squeeze(0)
    onnx_result = onnx_result.clip(0, 1)
    onnx_result = onnx_result.transpose((1, 2, 0))
    onnx_result = cv2.cvtColor(onnx_result, cv2.COLOR_RGB2BGR)
    onnx_result = (onnx_result * 255.0).round().astype(np.uint8)

    #
    # Save and check results
    #

    # save original image
    cv2.imwrite(os.path.join(result_dirname, "Input_LR_image_ESRGAN.png"),
                cv2.imread(image_path, cv2.IMREAD_COLOR))
    # save pytorch result
    cv2.imwrite(os.path.join(result_dirname,
                "Pytorch_ESRGAN.png"), pytorch_result)
    # save pytorch result
    cv2.imwrite(os.path.join(result_dirname,
                "Onnx_ESRGAN.png"), onnx_result)

    np.testing.assert_array_almost_equal_nulp(pytorch_result, onnx_result, 4)


#  crt_net_path: str
def convert_edsr_model(pytorch_model_path: str, onnx_save_path: str, device: str = 'cpu', num_block=32):
    from core.archs.edsr_arch import EDSR
    """ Converts EDSR model
    Args:
        pytorch_model_path (str): Current network path.
    """

    net = EDSR(3, 3, num_feat=256, num_block=32, upscale=4, res_scale=0.1,
               img_range=255., channels_mean=[0.4488, 0.4371, 0.4040])
    net_state = torch.load(pytorch_model_path, map_location=device)['params']
    net.load_state_dict(net_state)
    net.eval()
    net = net.to(device)

    # ----- Convert -----
    IM_SZ = 84
    channel = 3
    height = width = IM_SZ
    dummy_input = torch.randn(1, channel, height, width)

    # no need for grad during inference
    with torch.no_grad():
        torch.onnx.export(net, dummy_input,
                          onnx_save_path, opset_version=11)

    # Check ONNX model
    onnx_model = onnx.load(onnx_save_path)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)


def test_edsr_model(pytorch_model_path: str, onnx_model_path: str, result_dirname: str, device: str = 'cpu'):
    from core.archs.edsr_arch import EDSR
    # read image
    image_path = r"H:\Datasets\increase-image-resolution-using-superresolution\dataset_30sm\val\low_res\000014809.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    #
    # Test pytorch model
    #

    net = EDSR(3, 3, num_feat=256, num_block=32, upscale=4, res_scale=0.1,
               img_range=255., channels_mean=[0.4488, 0.4371, 0.4040])
    net_state = torch.load(pytorch_model_path, map_location=device)
    net.load_state_dict(net_state["params"])
    # net.load_state_dict(pytorch_model_path)
    net.eval()
    net = net.to(device)

    img = torch.from_numpy(image).float()
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        t = time.time()
        pytorch_result = net(img)
        print("Pytorch inference time: ", time.time() - t)

    pytorch_result = pytorch_result.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    pytorch_result = pytorch_result.transpose((1, 2, 0))
    pytorch_result = cv2.cvtColor(pytorch_result, cv2.COLOR_RGB2BGR)
    pytorch_result = (pytorch_result * 255.0).round().astype(np.uint8)

    #
    # Test onnx model
    #

    img = np.array([image])
    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    t = time.time()
    pred = sess.run([label_name], {input_name: img.astype(np.float32)})[0]
    print("Onnx inference time: ", time.time() - t)

    onnx_result = pred.squeeze(0)
    onnx_result = onnx_result.clip(0, 1)
    onnx_result = onnx_result.transpose((1, 2, 0))
    onnx_result = cv2.cvtColor(onnx_result, cv2.COLOR_RGB2BGR)
    onnx_result = (onnx_result * 255.0).round().astype(np.uint8)

    #
    # Save and check results
    #

    # save original image
    cv2.imwrite(os.path.join(result_dirname, "Input_LR_image_EDSR.png"),
                cv2.imread(image_path, cv2.IMREAD_COLOR))
    # save pytorch result
    cv2.imwrite(os.path.join(result_dirname,
                "Pytorch_EDSR.png"), pytorch_result)
    # save pytorch result
    cv2.imwrite(os.path.join(result_dirname,
                "Onnx_EDSR.png"), onnx_result)

    np.testing.assert_array_almost_equal_nulp(pytorch_result, onnx_result, 4)


if __name__ == '__main__':
    device = 'cpu'
    result_dirname = "scripts/model_conversion/results"
    if not os.path.exists(result_dirname):
        os.makedirs(result_dirname)

    # ESRGAN

    # pytorch_model_path = "experiments/053_ESRGAN_x4_f64b23_Argis_B16G1_053_not_pretrained/models/net_g_80000.pth"
    # onnx_save_path = "experiments/onnx_models/esrgan_x4_Argis_30sm.onnx"
    # convert_esrgan_model(pytorch_model_path, onnx_save_path, device)
    # test_esrgan_model(pytorch_model_path, onnx_save_path,
    #                   result_dirname, device)

    # EDSR
    pytorch_model_path = "experiments/16_EDSR_x4_30sm_Argis_pretrained/models/net_g_155000.pth"
    onnx_save_path = "experiments/onnx_models/edsr_x4_Argis_30sm.onnx"
    convert_edsr_model(pytorch_model_path, onnx_save_path, device)
    test_edsr_model(pytorch_model_path, onnx_save_path,
                    result_dirname, device)
