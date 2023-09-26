import time
import keyboard
import numpy as np
import open3d as o3d
import cv2
import open3d.cpu.pybind.core
from sklearn.preprocessing import normalize
from PIL import Image, ImageEnhance, ImageOps

from reconstruction import Reconstruction
from config.reconstruction_config import ReconstructionConfig
from video_pre_processing import VideoPreProcessing
from logger.logger import Logger

log = Logger('main log')

def StereoSGBM_func(frame):
    # Вариант исполнения получения кадра глубины
    h = 0
    w = 0
    # Ширина и высота кадра
    x = 0
    x1 = 0
    y = 0
    # Отступы для первого кадра x y и для второго x1 y


    block_size = 3
    min_disp = -176
    max_disp = 176
    num_disp = max_disp - min_disp
    uniquenessRatio = 15
    speckleWindowSize = 16
    speckleRange = 2
    disp12MaxDiff = 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    # stereo = cv2.StereoSGBM_create()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    h = int(height)
    w = int(width / 2)
    x1 = w
    left_img = gray[y:y + h, x:x + w]
    right_img = gray[y:y + h, x1:x1 + w]
    disparity_SGBM = stereo.compute(left_img, right_img)
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=12, beta=10, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    disparity_SGBM = np.ascontiguousarray(np.uint16(disparity_SGBM[:, 176:disparity_SGBM.shape[1]-176]) * 256)
    left_img = np.ascontiguousarray(left_img[:, 176:left_img.shape[1]-176])
    # cv2.imshow('', disparity_SGBM)
    # cv2.waitKey(0)
    return left_img, disparity_SGBM



def StereoBM(left_img, right_img):
    # Вариант исполнения получения кадра глубины
    stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)

    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_img, right_img)
    # Normalize the values to a range from 0..255 for a grayscale image
    #disparity = cv2.normalize(disparity, disparity, alpha=125, beta=0, norm_type=cv2.NORM_MINMAX)

    disparity = np.uint16(disparity) * 256
    # cv2.imshow('', disparity)
    # cv2.waitKey(0)
    return disparity


def get_depth_map(frame):
    # Загрузка изображений

    # Преобразование изображений в градации серого

    left = frame[:, :frame.shape[1] // 2]
    right = frame[:, frame.shape[1] // 2:]

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Создание объекта стереозрения
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # Вычисление карты глубины
    disparity = stereo.compute(gray_left, gray_right)

    # Нормализация карты глубины для отображения
    #normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized_disparity = np.ascontiguousarray(np.uint16(disparity) * 256)
    # cv2.imshow('', normalized_disparity)
    # cv2.waitKey(0)
    return np.ascontiguousarray(left), normalized_disparity


def get_depth_and_color_frames_from_dir(dir, index):
    image_index = f'{index:05}'
    try:
        rgb = cv2.imread(f'{dir}\\rgb\\{image_index}.jpg')
        depth = cv2.imread(f'{dir}\\depth\\{image_index}.png')
        rgb = cv2.resize(rgb, (640, 480))
        depth = cv2.resize(depth, (640, 480))

        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.uint16) * 256
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)


        rgb = o3d.t.geometry.Image(rgb)
        depth = o3d.t.geometry.Image(depth)
        # rgb = o3d.t.io.read_image(f'{dir}\\rgb\\{image_index}.jpg')
        # depth = o3d.t.io.read_image(f'{dir}\\depth\\{image_index}.png')
        return rgb, depth
    except FileNotFoundError as e:
        log.error(f'Ошибка чтения кадра из файла.\n{e}')
        return None, None


def get_t_depth_and_color_frames(video_pre_processing, priority=False):
    color, depth = video_pre_processing.get_frame(priority)
    if color:
        video_pre_processing.write_video(np.asarray(color.get_data()))
        color = o3d.t.geometry.Image(np.asarray(color.get_data()))
        depth = o3d.t.geometry.Image(np.asarray(depth.get_data()))
        return color, depth

    return None, None


def setup_video_pre_processing():
    video_pre_processing = VideoPreProcessing()
    #video_pre_processing.set_config_realsense()
    #video_pre_processing.set_video_capture()
    #video_pre_processing.set_video_writer()
    return video_pre_processing


def setup_reconstruction(video_pre_processing, frame):
    reconstruction_config = ReconstructionConfig()
    #rgb, depth = get_t_depth_and_color_frames(video_pre_processing)
    # rgb, depth = get_depth_and_color_frames_from_dir('dataset', 0)
    # frame, depth = StereoSGBM_func(frame)
    iteration_time = time.time()
    reconstruction = Reconstruction(reconstruction_config, frame)
    log.info(f'{0} :  {time.time() - iteration_time}')
    return reconstruction


def run_system():
    cap = cv2.VideoCapture('cut.avi')
    _, init_frame = cap.read()
    init_frame = cv2.resize(init_frame, (1280, 480))
    image_index = 1
    priority = False
    video_pre_processing = setup_video_pre_processing()
    color, init_depth = StereoSGBM_func(init_frame)
    init_depth = o3d.t.geometry.Image(np.asarray(init_depth))
    reconstruction = setup_reconstruction(video_pre_processing, init_depth)
    total_time = time.time()

    while True:

        _, rgb = cap.read()
        rgb = cv2.resize(rgb, (1280, 480))
        rgb, depth = StereoSGBM_func(rgb)

        # cv2.imwrite(f'dataset2\\rgb\\{image_index}.jpg', rgb)
        # cv2.imwrite(f'dataset2\\depth\\{image_index}.png', depth)
        #
        # rgb = o3d.t.io.read_image(f'dataset\\rgb\\{image_index:05}.jpg')
        # depth = o3d.t.io.read_image(f'dataset\\depth\\{image_index:05}.png')
        # print(rgb.dtype)
        # print(depth.dtype)
        #
        # rgb = o3d.t.io.read_image(f'dataset2\\rgb\\{image_index}.jpg')
        # depth = o3d.t.io.read_image(f'dataset2\\depth\\{image_index}.png')
        # print(rgb.dtype)
        # print(depth.dtype)


        #rgb, depth = get_t_depth_and_color_frames(video_pre_processing, priority)
        # rgb, depth = get_depth_and_color_frames_from_dir('dataset', image_index)
        if priority:
            priority = False

        if rgb is not None:
            iteration_time = time.time()
            success = reconstruction.launch(rgb, depth)
            log.info(f'{image_index} :  {time.time() - iteration_time}')
            image_index += 1

            if not success:
                priority = True

        if image_index == 300:
            break

    log.info(f'FINAL TIME : {time.time() - total_time}')
    #video_pre_processing.video_writer_release()
    reconstruction.visualize_and_save_pcd()


if __name__ == '__main__':
    run_system()
    # img = o3d.t.io.read_image('dataset\\rgb\\00001.jpg')
    # np_img = np.array(img)
    # print(np_img.shape, np_img.dtype, img.dtype)