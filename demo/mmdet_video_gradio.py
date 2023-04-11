# Copyright (c) OpenMMLab. All rights reserved.
import time
import os
import warnings
from argparse import ArgumentParser
from typing import Union

import cv2
import mmcv
import torch
import torch.nn as nn
import numpy as np
import gradio as gr

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class DetModel:
    MODEL_DICT = {
        'Faster_RCNN-R50': {
            'config':
            'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        },
        # 'YOLOX-tiny': {
        #     'config':
        #     'mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py',
        #     'model':
        #     'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        # },
        # 'YOLOX-s': {
        #     'config':
        #     'mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py',
        #     'model':
        #     'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        # },
        # 'YOLOX-l': {
        #     'config':
        #     'mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py',
        #     'model':
        #     'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        # },
        # 'YOLOX-x': {
        #     'config':
        #     'mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
        #     'model':
        #     'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        # },
    }

    def __init__(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self._load_all_models_once()
        self.model_name = 'Faster_RCNN-R50'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        return init_detector(dic['config'], dic['model'], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple: #[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list: # [np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list, #[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79

        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(image,
                                     person_det,
                                     score_thr=score_threshold,
                                     bbox_color=None,
                                     text_color=(200, 200, 200),
                                     mask_color=None)
        return vis[:, :, ::-1]  # BGR -> RGB

pose_config = 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py',

class PoseModel:
    MODEL_DICT = {
        'HRnet coco wholebody 384x288': {
            'config':
            'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py',
            'model': 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
        },
    #     'ViTPose-B (single-task train)': {
    #         'config':
    #         'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
    #         'model': 'models/vitpose-b.pth',
    #     },
    #     'ViTPose-L (single-task train)': {
    #         'config':
    #         'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
    #         'model': 'models/vitpose-l.pth',
    #     },
    #     'ViTPose-B (multi-task train, COCO)': {
    #         'config':
    #         'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
    #         'model': 'models/vitpose-b-multi-coco.pth',
    #     },
    #     'ViTPose-L (multi-task train, COCO)': {
    #         'config':
    #         'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
    #         'model': 'models/vitpose-l-multi-coco.pth',
    #     },
    }

    def __init__(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self.model_name = 'HRnet coco wholebody 384x288'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        model = init_pose_model(dic['config'], dic['model'], device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list,#[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple:#[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list,#[np.ndarray],
            box_score_threshold: float = 0.5) -> list:#[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list,#[dict[str, np.ndarray]],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB


class AppModel:
    def __init__(self, device: Union[str, torch.device]):
        self.det_model = DetModel(device)
        self.pose_model = PoseModel(device)
            
    def inference(self, det_model_name,
                    pose_model_name,
                    video_path, progress = gr.Progress()
                ):
        
        out_video_root = './vis_results'
        show = False
        device = 'cuda:0'
        det_cat_id = 1
        bbox_thr = 0.3
        kpt_thr = 0.3
        radius = 4
        thickness = 1
        use_multi_frames = False
        online = False

        assert show or (out_video_root != '')
        assert det_model_name is not None
        assert pose_model_name is not None

        print('Initializing model...')
        # build the detection model from a config file and a checkpoint file
        
        self.det_model.set_model(det_model_name)
        self.pose_model.set_model(pose_model_name)

        dataset = self.pose_model.model.cfg.data['test']['type']
        # get datasetinfo
        dataset_info = self.pose_model.model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)

        # read video
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Faild to load video file {video_path}'

        if out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(out_video_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = video.fps
            size = (video.width, video.height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestr = time.strftime("%Y%m%d-%H%M%S")


            out_file_name = f'vis_{timestr}_{os.path.basename(video_path)}'
            out_loc = os.path.join(out_video_root, out_file_name)
            print(f"we will be saving to {out_loc}")

            # out_file_name = f'vis_{os.path.basename(video_path)}_{timestr}'
            # out_loc = os.path.join(out_video_root, out_file_name)
            videoWriter = cv2.VideoWriter(out_loc, fourcc,
                fps, size)

        # frame index offsets for inference, used in multi-frame inference setting
        if use_multi_frames:
            assert 'frame_indices_test' in self.pose_model.model.cfg.data.test.data_cfg
            indices = self.pose_model.model.cfg.data.test.data_cfg['frame_indices_test']

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = 'backbone'

        print('Running inference...')
        for frame_id, cur_frame in progress.tqdm(enumerate(mmcv.track_iter_progress(video)), desc="estimating pose"):
            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(self.det_model.model, cur_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, det_cat_id)

            if use_multi_frames:
                frames = collect_multi_frames(video, frame_id, indices,
                                            online)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model.model,
                frames if use_multi_frames else cur_frame,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            # show the results
            vis_frame = vis_pose_result(
                self.pose_model.model,
                cur_frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=kpt_thr,
                radius=radius,
                thickness=thickness,
                show=False)

            if show:
                cv2.imshow('Frame', vis_frame)

            if save_out_video:
                videoWriter.write(vis_frame)

            if show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save_out_video:
            videoWriter.release()
        if show:
            cv2.destroyAllWindows()
        
        return out_loc
    


