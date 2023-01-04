# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import time

cur_dir = Path(__file__).resolve().parent
sys.path.append(str(cur_dir.parent.parent))

import cv2
from rapidocr_onnxruntime import RapidOCR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='config.yaml')
    args = parser.parse_args()

    yaml_path = cur_dir / args.yaml_path
    rapid_ocr = RapidOCR(yaml_path)

    image_dir = cur_dir / 'test_images_benchmark'
    if not image_dir.exists():
        raise FileNotFoundError(f'{image_dir} does not exits!!')

    image_list = list(image_dir.iterdir())

    cost_time_list = []
    for image_path in tqdm(image_list, desc='Test'):
        img = cv2.imread(str(image_path))

        start_time = time.time()
        result = rapid_ocr(img)
        elapse = time.time() - start_time

        cost_time_list.append(elapse)

    total_time = sum(cost_time_list)
    avg_time = total_time / len(cost_time_list)
    print(f'Total Files: {len(image_list)}, '
          f'Total Time: {total_time:.5f}, '
          f'Average Time: {avg_time:.5f}')
