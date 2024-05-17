import onnxruntime as _ort
import numpy as np
import torch
import cv2
from .lib.utils.image import get_affine_transform, get_affine_transform_upper_left
from .lib.models.decode import ctdet_decode, corner_decode, ctdet_4ps_decode
from .lib.utils.post_process import ctdet_4ps_post_process_upper_left, ctdet_4ps_post_process, ctdet_corner_post_process
import os
from .table_recover import TableRecover
from .utils_table_recover import match_ocr_cell, plot_html_table, plot_html_wireless_table
from .utils_table_recover import merge_adjacent_polys, sorted_boxes
from qanything_kernel.configs.model_config import PDF_MODEL_PATH
import markdownify
import urllib
import urllib.request
import urllib.parse
import json


def pre_process(image, inp_height, inp_width, upper_left=False):
    height, width = image.shape[0:2]

    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    if upper_left:
        c = np.array([0, 0], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform_upper_left(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(
        image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)

    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

    meta = {'c': c, 's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    # images = torch.from_numpy(images)
    return images, meta


def post_process(dets, meta, corner_st, upper_left=False, scale=1):
    device = dets.device
    device_str = str(device)
    # print('device: ',device)
    if device_str.startswith('cuda'):
        dets = dets.detach().cpu().numpy()
    else:
        dets = dets.detach().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    #return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
    if upper_left:
        dets = ctdet_4ps_post_process_upper_left(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], 2)
    else:
        dets = ctdet_4ps_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], 2)
    corner_st = ctdet_corner_post_process(
        corner_st.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], 2)
    for j in range(1, 3):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
        dets[0][j][:, :8] /= scale
    return dets[0], corner_st[0]


def merge_outputs(detections):
    results = {}
    for j in range(1, 3):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)
    scores = np.hstack(
        [results[j][:, 8] for j in range(1, 3)])
    if len(scores) > 3000:
        kth = len(scores) - 3000
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 3):
            keep_inds = (results[j][:, 8] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def filter(results, logi, ps, vis_thresh):
    # this function select boxes
    device = logi.device
    batch_size, feat_dim = logi.shape[0], logi.shape[2]
    num_valid = sum(results[1][:, 8] >= vis_thresh)

    #if num_valid <= 900 : #opt.max_objs
    slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
    slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
    for i in range(batch_size):
        for j in range(num_valid):
            slct_logi[i, j, :] = logi[i, j, :].cpu()
            slct_dets[i, j, :] = ps[i, j, :].cpu()
    return torch.Tensor(slct_logi).to(device), torch.Tensor(slct_dets).to(device)


def _normalized_ps(ps, vocab_size):
    device = ps.device
    ps = torch.round(ps).to(torch.int64)
    ps = torch.where(ps < vocab_size, ps, (vocab_size - 1) * torch.ones(ps.shape).to(torch.int64).to(device))
    ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).to(device))
    return ps


def process_logi(logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev > 0.5, logi_floor + 1, logi_floor)
    return logi


def add_4ps_coco_bbox(image, bbox, logi=None):
    bbox = np.array(bbox, dtype=np.int32)

    if not logi is None:
        txt = '{:.0f},{:.0f},{:.0f},{:.0f}'.format(logi[0], logi[1], logi[2], logi[3])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.3, 2)[0]
    cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.line(image, (bbox[2], bbox[3]), (bbox[4], bbox[5]), (0, 0, 255), 2)
    cv2.line(image, (bbox[4], bbox[5]), (bbox[6], bbox[7]), (0, 0, 255), 2)
    cv2.line(image, (bbox[6], bbox[7]), (bbox[0], bbox[1]), (0, 0, 255), 2)  # 1 - 5

    if not logi is None:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2), (255, 128, 128), -1)
        cv2.putText(image, txt, (bbox[0], bbox[1] - 2),
                    font, 0.30, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)  #1 - 5 # 0.20 _ 0.60

    return image


def show_results(results, corner, logi=None):
    m, n = corner.shape
    polygons = []

    k = 0
    for m in range(len(results[1])):
        bbox = results[1][m]
        k = k + 1
        if bbox[8] >= 0.4:
            polygons.append([[bbox[0], bbox[1]], [bbox[6], bbox[7]], [bbox[4], bbox[5]], [bbox[2], bbox[3]]])
            # if len(logi.shape) == 1:
            #     add_4ps_coco_bbox(image, bbox[:8], logi)
            # else:
            #     add_4ps_coco_bbox(image, bbox[:8], logi[m,:])
    return polygons


# def sort_logi_by_polygons(
#         sorted_polygons: np.ndarray, polygons: np.ndarray, logi_points: np.ndarray
#     ) -> np.ndarray:
#         sorted_idx = []
#         for v in sorted_polygons:
#             loc_idx = np.argwhere(v[0, 0] == polygons[:, 0, 0]).squeeze()
#             sorted_idx.append(int(loc_idx))
#         logi_points = logi_points[sorted_idx]
#         return logi_points

def sort_logi_by_polygons(
        sorted_polygons: np.ndarray, polygons: np.ndarray, logi_points: np.ndarray
) -> np.ndarray:
    sorted_idx = []
    for v in sorted_polygons:
        loc_idx = np.argwhere((v[0] == polygons[:, 0]).all(axis=1)).squeeze()
        sorted_idx.append(int(loc_idx))
    logi_points = logi_points[sorted_idx]
    return logi_points


def sort_logi(
        sorted_polygons: np.ndarray, logi_points: np.ndarray
) -> np.ndarray:
    sorted_idx = []
    sorted_logi_points = np.array(sorted(logi_points, key=lambda x: (x[0], x[2])))
    for v in sorted_logi_points:
        loc_idx = np.argwhere((logi_points == v).all(axis=1)).squeeze()
        if loc_idx.size > 1:
            loc_idx = loc_idx[0]
        sorted_idx.append(int(loc_idx))
    sorted_polygons = sorted_polygons[sorted_idx]
    return sorted_logi_points, sorted_polygons


def wired_table_rec(image_path):
    # onnx_model = '/ssd8/exec/huangjy/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR/src/wired_model.onnx'
    # sess1 = _ort.InferenceSession(onnx_model, None, providers=['CUDAExecutionProvider'])
    model = torch.jit.load(
        '/ssd8/exec/huangjy/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR/src/wired_model.pt').to('cuda')
    model.eval()
    onnx_processor = '/ssd8/exec/huangjy/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR/src/wired_processor.onnx'
    sess2 = _ort.InferenceSession(onnx_processor, None, providers=['CUDAExecutionProvider'])
    table_recover = TableRecover()
    image = cv2.imread(image_path)
    input, meta = pre_process(image, 1024, 1024)
    # hm, st, wh, ax, cr, reg = sess1.run(None, {'image': input})
    # hm = torch.from_numpy(hm).sigmoid_().cuda()
    # st, wh, ax, cr, reg = torch.from_numpy(st).cuda(), torch.from_numpy(wh).cuda(), torch.from_numpy(ax).cuda(), torch.from_numpy(cr).cuda(), torch.from_numpy(reg).cuda()
    hm, st, wh, ax, cr, reg = model(torch.from_numpy(input).cuda())
    hm = hm.sigmoid_().detach().cuda()
    st, wh, ax, cr, reg = st.detach().cuda(), wh.detach().cuda(), ax.detach().cuda(), cr.detach().cuda(), reg.detach().cuda()
    scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:, 1:2, :, :], st, reg, K=1000)
    dets, keep, logi, cr = ctdet_4ps_decode(hm[:, 0:1, :, :], wh, ax, cr, corner_dict, reg=reg, K=3000, wiz_rev=True)
    raw_dets = dets
    corner_output = np.concatenate((np.transpose(xs.cpu().numpy()), np.transpose(ys.cpu().numpy()),
                                    np.array(st_reg.cpu().numpy()), np.transpose(scores.cpu().numpy())), axis=2)
    dets, corner_st_reg = post_process(dets, meta, corner_output)
    results = merge_outputs([dets])
    logi = logi + cr
    slct_logi, slct_dets = filter(results, logi, raw_dets[:, :, :8], 0.15)
    slct_dets = _normalized_ps(slct_dets, 256)
    _, slct_logi = sess2.run(None, {'vis_feat': slct_logi.cpu().numpy()})
    slct_logi = process_logi(torch.from_numpy(slct_logi).cuda())
    polygons = show_results(results, corner_st_reg, slct_logi.squeeze())
    polygons = np.array(polygons)
    polygons = sorted_boxes(polygons)
    try:
        polygons = merge_adjacent_polys(polygons)
    except:
        polygons = polygons
    table_res = table_recover(polygons)

    return polygons, table_res


def wireless_table_rec(image_path):
    onnx_model = '/ssd8/exec/huangjy/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR/src/wireless_model.onnx'
    sess1 = _ort.InferenceSession(onnx_model, None, providers=['CUDAExecutionProvider'])
    onnx_processor = '/ssd8/exec/huangjy/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR/src/wireless_processor.onnx'
    sess2 = _ort.InferenceSession(onnx_processor, None, providers=['CUDAExecutionProvider'])
    image = cv2.imread(image_path)
    input, meta = pre_process(image, 768, 768, upper_left=True)
    hm, st, wh, ax, cr, reg = sess1.run(None, {'image': input})
    hm = torch.from_numpy(hm).sigmoid_().cuda()
    st, wh, ax, cr, reg = torch.from_numpy(st).cuda(), torch.from_numpy(wh).cuda(), torch.from_numpy(
        ax).cuda(), torch.from_numpy(cr).cuda(), torch.from_numpy(reg).cuda()
    scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:, 1:2, :, :], st, reg, K=5000)
    dets, keep, logi, cr = ctdet_4ps_decode(hm[:, 0:1, :, :], wh, ax, cr, corner_dict, reg=reg, K=3000, wiz_rev=False)
    raw_dets = dets
    corner_output = np.concatenate(
        (np.transpose(xs.cpu()), np.transpose(ys.cpu()), np.array(st_reg.cpu()), np.transpose(scores.cpu())), axis=2)
    print(dets.device)
    dets, corner_st_reg = post_process(dets, meta, corner_output, upper_left=True)
    results = merge_outputs([dets])
    logi = logi + cr
    slct_logi, slct_dets = filter(results, logi, raw_dets[:, :, :8], 0.2)
    slct_dets = _normalized_ps(slct_dets, 256)
    # slct_dets = slct_dets.to(torch.float)
    _, slct_logi = sess2.run(None, {'slct_logi_feat': slct_logi.cpu().numpy(), 'dets': slct_dets.cpu().numpy()})
    slct_logi = process_logi(torch.from_numpy(slct_logi).cuda())
    polygons = show_results(results, corner_st_reg, slct_logi.squeeze())
    polygons = np.array(polygons)
    sorted_polygons = sorted_boxes(polygons)

    return polygons, sorted_polygons, slct_logi[0].cpu().numpy()


def request_service(input_dict, url):
    data = urllib.parse.urlencode(input_dict).encode("utf-8")
    f = urllib.request.urlopen(url=url, data=data)
    res_dict = json.loads(json.loads(f.read())['Result'])

    return res_dict


class TableParser(object):
    def __init__(self, device=torch.device("cpu")):
        # self.wired_model_stage1 = torch.jit.load('layout/table_rec/checkpoint/wired_model.pt').to('cuda')
        self.device = device
        print(self.device)
        self.device = device
        if self.device == torch.device('cuda'):
            onnx_backend = 'CUDAExecutionProvider'
        else:
            onnx_backend = 'CPUExecutionProvider'

        self.wired_model_stage1 = torch.jit.load(os.path.join(PDF_MODEL_PATH, 'checkpoints/table/wired_model.pt')).to(self.device)
        onnx_processor = os.path.join(PDF_MODEL_PATH, 'checkpoints/table/wired_processor.onnx')
        self.wired_model_stage2 = _ort.InferenceSession(onnx_processor, None, providers=[onnx_backend])
        wireless_onnx1 = os.path.join(PDF_MODEL_PATH, 'checkpoints/table/wireless_model.onnx')
        self.wireless_model_stage1 = _ort.InferenceSession(wireless_onnx1, None, providers=[onnx_backend])
        wireless_onnx2 = os.path.join(PDF_MODEL_PATH, 'checkpoints/table/wireless_processor.onnx')
        self.wireless_model_stage2 = _ort.InferenceSession(wireless_onnx2, None, providers=[onnx_backend])
        self.table_recover = TableRecover()

    def wired_rec(self, image):
        input, meta = pre_process(image, 1024, 1024)

        hm, st, wh, ax, cr, reg = self.wired_model_stage1(torch.from_numpy(input).to(self.device))
        hm = hm.sigmoid_().detach().to(self.device)
        st, wh, ax, cr, reg = st.detach().to(self.device), wh.detach().to(self.device), ax.detach().to(
            self.device), cr.detach().to(self.device), reg.detach().to(self.device)
        scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:, 1:2, :, :], st, reg, K=1000, device=self.device)
        # print(hm.device)
        dets, keep, logi, cr = ctdet_4ps_decode(hm[:, 0:1, :, :], wh, ax, cr, corner_dict, reg=reg, K=600, wiz_rev=True)
        raw_dets = dets
        if self.device == torch.device('cuda'):
            corner_output = np.concatenate((np.transpose(xs.cpu().numpy()), np.transpose(ys.cpu().numpy()),
                                            np.array(st_reg.cpu().numpy()), np.transpose(scores.cpu().numpy())), axis=2)
        else:
            corner_output = np.concatenate((
                                           np.transpose(xs.numpy()), np.transpose(ys.numpy()), np.array(st_reg.numpy()),
                                           np.transpose(scores.numpy())), axis=2)
        dets, corner_st_reg = post_process(dets, meta, corner_output)
        results = merge_outputs([dets])
        logi = logi + cr
        slct_logi, slct_dets = filter(results, logi, raw_dets[:, :, :8], 0.15)
        if self.device == torch.device('cuda'):
            _, slct_logi = self.wired_model_stage2.run(None, {'vis_feat': slct_logi.cpu().numpy()})
            slct_logi = process_logi(torch.from_numpy(slct_logi).to(self.device))
        else:
            _, slct_logi = self.wired_model_stage2.run(None, {'vis_feat': slct_logi.numpy()})
            slct_logi = process_logi(torch.from_numpy(slct_logi).to(self.device))
        polygons = show_results(results, corner_st_reg, slct_logi.squeeze())
        polygons = np.array(polygons)
        polygons = sorted_boxes(polygons)
        try:
            polygons = merge_adjacent_polys(polygons)
        except:
            polygons = polygons
        table_res = self.table_recover(polygons)
        return polygons, table_res

    def wireless_rec(self, image):
        input, meta = pre_process(image, 768, 768, upper_left=True)

        hm, st, wh, ax, cr, reg = self.wireless_model_stage1.run(None, {'image': input})
        hm = torch.from_numpy(hm).sigmoid_().to(self.device)
        st, wh, ax, cr, reg = torch.from_numpy(st).to(self.device), torch.from_numpy(wh).to(
            self.device), torch.from_numpy(ax).to(self.device), torch.from_numpy(cr).to(self.device), torch.from_numpy(
            reg).to(self.device)
        scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:, 1:2, :, :], st, reg, K=1000, device=self.device)
        dets, keep, logi, cr = ctdet_4ps_decode(hm[:, 0:1, :, :], wh, ax, cr, corner_dict, reg=reg, K=600,
                                                wiz_rev=False)
        raw_dets = dets
        if self.device == torch.device('cuda'):
            corner_output = np.concatenate(
                (np.transpose(xs.cpu()), np.transpose(ys.cpu()), np.array(st_reg.cpu()), np.transpose(scores.cpu())),
                axis=2)
        else:
            corner_output = np.concatenate((np.transpose(xs), np.transpose(ys), np.array(st_reg), np.transpose(scores)),
                                           axis=2)
        dets, corner_st_reg = post_process(dets, meta, corner_output, upper_left=True)
        results = merge_outputs([dets])
        logi = logi + cr
        slct_logi, slct_dets = filter(results, logi, raw_dets[:, :, :8], 0.2)
        slct_dets = _normalized_ps(slct_dets, 256)
        if self.device == torch.device('cuda'):
            _, slct_logi = self.wireless_model_stage2.run(None, {'slct_logi_feat': slct_logi.cpu().numpy(),
                                                                 'dets': slct_dets.cpu().numpy()})
            slct_logi = process_logi(torch.from_numpy(slct_logi).to(self.device))
        else:
            _, slct_logi = self.wireless_model_stage2.run(None, {'slct_logi_feat': slct_logi.numpy(),
                                                                 'dets': slct_dets.numpy()})
            slct_logi = process_logi(torch.from_numpy(slct_logi).to(self.device))
        polygons = show_results(results, corner_st_reg, slct_logi.squeeze())
        polygons = np.array(polygons)
        sorted_polygons = sorted_boxes(polygons)
        return polygons, sorted_polygons, slct_logi[0].cpu().numpy()

    def process(self, image, table_type, ocr_result, convert2markdown=True):
        if table_type == 'wired':
            polygons, table_res = self.wired_rec(image)
        else:
            polygons, sorted_polygons, slct_logi = self.wireless_rec(image)
        if table_type == 'wireless':
            slct_logi = sort_logi_by_polygons(
                sorted_polygons, polygons, slct_logi
            )
            sorted_logi_points, sorted_polygons = sort_logi(sorted_polygons, slct_logi)
            cell_box_map, _, _ = match_ocr_cell(sorted_polygons, ocr_result)
            table_str = plot_html_wireless_table(sorted_logi_points, cell_box_map)
            if convert2markdown:
                table_markdown = html2markdown(table_str)
            return table_str, table_markdown
        else:
            cell_box_map, head_box_map, tail_box_map = match_ocr_cell(polygons, ocr_result)
            table_str = plot_html_table(table_res, cell_box_map, head_box_map, tail_box_map)
            if convert2markdown:
                table_markdown = html2markdown(table_str)
            return table_str, table_markdown


def html2markdown(html_text):
    markdown_text = markdownify.markdownify(html_text)
    return markdown_text


# if __name__ == '__main__':
#     parser = TableParser()
#     image = cv2.imread('test.jpg')
#     table_str, table_markdown = parser.process(image, 'wireless')
#
#     print(table_markdown)
