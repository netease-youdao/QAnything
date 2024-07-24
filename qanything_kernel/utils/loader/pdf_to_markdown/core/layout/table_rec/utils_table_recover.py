# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import random
from typing import Dict, List, Union

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon


def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape (N, 4, 2)
    return:
        sorted boxes(array) with shape (N, 4, 2)
    """
    num_boxes = dt_boxes.shape[0]
    dt_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(dt_boxes)

    # 解决相邻框，后边比前面y轴小，则会被排到前面去的问题
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                and _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return np.array(_boxes)


def compute_poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个多边形的IOU

    Args:
        poly1 (np.ndarray): (4, 2)
        poly2 (np.ndarray): (4, 2)

    Returns:
        float: iou
    """
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull

    union_poly = np.concatenate((a, b))

    if not poly1.intersects(poly2):
        return 0.0

    try:
        inter_area = poly1.intersection(poly2).area
        union_area = MultiPoint(union_poly).convex_hull.area
    except shapely.geos.TopologicalError:
        print("shapely.geos.TopologicalError occured, iou set to 0")
        return 0.0

    if union_area == 0:
        return 0.0

    return float(inter_area) / union_area


def merge_adjacent_polys(polygons: np.ndarray) -> np.ndarray:
    """合并相邻iou大于阈值的框"""
    combine_iou_thresh = 0.1
    pair_polygons = list(zip(polygons, polygons[1:, ...]))
    pair_ious = np.array([compute_poly_iou(p1, p2) for p1, p2 in pair_polygons])
    idxs = np.argwhere(pair_ious >= combine_iou_thresh)

    if idxs.size <= 0:
        return polygons

    polygons = combine_two_poly(polygons, idxs)

    # 注意：递归调用
    polygons = merge_adjacent_polys(polygons)
    return polygons


def combine_two_poly(polygons: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    del_idxs, insert_boxes = [], []
    idxs = idxs.squeeze(0)
    for idx in idxs:
        # idx 和 idx + 1 是重合度过高的
        # 合并，取两者各个点的最大值
        new_poly = []
        pre_poly, pos_poly = polygons[idx], polygons[idx + 1]

        # 四个点，每个点逐一比较
        new_poly.append(np.minimum(pre_poly[0], pos_poly[0]))

        x_2 = min(pre_poly[1][0], pos_poly[1][0])
        y_2 = max(pre_poly[1][1], pos_poly[1][1])
        new_poly.append([x_2, y_2])

        # 第3个点
        new_poly.append(np.maximum(pre_poly[2], pos_poly[2]))

        # 第4个点
        x_4 = max(pre_poly[3][0], pos_poly[3][0])
        y_4 = min(pre_poly[3][1], pos_poly[3][1])
        new_poly.append([x_4, y_4])

        new_poly = np.array(new_poly)

        # 删除已经合并的两个框，插入新的框
        del_idxs.extend([idx, idx + 1])
        insert_boxes.append(new_poly)

    # 整合合并后的框
    polygons = np.delete(polygons, del_idxs, axis=0)

    insert_boxes = np.array(insert_boxes)
    polygons = np.append(polygons, insert_boxes, axis=0)
    polygons = sorted_boxes(polygons)
    return polygons


def match_ocr_cell(
    polygons: np.ndarray, ocr_res: List[Union[List[List[float]], str, str]]
) -> Dict[int, List]:
    if polygons.size == 0:
        return {}, {}, {}

    cell_box_map = {}
    head_box_map = {}
    tail_box_map = {}
    dt_boxes, rec_res, _ = list(zip(*ocr_res))
    dt_boxes = np.array(dt_boxes)
    iou_thresh = 0.05
    head_idx = 0
    tail_idx = 0
    for i, dt_box in enumerate(dt_boxes):
        ious = [compute_poly_iou(dt_box, cell_box) for cell_box in polygons]
        if sum(ious) == 0 and dt_box[-1][1] < polygons[-1][0][1]:
            head_box_map.setdefault(head_idx, rec_res[i])
            head_idx += 1
        elif sum(ious) == 0 and dt_box[0][1] > polygons[0][2][1]:
            tail_box_map.setdefault(tail_idx, rec_res[i])
            tail_idx += 1
    for i, cell_box in enumerate(polygons):
        ious = [compute_poly_iou(dt_box, cell_box) for dt_box in dt_boxes]

        # 对有iou的值，计算是否存在包含关系。如存在→iou=1
        have_iou_idxs = np.argwhere(ious)
        if have_iou_idxs.size > 0:
            have_iou_idxs = have_iou_idxs.squeeze(1)
            for idx in have_iou_idxs:
                if is_inclusive_each_other(cell_box, dt_boxes[idx]):
                    ious[idx] = 1.0

        if all(x <= iou_thresh for x in ious):
            # 说明这个cell中没有文本
            cell_box_map.setdefault(i, []).append("")
            continue

        same_cell_idxs = np.argwhere(np.array(ious) >= iou_thresh).squeeze(1)
        one_cell_txts = "\n".join([rec_res[idx] for idx in same_cell_idxs])
        cell_box_map.setdefault(i, []).append(one_cell_txts)
    return cell_box_map, head_box_map, tail_box_map


def is_inclusive_each_other(box1: np.ndarray, box2: np.ndarray):
    """判断两个多边形框是否存在包含关系

    Args:
        box1 (np.ndarray): (4, 2)
        box2 (np.ndarray): (4, 2)

    Returns:
        bool: 是否存在包含关系
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    poly1_area = poly1.convex_hull.area
    poly2_area = poly2.convex_hull.area

    if poly1_area > poly2_area:
        box_max = box1
        box_min = box2
    else:
        box_max = box2
        box_min = box1

    x0, y0 = np.min(box_min[:, 0]), np.min(box_min[:, 1])
    x1, y1 = np.max(box_min[:, 0]), np.max(box_min[:, 1])

    edge_x0, edge_y0 = np.min(box_max[:, 0]), np.min(box_max[:, 1])
    edge_x1, edge_y1 = np.max(box_max[:, 0]), np.max(box_max[:, 1])

    if x0 >= edge_x0 and y0 >= edge_y0 and x1 <= edge_x1 and y1 <= edge_y1:
        return True
    return False


def plot_html_table(table_res: Dict, cell_box_map: Dict, head_box_map: Dict, tail_box_map: Dict) -> str:
    table_str = f"<html><body>"
    for key in head_box_map.keys():
        table_str += f'<caption>' + head_box_map[key] + f'</caption><br/>'
    table_str = table_str + f"<table>"
    for v in table_res.values():
        table_str += "<tr>"
        for cell_idx, span_info in v.items():
            cur_txt = "".join(cell_box_map.get(cell_idx, ""))
            cur_txt = cur_txt.replace('\n', '<br/>')
            colspan, rowspan = span_info
            if colspan != 1 and rowspan != 1:
                table_str += (
                    f'<td rowspan="{rowspan}" colspan="{colspan}">{cur_txt}</td>'
                )
            elif colspan == 1 and rowspan != 1:
                table_str += f'<td rowspan="{rowspan}">{cur_txt}</td>'
            elif colspan != 1 and rowspan == 1:
                table_str += f'<td colspan="{colspan}">{cur_txt}</td>'
            else:
                table_str += f"<td>{cur_txt}</td>"

        table_str += "</tr>"
    table_str += "</table>"
    for key in tail_box_map.keys():
        table_str += f'<caption>' + tail_box_map[key] + f'</caption><br/>'
    table_str += "</body></html>"
    return table_str


# def plot_html_wireless_table(logi_points, cell_box_map):
#     logi_points = logi_points.astype(np.int32)
#     table_dict = {}
#     for cell_idx, v in enumerate(logi_points):
#         cur_row = v[0]
#         cur_txt = "\n".join(cell_box_map.get(cell_idx))
#         sr, er, sc, ec = v.tolist()
#         rowspan, colspan = er - sr + 1, ec - sc + 1
#         table_str = f'<td rowspan="{rowspan}" colspan="{colspan}">{cur_txt}</td>'
#         # table_str = f'<td rowspan="{rowspan}" colspan="{colspan}"><div style="line-height: 18px;">{cur_txt}</div></td>'
#         table_dict.setdefault(cur_row, []).append(table_str)

#     new_table_dict = {}
#     for k, v in table_dict.items():
#         new_table_dict[k] = ["<tr>"] + v + ["</tr>"]

#     html_start = """<html><body><table><tbody>"""
#     # html_start = """<html><style type="text/css">td {border-left: 1px solid;border-bottom:1px solid;}table, th {border-top:1px solid;font-size: 10px;border-collapse: collapse;border-right: 1px solid;}</style><body><table style="border-bottom:1px solid;border-top:1px solid;"><tbody>"""
#     html_end = "</tbody></table></body></html>"
#     html_middle = "".join([vv for v in new_table_dict.values() for vv in v])
#     table_str = f"{html_start}{html_middle}{html_end}"
#     return table_str


def plot_html_wireless_table(logi_points, cell_box_map):
    logi_points = logi_points.astype(np.int32)
    table_dict = {}
    col_dict = {}
    colspan_dict = {}
    for cell_idx, v in enumerate(logi_points):
        cur_row = v[0]
        cur_col = v[2]
        cur_txt = "\n".join(cell_box_map.get(cell_idx))
        sr, er, sc, ec = v.tolist()
        rowspan, colspan = er - sr + 1, ec - sc + 1
        table_str = f'<td rowspan="{rowspan}" colspan="{colspan}">{cur_txt}</td>'
        # table_str = f'<td rowspan="{rowspan}" colspan="{colspan}"><div style="line-height: 18px;">{cur_txt}</div></td>'
        table_dict.setdefault(cur_row, []).append(table_str)
        if rowspan != 1 and colspan == 1:
            for cur_row in range(v[0], v[1]+1):
                col_dict.setdefault(cur_row, []).append(cur_col)
                # colspan_dict.setdefault(cur_row, []).append(colspan)
        elif rowspan == 1 and colspan != 1:
            for cur_col in range(v[2], v[3]+1):
                col_dict.setdefault(cur_row, []).append(cur_col)
            # colspan_dict.setdefault(cur_row, []).append(colspan)
        else:
            col_dict.setdefault(cur_row, []).append(cur_col)
            # colspan_dict.setdefault(cur_row, []).append(colspan)

    new_table_dict = {}
    for k, v in table_dict.items():
        try:
            if len(col_dict[k]) < max(col_dict[k]):
                for i in range(max(col_dict[k])+1):
                    if i not in col_dict[k]: 
                        v.insert(i, '<td></td>')
                new_table_dict[k] = ["<tr>"] + v + ["</tr>"]
            else:
                new_table_dict[k] = ["<tr>"] + v + ["</tr>"]
        except Exception as e:
            continue

    html_start = """<html><body><table><tbody>"""
    # html_start = """<html><style type="text/css">td {border-left: 1px solid;border-bottom:1px solid;}table, th {border-top:1px solid;font-size: 10px;border-collapse: collapse;border-right: 1px solid;}</style><body><table style="border-bottom:1px solid;border-top:1px solid;"><tbody>"""
    html_end = "</tbody></table></body></html>"
    html_middle = "".join([vv for v in new_table_dict.values() for vv in v])
    table_str = f"{html_start}{html_middle}{html_end}"
    return table_str


def vis_table(img: np.ndarray, polygons: np.ndarray) -> np.ndarray:
    # mask = img.copy()
    for i, poly in enumerate(polygons):
        poly = np.round(poly).astype(np.int32).reshape(4, 2)

        random_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        cv2.polylines(img, [poly], 3, random_color)
        # cv2.fillPoly(mask, np.array([poly], dtype=np.int32), random_color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), poly[0], font, 1, random_color, 1)
    # img = img * 0.67 + mask * 0.33
    
    return img
