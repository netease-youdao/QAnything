# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Dict, List, Tuple

import numpy as np


class TableRecover:
    def __init__(
        self,
    ):
        pass

    def __call__(self, polygons: np.ndarray) -> Dict[int, Dict]:
        rows = self.get_rows(polygons)
        longest_col, each_col_widths, col_nums = self.get_benchmark_cols(rows, polygons)
        each_row_heights, row_nums = self.get_benchmark_rows(rows, polygons)
        table_res = self.get_merge_cells(
            polygons,
            rows,
            row_nums,
            col_nums,
            longest_col,
            each_col_widths,
            each_row_heights,
        )
        return table_res

    @staticmethod
    def get_rows(polygons: np.array) -> Dict[int, List[int]]:
        """对每个框进行行分类，框定哪个是一行的"""
        if polygons.size == 0:
            return {0: [0]}
            
        y_axis = polygons[:, 0, 1]
        if y_axis.size == 1:
            return {0: [0]}

        concat_y = np.array(list(zip(y_axis, y_axis[1:])))
        minus_res = concat_y[:, 1] - concat_y[:, 0]

        result = {}
        thresh = 5.0
        split_idxs = np.argwhere(minus_res > thresh).squeeze()
        if split_idxs.ndim == 0:
            split_idxs = split_idxs[None, ...]

        if split_idxs.size == 0:
            return {0: [0]}

        if max(split_idxs) != len(minus_res):
            split_idxs = np.append(split_idxs, len(minus_res))

        start_idx = 0
        for row_num, idx in enumerate(split_idxs):
            if row_num != 0:
                start_idx = split_idxs[row_num - 1] + 1
            result.setdefault(row_num, []).extend(range(start_idx, idx + 1))

        # 计算每一行相邻cell的iou，如果大于0.2，则合并为同一个cell
        return result

    def get_benchmark_cols(
        self, rows: Dict[int, List], polygons: np.ndarray
    ) -> Tuple[np.ndarray, List[float], int]:
        if polygons.size == 0:
            return None, [], 0

        longest_col = max(rows.values(), key=lambda x: len(x))
        longest_col_points = polygons[longest_col]
        longest_x = longest_col_points[:, 0, 0]

        theta = 10
        for row_value in rows.values():
            cur_row = polygons[row_value][:, 0, 0]

            range_res = {}
            for idx, cur_v in enumerate(cur_row):
                start_idx, end_idx = None, None
                for i, v in enumerate(longest_x):
                    if cur_v - theta <= v <= cur_v + theta:
                        break

                    if cur_v > v:
                        start_idx = i
                        continue

                    if cur_v < v:
                        end_idx = i
                        break

                range_res[idx] = [start_idx, end_idx]

            sorted_res = dict(
                sorted(range_res.items(), key=lambda x: x[0], reverse=True)
            )
            for k, v in sorted_res.items():
                if not all(v):
                    continue

                longest_x = np.insert(longest_x, v[1], cur_row[k])
                longest_col_points = np.insert(
                    longest_col_points, v[1], polygons[row_value[k]], axis=0
                )

        # 求出最右侧所有cell的宽，其中最小的作为最后一列宽度
        rightmost_idxs = [v[-1] for v in rows.values()]
        rightmost_boxes = polygons[rightmost_idxs]
        min_width = min([self.compute_L2(v[3, :], v[0, :]) for v in rightmost_boxes])

        each_col_widths = (longest_x[1:] - longest_x[:-1]).tolist()
        each_col_widths.append(min_width)

        col_nums = longest_x.shape[0]
        return longest_col_points, each_col_widths, col_nums

    def get_benchmark_rows(
        self, rows: Dict[int, List], polygons: np.ndarray
    ) -> Tuple[np.ndarray, List[float], int]:
        if polygons.size == 0:
            return None, []

        leftmost_cell_idxs = [v[0] for v in rows.values()]
        benchmark_x = polygons[leftmost_cell_idxs][:, 0, 1]

        theta = 10
        # 遍历其他所有的框，按照y轴进行区间划分
        range_res = {}
        for cur_idx, cur_box in enumerate(polygons):
            if cur_idx in benchmark_x:
                continue

            cur_y = cur_box[0, 1]

            start_idx, end_idx = None, None
            for i, v in enumerate(benchmark_x):
                if cur_y - theta <= v <= cur_y + theta:
                    break

                if cur_y > v:
                    start_idx = i
                    continue

                if cur_y < v:
                    end_idx = i
                    break

            range_res[cur_idx] = [start_idx, end_idx]

        sorted_res = dict(sorted(range_res.items(), key=lambda x: x[0], reverse=True))
        for k, v in sorted_res.items():
            if not all(v):
                continue

            benchmark_x = np.insert(benchmark_x, v[1], polygons[k][0, 1])

        each_row_widths = (benchmark_x[1:] - benchmark_x[:-1]).tolist()

        # 求出最后一行cell中，最大的高度作为最后一行的高度
        bottommost_idxs = list(rows.values())[-1]
        bottommost_boxes = polygons[bottommost_idxs]
        max_height = max([self.compute_L2(v[3, :], v[0, :]) for v in bottommost_boxes])
        each_row_widths.append(max_height)

        row_nums = benchmark_x.shape[0]
        return each_row_widths, row_nums

    @staticmethod
    def compute_L2(a1: np.ndarray, a2: np.ndarray) -> float:
        return np.linalg.norm(a2 - a1)

    def get_merge_cells(
        self,
        polygons: np.ndarray,
        rows: Dict,
        row_nums: int,
        col_nums: int,
        longest_col: np.ndarray,
        each_col_widths: List[float],
        each_row_heights: List[float],
    ) -> Dict[int, Dict[int, int]]:
        if polygons.size == 0:
            return {}

        col_res_merge, row_res_merge = {}, {}
        merge_thresh = 20
        for cur_row, col_list in rows.items():
            one_col_result, one_row_result = {}, {}
            for one_col in col_list:
                box = polygons[one_col]
                box_width = self.compute_L2(box[3, :], box[0, :])

                # 不一定是从0开始的，应该综合已有值和x坐标位置来确定起始位置
                loc_col_idx = np.argmin(np.abs(longest_col[:, 0, 0] - box[0, 0]))
                merge_col_cell = max(sum(one_col_result.values()), loc_col_idx)

                # 计算合并多少个列方向单元格
                for i in range(merge_col_cell, col_nums):
                    col_cum_sum = sum(each_col_widths[merge_col_cell : i + 1])
                    if i == merge_col_cell and col_cum_sum > box_width:
                        one_col_result[one_col] = 1
                        break
                    elif abs(col_cum_sum - box_width) <= merge_thresh:
                        one_col_result[one_col] = i + 1 - merge_col_cell
                        break
                else:
                    one_col_result[one_col] = i + 1 - merge_col_cell + 1

                box_height = self.compute_L2(box[1, :], box[0, :])
                merge_row_cell = cur_row
                for j in range(merge_row_cell, row_nums):
                    row_cum_sum = sum(each_row_heights[merge_row_cell : j + 1])
                    # box_height 不确定是几行的高度，所以要逐个试验，找一个最近的几行的高
                    # 如果第一次row_cum_sum就比box_height大，那么意味着？丢失了一行
                    if j == merge_row_cell and row_cum_sum > box_height:
                        one_row_result[one_col] = 1
                        break

                    elif abs(box_height - row_cum_sum) <= merge_thresh:
                        one_row_result[one_col] = j + 1 - merge_row_cell
                        break
                else:
                    one_row_result[one_col] = j + 1 - merge_row_cell + 1

            col_res_merge[cur_row] = one_col_result
            row_res_merge[cur_row] = one_row_result

        res = {}
        for i, (c, r) in enumerate(zip(col_res_merge.values(), row_res_merge.values())):
            res[i] = {k: [cc, r[k]] for k, cc in c.items()}
        return res
