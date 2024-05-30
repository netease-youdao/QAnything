import os
import re
from collections import Counter
from copy import deepcopy
import numpy as np
from qanything_kernel.utils.loader.pdf_to_markdown.core.vision import Recognizer
from qanything_kernel.configs.model_config import PDF_MODEL_PATH
from tqdm import tqdm


class LayoutRecognizer(Recognizer):
    labels = ['Text', 'Title', 'Figure', 'Equation', 'Table', 
        'Caption', 'Header', 'Footer', 'BibInfo', 'Reference',
        'Content', 'Code', 'Other', 'Item', 'Author']

    def __init__(self, domain):
        model_dir = os.path.join(
                    PDF_MODEL_PATH,
                    "checkpoints/layout")
        super().__init__(self.labels, domain, model_dir)
        self.garbage_layouts = ["footer", "header"]

    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.4, batch_size=16, drop=True):
        def __is_garbage(b):
            patt = ['\* Corresponding Author', '\*Corresponding to']
            return any([re.search(p, b["text"]) for p in patt])

        layouts = super().__call__(image_list, thr, batch_size)
        # save_results(image_list, layouts, self.labels, output_dir='output/', threshold=0.7)
        assert len(image_list) == len(ocr_res)
        # Tag layout type
        boxes = []
        assert len(image_list) == len(layouts)
        garbages = {}
        page_layout = []
        for pn, lts in tqdm(enumerate(layouts)):
            bxs = ocr_res[pn]
            lts = [{"type": b["type"],
                    "score": float(b["score"]),
                    "x0": b["bbox"][0] / scale_factor, "x1": b["bbox"][2] / scale_factor,
                    "top": b["bbox"][1] / scale_factor, "bottom": b["bbox"][-1] / scale_factor,
                    "page_number": pn,
                    } for b in lts]
            lts = self.sort_Y_firstly(lts, np.mean(
                [l["bottom"] - l["top"] for l in lts]) / 2)
            lts = self.layouts_cleanup(bxs, lts)
            if pn == 0:
                try:
                    idx = [b['x0'] for b in lts].index(min([b['x0'] for b in lts if b['type'] == 'text']))
                    if (lts[idx]['bottom']-lts[idx]['top'])/(lts[idx]['x1']-lts[idx]['x0']) > 15:
                        lts.pop(idx)
                except:
                    lts = lts
            page_layout.append(lts)

            # Tag layout type, layouts are ready
            def findLayout(ty):
                nonlocal bxs, lts, self
                lts_ = [lt for lt in lts if lt["type"] == ty]
                i = 0
                while i < len(bxs):
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue
                    if __is_garbage(bxs[i]):
                        bxs.pop(i)
                        continue

                    ii = self.find_overlapped_with_threashold(bxs[i], lts_,
                                                              thr=0.4)

                    if ii is None:  # belong to nothing
                        bxs[i]["layout_type"] = ""
                        i += 1
                        continue
                    lts_[ii]["visited"] = True
                    keep_feats = [
                        lts_[
                            ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].size[1] * 0.9 / scale_factor,
                        lts_[
                            ii]["type"] == "header" and bxs[i]["top"] > image_list[pn].size[1] * 0.1 / scale_factor,
                    ]
                    if drop and lts_[
                            ii]["type"] in self.garbage_layouts and not any(keep_feats):
                        if lts_[ii]["type"] not in garbages:
                            garbages[lts_[ii]["type"]] = []
                        garbages[lts_[ii]["type"]].append(bxs[i]["text"])
                        bxs.pop(i)
                        continue

                    bxs[i]["layoutno"] = f"{ty}-{ii}"
                    bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[
                        ii]["type"] != "equation" else "figure"
                    i += 1

            for ty in ["footer", "header", "reference", "caption", "author",
                       "title", "table", "text", "figure", "equation", "content"]:
                findLayout(ty)
            # add box to figure layouts which has not text box
            for i, lt in enumerate(
                    [lt for lt in lts if lt["type"] in ["figure", "equation"]]):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                del lt["type"]
                lt["text"] = ""
                lt["layout_type"] = "figure"
                lt["layoutno"] = f"figure-{i}"
                lt["page_number"] = pn + 1
                bxs.append(lt)
            
            lts_ = [lt for lt in lts if lt["type"] == 'item']
            for i, bx in enumerate(bxs):
                if bx["layout_type"] != 'reference': continue
                ii = self.find_overlapped_with_threashold(bx, lts_,
                                                              thr=0.4)
                if ii is None:
                    continue
                layoutno = bx["layoutno"]
                bxs[i]["layoutno"] = f"{layoutno}-item-{ii}"

            boxes.extend(bxs)

        ocr_res = boxes

        garbag_set = set()
        for k in garbages.keys():
            garbages[k] = Counter(garbages[k])
            for g, c in garbages[k].items():
                if c > 1:
                    garbag_set.add(g)

        ocr_res = [b for b in ocr_res if b["text"].strip() not in garbag_set]
        return ocr_res, page_layout
