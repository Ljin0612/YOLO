"""Evaluate UNIV + YOLO-style detector on SeaShips24790."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.datasets.seaships_yolo_detection_dataset import CLASS_NAMES,SeaShipsYoloDetectionDataset,collate_fn
from scripts.models.univ_yolo_detector import build_univ_yolo_detector,decode_predictions
from scripts.eval_univ_detector_seaships24790 import evaluate_model
class Wrap(torch.nn.Module):
 def __init__(self,m,imgsz,conf,iou): super().__init__(); self.m=m; self.imgsz=imgsz; self.conf=conf; self.iou=iou
 def eval(self): self.m.eval(); return self
 def forward(self,images):
  x=torch.stack(images) if isinstance(images,(list,tuple)) else images
  return decode_predictions(self.m(x),self.imgsz,nc=self.m.nc,conf_thres=self.conf,iou_thres=self.iou)
def args():
 p=argparse.ArgumentParser(description="Evaluate UNIV-YOLO on SeaShips24790")
 p.add_argument("--weights",required=True); p.add_argument("--data",default="configs/seaships24790.local.yaml"); p.add_argument("--split",default="val",choices=["train","val","test"]); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--batch",type=int,default=1); p.add_argument("--device",default="0"); p.add_argument("--project",default="runs/detect/univ_yolo_seaships24790"); p.add_argument("--name",default="eval"); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--conf-thres",type=float,default=0.001); p.add_argument("--iou-thres",type=float,default=0.6); return p.parse_args()
def dev(a): return torch.device("cpu" if a=="cpu" or not torch.cuda.is_available() else f"cuda:{a}")
def main():
 a=args(); d=dev(a.device); out=Path(a.project)/a.name; out.mkdir(parents=True,exist_ok=True)
 ds=SeaShipsYoloDetectionDataset(a.data,a.split,a.imgsz); dl=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=a.num_workers,collate_fn=collate_fn)
 ck=torch.load(a.weights,map_location="cpu",weights_only=False); m=build_univ_yolo_detector(nc=6,univ_weights=None).to(d); m.load_state_dict(ck.get("model",ck),strict=False)
 metrics=evaluate_model(Wrap(m,a.imgsz,a.conf_thres,a.iou_thres),dl,d,6)
 md=out/"eval_summary.md"; cp=out/"eval_metrics.csv"
 lines=["# UNIV-YOLO Evaluation Summary","",f"- Weights: `{a.weights}`",f"- Split: `{a.split}`",f"- imgsz: `{a.imgsz}`","","| Metric | Value |","| --- | ---: |"]
 for k,v in metrics.items(): lines.append(f"| {k} | {v:.6f} |")
 md.write_text("\n".join(lines)+"\n",encoding="utf-8")
 with cp.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(metrics.keys())); w.writeheader(); w.writerow(metrics)
 for key in ["Precision","Recall","mAP50","mAP50:95"]: print(f"{key}: {metrics.get(key,0):.6f}")
 for c in CLASS_NAMES.values(): print(f"AP50/{c}: {metrics.get('AP50/'+c,0):.6f}")
 print(f"Markdown: {md}\nCSV: {cp}")
if __name__=="__main__": main()
