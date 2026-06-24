"""Visualize UNIV-YOLO predictions."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
import torchvision.transforms.functional as TF
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.datasets.seaships_yolo_detection_dataset import CLASS_NAMES,IMAGE_SUFFIXES,load_data_yaml,resolve_split_dir
from scripts.models.univ_yolo_detector import build_univ_yolo_detector,decode_predictions
def parse():
 p=argparse.ArgumentParser(description="Predict with UNIV-YOLO")
 p.add_argument("--weights",required=True); p.add_argument("--source",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--device",default="0"); p.add_argument("--conf-thres",type=float,default=0.25); p.add_argument("--save-dir",default="runs/detect/univ_yolo_seaships24790/predict"); return p.parse_args()
def device(a): return torch.device("cpu" if a=="cpu" or not torch.cuda.is_available() else f"cuda:{a}")
def images(src):
 p=Path(src).expanduser()
 if p.suffix.lower() in {".yaml",".yml"}: p=resolve_split_dir(load_data_yaml(p),"test")
 return sorted([p] if p.is_file() else [x for x in p.rglob("*") if x.suffix.lower() in IMAGE_SUFFIXES])
def main():
 a=parse(); d=device(a.device); out=Path(a.save_dir); out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(a.weights,map_location="cpu",weights_only=False); m=build_univ_yolo_detector(nc=6,univ_weights=None).to(d); m.load_state_dict(ck.get("model",ck),strict=False); m.eval()
 colors=["red","lime","cyan","yellow","magenta","orange"]
 for ip in images(a.source):
  im=Image.open(ip).convert("RGB"); ow,oh=im.size; tin=TF.to_tensor(im.resize((a.imgsz,a.imgsz))).unsqueeze(0).to(d)
  with torch.no_grad(): det=decode_predictions(m(tin),a.imgsz,6,a.conf_thres,0.45)[0]
  draw=ImageDraw.Draw(im); sx=ow/a.imgsz; sy=oh/a.imgsz
  for box,score,lab in zip(det["boxes"].cpu(),det["scores"].cpu(),det["labels"].cpu()):
   cls=int(lab)-1; x1,y1,x2,y2=[float(v) for v in box]; xy=[x1*sx,y1*sy,x2*sx,y2*sy]; color=colors[cls%len(colors)]
   name=CLASS_NAMES.get(cls,str(cls)); draw.rectangle(xy,outline=color,width=3); draw.text((xy[0],max(0,xy[1]-14)),f"{name} {float(score):.2f}",fill=color)
  im.save(out/ip.name)
 print(f"Saved predictions to {out}")
if __name__=="__main__": main()
