"""Train UNIV + YOLO-style detector on SeaShips24790."""
from __future__ import annotations
import argparse, csv, random, sys
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.datasets.seaships_yolo_detection_dataset import SeaShipsYoloDetectionDataset, collate_fn
from scripts.models.univ_yolo_detector import build_univ_yolo_detector, decode_predictions, UNIVYoloDetector
from scripts.eval_univ_detector_seaships24790 import evaluate_model as eval_torchvision

def str2bool(v): return v if isinstance(v,bool) else str(v).lower() in {"1","true","yes","y"}
def parse_args():
 p=argparse.ArgumentParser(description="Train UNIV-YOLO on SeaShips24790")
 p.add_argument("--data",default="configs/seaships24790.local.yaml"); p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch",type=int,default=1); p.add_argument("--imgsz",type=int,default=640,choices=[320,640]); p.add_argument("--device",default="0"); p.add_argument("--project",default="runs/detect/univ_yolo_seaships24790"); p.add_argument("--name",default="univ_yolo")
 p.add_argument("--univ-weights",default="pretrained/checkpoint0400.pth"); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--amp",type=str2bool,default=False); p.add_argument("--freeze-backbone",type=str2bool,default=True); p.add_argument("--unfreeze-last-blocks",type=int,default=0); p.add_argument("--resume",default=None)
 return p.parse_args()
def device_select(a): return torch.device("cpu" if a=="cpu" or not torch.cuda.is_available() else f"cuda:{a}")
def set_seed(s=42): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def yolo_loss(preds, targets, imgsz, nc=6):
 device=preds[0].device; lbox=torch.tensor(0.,device=device); lcls=torch.tensor(0.,device=device); lobj=torch.tensor(0.,device=device)
 for si,(p,stride) in enumerate(zip(preds,UNIVYoloDetector.strides)):
  b,_,h,w=p.shape; obj_t=torch.zeros((b,1,h,w),device=device); cls_t=torch.zeros((b,nc,h,w),device=device); box_t=torch.zeros((b,4,h,w),device=device); pos=torch.zeros((b,1,h,w),dtype=torch.bool,device=device)
  for bi,t in enumerate(targets):
   boxes=t["boxes"].to(device); labels=(t["labels"].to(device)-1).clamp(0,nc-1)
   for box,lab in zip(boxes,labels):
    cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2; gx=torch.clamp((cx/stride).long(),0,w-1); gy=torch.clamp((cy/stride).long(),0,h-1)
    obj_t[bi,0,gy,gx]=1.; cls_t[bi,lab,gy,gx]=1.; pos[bi,0,gy,gx]=True
    ccx=(gx.float()+0.5)*stride; ccy=(gy.float()+0.5)*stride
    box_t[bi,:,gy,gx]=torch.stack(((ccx-box[0])/stride,(ccy-box[1])/stride,(box[2]-ccx)/stride,(box[3]-ccy)/stride)).clamp(min=0)
  lobj=lobj+F.binary_cross_entropy_with_logits(p[:,4:5],obj_t)
  if pos.any():
   mask=pos.expand_as(p[:,:4]); lbox=lbox+F.smooth_l1_loss(F.softplus(p[:,:4])[mask],box_t[mask]); cmask=pos.expand_as(p[:,5:5+nc]); lcls=lcls+F.binary_cross_entropy_with_logits(p[:,5:5+nc][cmask],cls_t[cmask])
 return {"loss_box":lbox,"loss_cls":lcls,"loss_obj":lobj,"loss_total":lbox*5.0+lcls+lobj}

class EvalWrap(torch.nn.Module):
 def __init__(self, m, imgsz): super().__init__(); self.m=m; self.imgsz=imgsz
 def eval(self): self.m.eval(); return self
 def forward(self, images):
  x=torch.stack(images) if isinstance(images,(list,tuple)) else images
  return decode_predictions(self.m(x), self.imgsz, nc=self.m.nc, conf_thres=0.001, iou_thres=0.6)

def append_csv(path,row):
 old=[]; fields=[]
 if path.exists():
  with path.open(newline="",encoding="utf-8") as f: r=csv.DictReader(f); fields=list(r.fieldnames or []); old=list(r)
 fields += [k for k in row if k not in fields]
 with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(old); w.writerow(row)

def main():
 args=parse_args(); set_seed(); dev=device_select(args.device); save_dir=Path(args.project)/args.name; save_dir.mkdir(parents=True,exist_ok=True); log_path=save_dir/"train_log.csv"
 train=SeaShipsYoloDetectionDataset(args.data,"train",args.imgsz); val=SeaShipsYoloDetectionDataset(args.data,"val",args.imgsz)
 tl=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,collate_fn=collate_fn,pin_memory=dev.type=="cuda"); vl=DataLoader(val,batch_size=args.batch,shuffle=False,num_workers=args.num_workers,collate_fn=collate_fn)
 model=build_univ_yolo_detector(nc=6,univ_weights=args.univ_weights,freeze_backbone=args.freeze_backbone,unfreeze_last_blocks=args.unfreeze_last_blocks).to(dev)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay); scaler=torch.cuda.amp.GradScaler(enabled=args.amp and dev.type=="cuda"); start=0; best=-1.
 if args.resume:
  ck=torch.load(args.resume,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start=int(ck.get("epoch",0)); best=float(ck.get("best_map50",-1))
 for ep in range(start,args.epochs):
  model.train(); totals=defaultdict(float); steps=0
  for i,(imgs,tgts) in enumerate(tl,1):
   x=torch.stack([im.to(dev) for im in imgs]); opt.zero_grad(set_to_none=True)
   with torch.cuda.amp.autocast(enabled=args.amp and dev.type=="cuda"):
    losses=yolo_loss(model(x),tgts,args.imgsz); loss=losses["loss_total"]
   scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); steps+=1
   for k,v in losses.items(): totals[k]+=float(v.detach().cpu())
   if i%10==0: print(f"epoch {ep+1}/{args.epochs} batch {i}/{len(tl)} loss_total={totals['loss_total']/steps:.4f}",flush=True)
  avg={k:v/max(steps,1) for k,v in totals.items()}; metrics=eval_torchvision(EvalWrap(model,args.imgsz),vl,dev,6); row={"epoch":ep+1,**avg,**{k:metrics.get(k,0.) for k in ["Precision","Recall","mAP50","mAP50:95"]}}
  print(f"Epoch {ep+1}: loss_total={row['loss_total']:.4f} loss_box={row['loss_box']:.4f} loss_cls={row['loss_cls']:.4f} loss_obj={row['loss_obj']:.4f} mAP50={row['mAP50']:.4f} mAP50:95={row['mAP50:95']:.4f}",flush=True)
  append_csv(log_path,row); ck={"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":ep+1,"best_map50":best,"args":vars(args)}; torch.save(ck,save_dir/"last.pth")
  if row["mAP50"]>best: best=row["mAP50"]; ck["best_map50"]=best; torch.save(ck,save_dir/"best.pth")
 print(f"save_dir: {save_dir}")
if __name__=="__main__": main()
