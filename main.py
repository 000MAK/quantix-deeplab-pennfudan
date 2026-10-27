import argparse, json, urllib.request, zipfile, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
from torchvision import models,transforms
URL='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
class Seg(Dataset):
 def __init__(self,r): self.r=r; self.img=sorted((r/'PennFudanPed/PNGImages').glob('*.png')); self.m=r/'PennFudanPed/PedMasks'; self.tf=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
 def __len__(self): return len(self.img)
 def __getitem__(self,i):
  x=self.tf(Image.open(self.img[i]).convert('RGB')); m=np.zeros((128,128),np.int64)
  for f in sorted(self.m.glob(self.img[i].stem+'*')): m=np.maximum(m,np.asarray(Image.open(f).resize((128,128),Image.Resampling.NEAREST))>0)
  return x,torch.tensor(m,dtype=torch.long)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=2); a=p.parse_args(); r=Path(__file__).parent; d=r/'data'; (r/'outputs').mkdir(exist_ok=True); d.mkdir(exist_ok=True); z=d/'source.zip'
 if not z.exists(): urllib.request.urlretrieve(URL,z)
 if not (d/'PennFudanPed').exists():
  with zipfile.ZipFile(z) as f:
   if any('..' in Path(n).parts or n.startswith('/') for n in f.namelist()): raise ValueError('Unsafe archive')
   f.extractall(d)
 ds=Seg(d); n=len(ds); train_size=int(n*.7); val_size=int(n*.15); test_size=n-train_size-val_size; tr,va,te=random_split(ds,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42)); device='cuda' if torch.cuda.is_available() else 'cpu'; model=models.segmentation.deeplabv3_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=2).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
 def run(part,train=False):
  model.train(train); losses=[]; hit=tot=0
  for x,y in DataLoader(part,batch_size=2,shuffle=train):
   x,y=x.to(device),y.to(device); opt.zero_grad(); o=model(x)['out']; loss=torch.nn.functional.cross_entropy(o,y)
   if train: loss.backward(); opt.step()
   losses.append(float(loss)); hit+=(o.argmax(1)==y).sum().item(); tot+=y.numel()
  return {'loss':round(float(np.mean(losses)),4),'pixel_accuracy':round(hit/tot,4)}
 out={}
 for e in range(a.epochs): out[f'epoch_{e+1}']={'train':run(tr,True),'validation':run(va)}
 out['test']=run(te); torch.save(model.state_dict(),r/'outputs/deeplab.pt'); (r/'outputs/metrics.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
