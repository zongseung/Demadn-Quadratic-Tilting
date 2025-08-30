[Hierachical_Quadratic_Tilting.pdf](https://github.com/user-attachments/files/21681763/Hierachical_Quadratic_Tilting.pdf)
---

### 맥에서 추론 모델 사용시 `cuda` 아닌 mps 사용
``` bash
device = "mps" if torch.cuda.is_availabel() else "cpu"
```

### uv sync 하면 알아서 가상환경 설치됨
- ipykernel은 수정이 필요할 수 있음

### tiling은 추석, 설날 이전, 이후 하루씩 포함시켜서 윈도우를 생성
- 틸팅에 문제가 지속적으로 생김.


### .pth 파일 저징 및 추론 모델 확장코드는 다음과 같다.
```bash
# 기존 모델을 꼭 임포트 해서 불러와야 한다.!

from torch.utils.data import DataLoader, TensorDataset

def run_inference(model, X, Y, M, y0, scaler_y, device, batch_size=32):
    """
    DataLoader 기반 추론 → 역스케일 결과와 마스크 반환"
    """
    Xt  = torch.from_numpy(X).float()
    Yt  = torch.from_numpy(Y).float()
    Mt  = torch.from_numpy(M).float()
    y0t = torch.from_numpy(y0).float()

    loader = DataLoader(TensorDataset(Xt, Yt, Mt, y0t),
                        batch_size=batch_size, shuffle=False)

    preds_list, masks_list, targets_list = [], [], []
    with torch.no_grad():
        for xb, yb, mb, y0b in loader:
            xb, y0b = xb.to(device), y0b.to(device)
            pred = model(xb, y0b, y_target=None, y_mask=None, teacher_forcing_ratio=0.0)
            preds_list.append(pred.cpu())
            masks_list.append(mb)
            targets_list.append(yb)

    preds   = torch.cat(preds_list, 0).numpy().squeeze(-1)     # [N, out_len]
    targets = torch.cat(targets_list,0).numpy().squeeze(-1)    # [N, out_len]
    masks   = torch.cat(masks_list,  0).numpy().astype(bool)   # [N, out_len]

    # 역스케일
    y_pred_inv = scaler_y.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)
    y_true_inv = scaler_y.inverse_transform(targets.reshape(-1,1)).reshape(targets.shape)

    return y_pred_inv, y_true_inv, masks

```
