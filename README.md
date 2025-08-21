[Hierachical_Quadratic_Tilting.pdf](https://github.com/user-attachments/files/21681763/Hierachical_Quadratic_Tilting.pdf)
---
### 8.22 -> .pth 파일 저징 및 추론 모델 확장코드는 다음과 같다.
'''
# 기존 모델을 꼭 임포트 해서 불러와야 한다.!

enc_input_dim = len(feature_cols)      
dec_input_dim = 1
hid_dim = int(best_params["units"])
n_layers = int(best_params["lstm_layers"])
dropout = float(best_params.get("lstm_dropout_rate", 0.0))
out_len = your_out_days * 24           

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
model = Seq2Seq(
    enc_input_dim=enc_input_dim,
    dec_input_dim=dec_input_dim,
    hid_dim=hid_dim,
    out_len=out_len,
    n_layers=n_layers,
    dropout=dropout
).to(device)

# 체크포인트 로드
state = torch.load("/mnt/nvme/tilting/src/rs_trial_015.pth", map_location=device) # 맥 환경에서 노트북을 돌릴때는 mps 로 값을 넣어준다. 그 외 경우는 cuda 를 반영
model.load_state_dict(state)
model.eval()

'''