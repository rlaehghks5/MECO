from glob import glob
import os
import shutil
from os.path import join
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio import load
from tqdm import tqdm
from soundfile import write

from pesq import pesq
from pystoi import stoi

from flowmse.model_MeCo import VFModel
from flowmse.data_module_multichannel import SpecsDataModule
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import ensure_dir, energy_ratios, print_mean_std

# ==============================================================================
# [FIX] PyTorch 2.6+ 호환성 패치
# ==============================================================================
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load
# ==============================================================================

def _auto_pick_solver(model, user_choice):
    if user_choice is not None:
        return user_choice
    use_mf = getattr(model, "use_mfse", False)
    return "euler_mf" if use_mf else "euler"

def _safe_resample(wav, sr_in, sr_out):
    if sr_in == sr_out:
        return wav
    return torchaudio.transforms.Resample(sr_in, sr_out)(wav)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, default='', 
                        help='Directory containing the test data')
    parser.add_argument("--folder_destination", type=str, default="",
                        help="Destination path of inference results.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to model checkpoint (.ckpt).")
    parser.add_argument("--odesolver_type", type=str, choices=("white",), default="white",
                        help="Sampler family. We use 'white' (Euler family).")
    parser.add_argument("--odesolver", type=str, choices=("euler", "euler_mf"), default="euler_mf",
                        help="Numerical integrator. Default auto pick based on checkpoint config.")
    parser.add_argument("--reverse_starting_point", type=float, default=1.0,
                        help="Starting point t_N in the reverse ODE default 1.0.")
    parser.add_argument("--last_eval_point", type=float, default=0.03,
                        help="Terminal time t_eps for numerical stability default 0.03.")
    parser.add_argument("--N", type=int, default=1, help="Number of time steps multi-step.")
    parser.add_argument("--one_step", action="store_true",
                        help="Use 1-NFE displacement MeanFlow. If set, N is ignored and N=1.")
    parser.add_argument("--N_mid", type=int, default=0, help="It is not related to FlowSE")
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for prior sampling")

    parser.add_argument("--test_mix_dir", type=str, default='')

    args = parser.parse_args()
    mixture_files = []
    noisy_files = []  # Separator Output (_p)
    clean_files = []  # Ground Truth (Clean)
    format = "wsj_hierarchical"
    if format == "default":
        # (기존 코드 유지)
        noisy_files1 = sorted(glob(os.path.join(args.test_dir, '*_source1hatP.wav')))
        clean_files1 = [item.replace('_source1hatP.wav', '_source1.wav') for item in noisy_files1]
        mixture_files1 = [item.replace('_source1hatP.wav', '_mix.wav') for item in noisy_files1]
        noisy_files2 = sorted(glob(os.path.join(args.test_dir, '*_source2hatP.wav')))
        clean_files2 = [item.replace('_source2hatP.wav', '_source2.wav') for item in noisy_files2]
        mixture_files2 = [item.replace('_source2hatP.wav', '_mix.wav') for item in noisy_files2]
        
        mixture_files = [*mixture_files1, *mixture_files2]
        noisy_files = [*noisy_files1, *noisy_files2]
        clean_files = [*clean_files1, *clean_files2]

    elif format == "wsj_hierarchical":
        # 대상 폴더 목록
        target_spks = ['spk_1', 'spk_2', 'spk_3']
        
        spk_to_mix_folder = {
            'spk_1': '1speaker_reverb',
            'spk_2': '2speaker_reverb',
            'spk_3': '3speaker_reverb'
        }
        
        for spk_name in target_spks:
            spk_path = os.path.join(args.test_dir, spk_name)
            if not os.path.exists(spk_path):
                continue
            
            # 폴더 이름에서 화자 수 추출
            try:
                num_speakers = int(spk_name.split('_')[-1])
            except ValueError:
                print(f"Warning: Could not parse speaker count from {spk_name}. Skipping.")
                continue

            # Case 폴더 나열 (예: 01aa0108_m0p2374)
            case_dirs = sorted([d for d in os.listdir(spk_path) if os.path.isdir(os.path.join(spk_path, d))])
            
            for case_dir in case_dirs:
                full_case_path = os.path.join(spk_path, case_dir)
                
                # 1. Clean/Est 파일을 찾기 위해 기존 폴더에서 prefix(예: 95) 추출
                #    기존 폴더에 있는 _mix.wav 파일을 기준으로 prefix를 따옵니다.
                mix_candidates_original = glob(os.path.join(full_case_path, '*_mix.wav'))
                if not mix_candidates_original:
                    continue
                
                # 예: 457_mix.wav -> 457
                original_mix_filename = os.path.basename(mix_candidates_original[0])
                file_prefix = original_mix_filename.replace('_mix.wav', '')
                
                # 2. 실제 사용할 Mixture 경로 설정
                if args.test_mix_dir is not None:
                    # mix_dir가 제공된 경우: 다른 경로에서 mix.wav 찾기
                    # 구조: mix_dir / 1speaker_reverb / case_dir / mix.wav
                    mix_folder_name = spk_to_mix_folder.get(spk_name)
                    if mix_folder_name is None:
                            print(f"Warning: No mapping for {spk_name}. Skipping mixture path override.")
                            continue
                            
                    final_mix_path = os.path.join(args.test_mix_dir, mix_folder_name, case_dir, 'mix.wav')
                else:
                    # 기존 방식대로
                    final_mix_path = mix_candidates_original[0]

                    
                if not os.path.exists(final_mix_path):
                    continue

                # 화자 수만큼 반복 (spk1, spk2, spk3 ...)
                for k in range(1, num_speakers + 1):
                    # Clean/Est 파일명 패턴 (기존 폴더 사용)
                    est_filename = f"{file_prefix}_spk{k}_p.wav"
                    clean_filename = f"{file_prefix}_spk{k}.wav"
                    
                    est_file_path = os.path.join(full_case_path, est_filename)
                    clean_file_path = os.path.join(full_case_path, clean_filename)

                    # 쌍이 모두 존재할 경우에만 추가
                    if os.path.exists(est_file_path) and os.path.exists(clean_file_path):
                        mixture_files.append(final_mix_path)  # 교체된 경로 혹은 기존 경로
                        noisy_files.append(est_file_path)     # 기존 경로
                        clean_files.append(clean_file_path)   # 기존 경로

    else:
        raise NotImplementedError(f"Directory format {format} unknown!")
    
    if args.debug:
        clean_files = clean_files[:]
        noisy_files = noisy_files[:]
        mixture_files = mixture_files[:]
    print(len(clean_files), len(noisy_files), len(mixture_files))

    target_dir = args.folder_destination
    files_dir = join(target_dir, "files")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(files_dir, exist_ok=True)  

    checkpoint_file = args.ckpt
    sr = 16000
    odesolver_type = args.odesolver_type
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = VFModel.load_from_checkpoint(
        checkpoint_file,
        data_module_cls=SpecsDataModule,
        map_location=device
    )
    model = model.eval(no_ema=False).to(device)
    
    # 결과 폴더 생성
    target_dir = f"{args.folder_destination}/"
    ensure_dir(target_dir)
    ensure_dir(join(target_dir, "files"))

    # Solver 설정
    solver_name = _auto_pick_solver(model, args.odesolver)
    N_eff = 1 if args.one_step else int(args.N)
    
    reverse_starting_point = float(args.reverse_starting_point)
    reverse_end_point = float(args.last_eval_point)

    # Time horizon 설정
    model.T_rev = reverse_starting_point
    if hasattr(model, "ode"):
        model.ode.T_rev = reverse_starting_point

    if args.one_step and solver_name == "euler_mf" and reverse_end_point > 0:
        print(f"[Warn] 1-NFE + euler_mf, last_eval_point={reverse_end_point}. "
              f"Strict 1-step (1->0) requires --last_eval_point 0.0.")

    # 메트릭 저장용 딕셔너리
    data = {
        "filename": [], 
        "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": [],
        "pesq_pred": [], "estoi_pred": [], "si_sdr_pred": [], "si_sir_pred": [], "si_sar_pred": []
    }

    with torch.no_grad():
        for clean_file, noisy_file, mixture_file in tqdm(zip(clean_files, noisy_files, mixture_files)):        

            filename = noisy_file.split('/')[-1]
            # Load wav
            x, sr_ = torchaudio.load(clean_file)
            if sr_ != sr:
                x = torchaudio.transforms.Resample(sr_, sr)(x)
            y, sr_ = torchaudio.load(noisy_file)
            if sr_ != sr:
                y = torchaudio.transforms.Resample(sr_, sr)(y)
            m, sr_ = torchaudio.load(mixture_file)
            if sr_ != sr:
                m = torchaudio.transforms.Resample(sr_, sr)(m)
            # print(x.shape,y.shape,m.shape)
            min_leng = min(x.shape[-1],y.shape[-1],m.shape[-1])
            x = x[...,:min_leng]
            y = y[...,:min_leng]
            m = m[...,:min_leng]

            T_orig = y.size(1)
            norm_factor = y.abs().max().item()
            if norm_factor < 1e-12: norm_factor = 1.0
            
            # 모델 입력 준비 (Noisy)
            y_in = y / norm_factor
            Y = torch.unsqueeze(model._forward_transform(model._stft(y_in.to(device))), 0)
            Y = pad_spec(Y)

            m_in = m / norm_factor
            M = torch.unsqueeze(model._forward_transform(model._stft(m_in.to(device))), 0)
            M = pad_spec(M)

            if odesolver_type == "white":
                sampler = get_white_box_solver(
                    odesolver_name=solver_name,
                    ode=model.ode,
                    VF_fn=model,            # VF_fn으로 모델 전달
                    Y=Y,
                    Y_prior=Y,              # Y_prior는 Y와 동일하게
                    T_rev=reverse_starting_point,
                    t_eps=reverse_end_point,
                    N=N_eff,
                    M=M                  # 명시적으로 None 전달 (필요 시 마스크 전달)
                )
            else:
                raise ValueError(f"{odesolver_type} is not a valid sampler type!")

            # Sampling
            sample, _ = sampler()
            sample = sample.squeeze()
            
            # Audio 복원
            x_hat = model.to_audio(sample, T_orig)
            x_hat = x_hat * norm_factor

            # 최종 길이 정렬 (생성된 오디오와 원본 간 미세 차이 보정)
            min_len_final = min(x.shape[-1], y.shape[-1], x_hat.shape[-1])
            x = x[..., :min_len_final]
            y = y[..., :min_len_final]
            m = m[..., :min_len_final] # mix도 저장해야 하므로 자름
            x_hat = x_hat[..., :min_len_final]

            # Wav 저장
            save_name_base = os.path.splitext(filename)[0]
            
            # 1. Enhanced
            write(join(target_dir, "files", f"{save_name_base}.wav"), x_hat.squeeze().cpu().numpy(), sr)
            # 2. Ref (Clean)
            write(join(target_dir, "files", f"{save_name_base}_ref.wav"), x.squeeze().cpu().numpy(), sr)
            # 3. Mix
            m=m[0]
            write(join(target_dir, "files", f"{save_name_base}_mix.wav"), m.squeeze().cpu().numpy(), sr)
            # 4. Pred (Noisy input)
            write(join(target_dir, "files", f"{save_name_base}_pred.wav"), y.squeeze().cpu().numpy(), sr)

            # 메트릭 계산을 위한 Numpy 변환
            x_np = x.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()     # Noisy
            x_hat_np = x_hat.squeeze().cpu().numpy() # Enhanced
            
            # SI-SDR 계산용 noise (enhanced 관점이 아닌, 일반적인 metric 계산 루틴)
            # eval-geco.py에서는 energy_ratios(estimated, target, noise_component)를 사용
            # Clean(x)을 Target으로 봄.
            
            # 1. Enhanced vs Clean
            n_np = y_np - x_np # Noise component estimate (approx) or just use residual
            
            data["filename"].append(filename)
            
            # PESQ
            try:
                p = pesq(sr, x_np, x_hat_np, 'wb')
            except:
                p = float("nan")
            data["pesq"].append(p)
            data["estoi"].append(stoi(x_np, x_hat_np, sr, extended=True))
            
            # SI-SDR/SIR/SAR
            sdr, sir, sar = energy_ratios(x_hat_np, x_np, n_np)
            data["si_sdr"].append(sdr)
            data["si_sir"].append(sir)
            data["si_sar"].append(sar)

            # 2. Noisy(Pred) vs Clean
            try:
                p_pred = pesq(sr, x_np, y_np, 'wb')
            except:
                p_pred = float("nan")
            data["pesq_pred"].append(p_pred)
            data["estoi_pred"].append(stoi(x_np, y_np, sr, extended=True))
            
            sdr_p, sir_p, sar_p = energy_ratios(y_np, x_np, n_np)
            data["si_sdr_pred"].append(sdr_p)
            data["si_sir_pred"].append(sir_p)
            data["si_sar_pred"].append(sar_p)

    # 결과 CSV 저장
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # 평균 결과 텍스트 저장
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))
        file.write("PESQ_pred: {} \n".format(print_mean_std(data["pesq_pred"])))
        file.write("ESTOI_pred: {} \n".format(print_mean_std(data["estoi_pred"])))
        file.write("SI-SDR_pred: {} \n".format(print_mean_std(data["si_sdr_pred"])))
        file.write("SI-SIR_pred: {} \n".format(print_mean_std(data["si_sir_pred"])))
        file.write("SI-SAR_pred: {} \n".format(print_mean_std(data["si_sar_pred"])))

    # 설정 저장
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write(f"checkpoint file: {checkpoint_file}\n")
        file.write(f"odesolver: {solver_name}\n")
        file.write(f"N: {N_eff}\n")
        file.write(f"Reverse starting point: {reverse_starting_point}\n")
        file.write(f"Last eval point: {reverse_end_point}\n")
        file.write(f"One step: {args.one_step}\n")

    print(f"Done! Results saved to {target_dir}")
