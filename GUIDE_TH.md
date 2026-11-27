# คู่มือรันบนเครื่องตัวเอง (Windows 11 + WSL2, RTX 3070 Ti 8 GB)

## 1. ติดตั้ง WSL2 (ครั้งเดียว ~20 นาที, Windows และข้อมูลอยู่ครบ)

1. อัปเดต NVIDIA driver **บน Windows** ให้เป็นตัวล่าสุด (GeForce Experience หรือ nvidia.com)
   - ต้องเป็นเวอร์ชัน ≥ 550
   - **ห้ามลง NVIDIA driver ใน Ubuntu** เพราะ WSL2 ใช้ driver ของ Windows
2. เปิด PowerShell แบบ **Run as Administrator** แล้วรัน:
   ```bash
   wsl --install -d Ubuntu-22.04
   ```
3. รีสตาร์ทเครื่อง แล้วเปิด "Ubuntu 22.04" จาก Start menu และตั้ง username/password
4. ใน Ubuntu เช็คว่ามองเห็นการ์ดจอ ต้องเห็น "NVIDIA GeForce RTX 3070 Ti":
   ```bash
   nvidia-smi
   ```
5. ติดตั้งเครื่องมือพื้นฐาน:
   ```bash
   sudo apt update && sudo apt install -y git curl tmux build-essential python3-pip
   ```

> 💡 **ควรทำ:** ให้ WSL ใช้ RAM ได้มากขึ้น โดยสร้างไฟล์ `C:\Users\<ชื่อคุณ>\.wslconfig` ใส่ข้อความด้านล่าง แล้วรัน `wsl --shutdown` ใน PowerShell หนึ่งครั้ง
> ```
> [wsl2]
> memory=24GB
> ```

## 2. ดึงโค้ดและติดตั้ง env

เก็บโค้ดไว้ในบ้านของ Ubuntu (`~`) ไม่ใช่ใน `/mnt/c/...` เพราะไดรฟ์ Windows ช้ากว่ามาก

```bash
cd ~ && git clone https://github.com/KanutT1999/IFEval.git && cd IFEval
```

```bash
export HF_TOKEN=hf_xxx        # สำหรับ Llama: กด accept license บน huggingface.co ก่อน (ถ้าไม่ใช้ Llama ข้ามได้)
```

```bash
bash setup.sh                 # ~10-15 นาที
```

## 3. รัน

```bash
tmux new -s ifeval            # กันงานหลุดถ้าปิดหน้าต่าง; กลับมาดูด้วย: tmux attach -t ifeval
```

```bash
bash smoke.sh                 # ~3 นาที ถ้าผ่านแปลว่า vLLM ใช้ GPU ได้
```

```bash
bash run_all.sh               # งานจริง ระบบเลือก profile rtx_8gb ให้อัตโนมัติ
```

**ระหว่างรัน:** ปิด Chrome/เกม/โปรแกรมที่ใช้ GPU เพื่อเหลือ VRAM ให้โมเดลมากที่สุด

**ถ้าไม่มี HF_TOKEN:**
```bash
MODEL_SET=rtx_8gb_open bash run_all.sh
```

### ถ้าเจอปัญหา

| อาการ | แก้ |
|---|---|
| `CUDA out of memory` / `No available memory for the cache blocks` | `GPU_MEM_UTIL=0.85 bash run_all.sh` (ถ้ายังไม่ได้ลด 0.80) |
| เฉพาะโมเดล 3B ไม่พอ | ปกติสำหรับการ์ด 8 GB สคริปต์จะข้ามไปเองและบันทึกใน `results/logs/failures.txt` ให้เขียนลงรายงานว่าเป็นข้อจำกัดของฮาร์ดแวร์ |
| vLLM พังตั้งแต่เริ่ม | ใช้ backend สำรอง: `BACKEND=hf bash smoke.sh` แล้ว `BACKEND=hf PHASES="1 2" bash run_all.sh` (ช้ากว่า) |
| ไฟดับหรือปิดเครื่องกลางทาง | สั่ง `bash run_all.sh` ใหม่ ระบบจะข้าม run ที่เสร็จแล้ว |

## 4. เวลาโดยประมาณ (RTX 3070 Ti)

| ขั้น | เวลา |
|---|---|
| setup + smoke | ~20 นาที |
| Phase 1: headline ทุกโมเดล | ~1 ชม. |
| Phase 2: repeats + rewordings | ~3-4 ชม. |
| Phase 3: no-template + sampling | ~3-4 ชม. |
| **รวม** | **~8-10 ชม.** ปล่อยรันข้ามคืนได้ ไม่เสียค่าเช่า |

ถ้าอยากเร็วขึ้น ให้รันแค่ phase 1 กับ 2 ก่อน:

```bash
PHASES="1 2" bash run_all.sh
```

## 5. ผลลัพธ์

- เปิดดูจาก Windows ได้ที่ `\\wsl$\Ubuntu-22.04\home\<user>\IFEval\results\report\REPORT.md`
- กราฟ `.png` อยู่ในโฟลเดอร์เดียวกัน

## 6. เรื่องที่ต้องเขียนในรายงาน

โจทย์พูดถึง V100 แต่เรารันบน RTX 3070 Ti ข้อนี้**ไม่ผิด** เพราะโจทย์ให้คะแนนจากการสร้าง instrument ที่ทำซ้ำได้ แต่ต้องเขียนให้ชัด:

- **ฮาร์ดแวร์:** RTX 3070 Ti 8 GB, WSL2, fp16 ข้อมูลนี้ถูกบันทึกใน manifest อัตโนมัติ และประโยค claim ใน REPORT.md จะใส่ชื่อ GPU จริงให้
- **Confound:** ฮาร์ดแวร์เป็นปัจจัยที่ต่างจากเปเปอร์อีกหนึ่งข้อ
- **ทำไมยังใช้ fp16 ทั้งที่การ์ดรองรับ bf16:** เพื่อให้เทียบกับ V100 และกับคนอื่นที่รันซ้ำได้

**(Optional, ได้คะแนน Harness):** ถ้าภายหลังเช่า V100 ด้วย ให้รัน `IFEVAL_RESULTS=results_v100 PHASES="1" bash run_all.sh` แล้วเทียบผลโมเดลเดียวกันระหว่างสองการ์ด จะได้ finding ว่าฮาร์ดแวร์ทำให้คะแนนแกว่งแค่ไหน
