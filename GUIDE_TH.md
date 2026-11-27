# คู่มือรันบน V100 (ภาษาไทย)

## 1. เช่า GPU บน Vast.ai (ถูกที่สุดสำหรับ V100)

1. สมัครที่ <https://vast.ai> แล้วเติมเครดิต ราว $10–15 ก็พอ
   - V100 บน Vast มักอยู่ที่ราว $0.10–0.70 ต่อชั่วโมง ราคาขึ้นกับเครื่อง
2. เพิ่ม SSH public key ที่ Account → SSH Keys
   - ถ้ายังไม่มี key สร้างด้วย `ssh-keygen -t ed25519`
3. หน้า Search ตั้ง filter ดังนี้

   | ตัวกรอง | ค่า |
   |---|---|
   | GPU | V100 (เลือก 16GB หรือ 32GB ก็ได้ ถ้าได้ 32GB ราคาใกล้กัน ให้เลือก 32GB แล้วใช้ `MODEL_SET=v100_32gb`) |
   | จำนวน GPU | 1 |
   | Max CUDA | ≥ 12.4 (ต้องการ driver ≥ 550) |
   | Disk | ≥ 80 GB |
   | Reliability | ≥ 98% |
   | Internet download | ≥ 200 Mbps |

4. Template / Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` แบบ SSH launch mode
5. กด Rent แล้วคัดลอกคำสั่ง SSH จากหน้า Instances

ตัวเลือกสำรองที่น่าเชื่อถือแต่แพงกว่า:
- **Google Cloud** (V100 ราว $2.5/ชม.)
- **AWS p3.2xlarge** (ราว $3/ชม.)

> ⚠️ **อย่าลืม Destroy instance หลังดาวน์โหลดผลเสร็จ** ถ้าแค่ Stop ระบบยังคิดค่า disk อยู่

## 2. ส่งโค้ดขึ้นเครื่อง

ใช้ได้ 2 วิธี

**วิธี A: rsync** รันบน Mac ของคุณ โดยแก้ PORT และ IP ตามหน้า Instances

```bash
rsync -avz -e "ssh -p PORT" --exclude .venv-local --exclude results ./ifeval-v100 root@IP:/workspace/
```

**วิธี B: GitHub private repo** โดย push ขึ้น repo ของคุณก่อน แล้ว `git clone` บนเครื่องเช่า

## 3. รัน (ใช้ `tmux` เพื่อไม่ให้งานหลุดตอน SSH ขาด)

```bash
ssh -p PORT root@IP
tmux new -s ifeval
cd /workspace/ifeval-v100
export HF_TOKEN=hf_xxx        # ถ้าจะรัน Llama ต้องกด accept license บน huggingface.co ก่อน
bash setup.sh                 # ~10-15 นาที
bash smoke.sh                 # ~3 นาที ถ้าผ่านแปลว่า vLLM ใช้ได้บน V100
bash run_all.sh               # งานจริง
```

- ถ้า SSH หลุด ให้ `ssh` เข้าใหม่แล้วพิมพ์ `tmux attach -t ifeval`
- ถ้ารันพังกลางทาง สั่ง `bash run_all.sh` ซ้ำได้เลย ระบบจะข้าม run ที่เสร็จแล้ว

**ถ้า smoke.sh พังที่ vLLM** (เช่น error เรื่อง compute capability หรือ xformers) ให้ใช้ backend สำรอง:

```bash
BACKEND=hf bash smoke.sh
BACKEND=hf PHASES="1 2" bash run_all.sh
```

backend นี้ช้ากว่า vLLM ราว 3-5 เท่า จึงตัด phase 3 ออก

## 4. งบเวลา (เป้าหมาย ≤ 10 ชม.)

| ขั้น | เวลาโดยประมาณ (vLLM) |
|---|---|
| setup + smoke | ~20 นาที |
| Phase 1: headline ทุกโมเดล (6 โมเดล × 1 run) | ~30-45 นาที |
| Phase 2: repeats ×2 + rewordings ×3 (6 × 5 runs) | ~2-3 ชม. |
| Phase 3: no-template + sampling ×5 (6 × 6 runs) | ~2-3 ชม. |
| validate + analyze | ~5 นาที |
| **รวม** | **~5-7 ชม.** (เหลือ buffer ~3 ชม.) |

- ทุกครั้งที่จบ run สคริปต์จะพิมพ์ `elapsed` ออกมา ให้ดูประกอบ
- ถ้าเวลาไม่พอ ให้กด Ctrl-C แล้วสั่ง `python -m ifeval_v100.analyze` ได้เลย เพราะ phase 1 ครบทุกโมเดลตั้งแต่ต้นแล้ว

## 5. ดาวน์โหลดผลกลับ

รันบน Mac:

```bash
rsync -avz -e "ssh -p PORT" root@IP:/workspace/ifeval-v100/results ./ifeval-v100/
```

ผลหลักอยู่ที่ `results/report/REPORT.md` และกราฟ `.png` ในโฟลเดอร์เดียวกัน

## 6. สิ่งที่ต้องเขียนเองในรายงาน (40% ส่วน Findings)

`REPORT.md` สร้างตาราง ช่วงความเชื่อมั่น และประโยค claim ให้อัตโนมัติแล้ว ส่วนที่คุณต้องเขียนเองคือการตีความ:

- คำสั่งประเภทไหนพังเมื่อโมเดลเล็กลง (ดูข้อ 3 และ heatmap)
- benchmark ยังแยกความเก่งระหว่างโมเดลเล็กได้ไหม (ดูข้อ 7)
- gap กับเปเปอร์ปนอยู่กับปัจจัยอะไรบ้าง (ข้อ 6) และส่วนที่ยังอธิบายไม่ได้ (residual, ข้อ 8b)
