# Week 3 — Bias–Variance Decomposition & Learning Curves

## โจทย์ที่ได้รับ

ศึกษา **Bias–Variance Decomposition** ของ polynomial regression models บนปัญหา regression ที่มีข้อมูลน้อย โดยสุ่มจากการแจกแจงแบบเอกรูปบนช่วง `[-1, 1]`

ต้นฉบับจาก playground ของอาจารย์:
- [Generalization Playground](https://waranyuwongseree.github.io/generalization-playground/)
- [Learning Curve Playground](https://waranyuwongseree.github.io/learning-curve-playground/)

### ฟังก์ชันเป้าหมาย (Target Functions)
1. `f(x) = sin(πx)`
2. `f(x) = x²`
3. `f(x) = x³`

### แบบจำลองที่ต้องเปรียบเทียบ
ทั้งหมดเป็น **polynomial regression** ที่ fit ด้วย Normal Equation:
1. **constant**    : degree 0
2. **linear**      : degree 1
3. **quadratic**   : degree 2
4. **cubic**       : degree 3

### สิ่งที่ต้องทำ
1. คำนวณ **bias²** และ **variance** ด้วย Monte Carlo simulation
2. เปรียบเทียบผลระหว่างโมเดลทั้ง 4 บนทั้ง 3 ฟังก์ชันเป้าหมาย
3. สร้าง **Learning Curve** แสดง `E_in` และ `E_out` เมื่อจำนวนตัวอย่าง `n` เปลี่ยนไป
4. ทดลองเพิ่ม **สัญญาณรบกวน (noise)** ที่ระดับ `σ = 0.0, 0.1, 0.3`
5. ใช้ **Normal Equation** (`numpy.linalg.lstsq`) ในการ fit โมเดล

---

## ทฤษฎีสำคัญ

### Bias-Variance Decomposition

สำหรับโมเดล `g_D(x)` ที่ฝึกจากชุดข้อมูล `D` ค่าคลาดเคลื่อนนอกตัวอย่างเฉลี่ยสามารถแยกได้เป็น:

```
E_D[E_out(g_D)] = E_x[bias(x)² + var(x)] + σ²
```

โดย:

- **แบบจำลองเฉลี่ย (average hypothesis)**: `ḡ(x) = E_D[g_D(x)]`
- **bias²(x)**: `(ḡ(x) − f(x))²` — ความห่างของโมเดลเฉลี่ยจากเป้าหมายจริง
- **var(x)**: `E_D[(g_D(x) − ḡ(x))²]` — ความกว้างหรือความแกว่งของแบบจำลองจากชุดข้อมูลต่างกัน
- **σ²**: noise variance (irreducible error)

### Expected Value บนช่วง `[-1, 1]`

เนื่องจาก `x ~ Uniform[-1, 1]` ความหนาแน่นคือ `p(x) = 1/2` ดังนั้น:

```
E_x[h(x)] = ∫_{-1}^{1} h(x) · (1/2) dx
```

---

## วิธีการคำนวณ

### 1. Bias–Variance Simulation

- สุ่มชุดข้อมูลจำนวนมาก (Monte Carlo) จาก `f(x)` บน `[-1, 1]`
- fit polynomial แต่ละ degree บนแต่ละชุด
- ทำนายบนกริด test points จำนวนมาก
- คำนวณ `ḡ(x)`, `bias²(x)`, `var(x)` แล้วอินทิเกรตบน `[-1, 1]` ด้วย trapezoidal rule

### 2. Learning Curve

- ทดลองกับ `n = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]`
- สำหรับแต่ละ `n` สุ่มหลายชุดข้อมูล แล้วหา `E_in` (MSE บนชุดฝึก) และ `E_out` (MSE บน test set)
- ทดลองกับ noise `σ = 0.0, 0.1, 0.3`

---

## ผลลัพธ์ (N = 2, σ = 0)

### Target: `f(x) = sin(πx)`

| Model     | bias²  | variance | Eout   |
|-----------|--------|----------|--------|
| constant  | 0.5000 | 0.2512   | 0.7512 |
| linear    | 0.2059 | 1.6755   | 1.8813 |
| quadratic | 0.1961 | 0.6871   | 0.8832 |
| cubic     | 0.2066 | 0.7155   | 0.9221 |

### Target: `f(x) = x²`

| Model     | bias²  | variance | Eout   |
|-----------|--------|----------|--------|
| constant  | 0.0889 | 0.0442   | 0.1331 |
| linear    | 0.2004 | 0.3315   | 0.5319 |
| quadratic | 0.0688 | 0.0739   | 0.1428 |
| cubic     | 0.0814 | 0.0825   | 0.1639 |

### Target: `f(x) = x³`

| Model     | bias²  | variance | Eout   |
|-----------|--------|----------|--------|
| constant  | 0.1429 | 0.0725   | 0.2154 |
| linear    | 0.0244 | 0.2307   | 0.2551 |
| quadratic | 0.0447 | 0.0302   | 0.0750 |
| cubic     | 0.0307 | 0.0189   | 0.0496 |

### ข้อสังเกต

- **Constant model กับ `sin(πx)`**: bias² ≈ 0.5, variance ≈ 0.25 ตรงกับค่าในสไลด์ (`E[f] = 0`, `E[f²] = 0.5`)
- **Linear model กับ `sin(πx)`**: bias² ต่ำ (~0.21) แต่ variance สูงมาก (~1.68) สอดคล้องกับที่อาจารย์สอนไว้
- **Complex model ไม่ได้ดีเสมอไป**: ที่ `n=2` quadratic/cubic มี variance สูงกว่าที่คาด เพราะ fit polynomial สูงด้วยข้อมูลน้อยเกิด overfit
- **Learning curve**: เมื่อ `n` เพิ่ม `E_in` และ `E_out` ลู่เข้าหากันที่ `bias² + σ²`

---

## ไฟล์ภาพ (Plots)

| ไฟล์ | คำอธิบาย |
|------|----------|
| `plots/average_fit.png` | 3×4 grid: target (เขียว), sample fits (เทา), average fit `ḡ(x)` (แดงประ) |
| `plots/bias_variance_band.png` | average fit ±1 std band เทียบ target |
| `plots/bias2_var_curves.png` | `bias²(x)` และ `var(x)` สำหรับทุก target–model |
| `plots/learning_curve_sinpix.png` | Learning curve ของ `sin(πx)` แยกตามโมเดลและ noise |
| `plots/learning_curve_x2.png` | Learning curve ของ `x²` |
| `plots/learning_curve_x3.png` | Learning curve ของ `x³` |

---

## วิธีรันโค้ด

```bash
cd "/Users/dolphin/Desktop/Machine_Learning/Assignment/Week3"
python3 bias_variance_lab.py
```

ความต้องการ:
- Python 3
- `numpy`
- `matplotlib`

---

## โครงสร้างโฟลเดอร์

```
Assignment/Week3/
├── bias_variance_lab.py    # โค้ดหลัก
├── index.html              # interactive playground รวมทั้ง 2 ตัว
├── README.md               # คู่มือและอธิบายผลลัพธ์
├── results.json            # ค่าตัวเลขที่คำนวณได้
└── plots/
    ├── average_fit.png
    ├── bias_variance_band.png
    ├── bias2_var_curves.png
    ├── learning_curve_sinpix.png
    ├── learning_curve_x2.png
    └── learning_curve_x3.png
```

---

## อ้างอิง

- สไลด์ Week 3: `Generalization.pdf`
- [Generalization Playground](https://waranyuwongseree.github.io/generalization-playground/)
- [Learning Curve Playground](https://waranyuwongseree.github.io/learning-curve-playground/)
