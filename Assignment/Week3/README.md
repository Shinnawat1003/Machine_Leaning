# Week 3 — Bias-Variance Decomposition & Learning Curves

## โจทย์ที่ได้รับ

ศึกษา **Bias–Variance Decomposition** ของโมเดล Machine Learning 3 แบบ บนปัญหา regression ที่มีข้อมูลน้อย (N = 2) โดยสุ่มจากการแจกแจงแบบเอกรูปบนช่วง `[-1, 1]`

### ฟังก์ชันเป้าหมาย (Target Functions)
1. `f(x) = sin(πx)`
2. `f(x) = x²`

### แบบจำลองที่ต้องเปรียบเทียบ
1. **Constant model**: `h(x) = b`
2. **Linear model**: `h(x) = w₀ + w₁x`
3. **Linear model through origin**: `h(x) = wx`

### สิ่งที่ต้องทำ
1. คำนวณ **bias²** และ **variance** ด้วยวิธี **analytical** และ **simulation**
2. เปรียบเทียบผลระหว่างโมเดลทั้ง 3 บนทั้ง 2 ฟังก์ชันเป้าหมาย
3. สร้าง **Learning Curve** แสดง `Ein` และ `Eout` เมื่อจำนวนตัวอย่าง `n` เปลี่ยนไป
4. ทดลองเพิ่ม **สัญญาณรบกวน (noise)** ที่ระดับ `σ = 0.0, 0.1, 0.3`
5. ใช้ **Normal Equation** (`numpy.linalg.lstsq`) ในการ fit โมเดล ไม่ใช้ Gradient Descent

---

## ทฤษฎีสำคัญ

### Bias-Variance Decomposition

สำหรับโมเดล `g_D(x)` ที่ฝึกจากชุดข้อมูล `D` ค่าคลาดเคลื่อนนอกตัวอย่างเฉลี่ยสามารถแยกได้เป็น:

```
E_D[E_out(g_D)] = E_x[bias(x)² + var(x)]
```

โดย:

- **แบบจำลองเฉลี่ย (average hypothesis)**: `ḡ(x) = E_D[g_D(x)]`
- **bias²(x)**: `(ḡ(x) − f(x))²` — ความห่างของโมเดลเฉลี่ยจากเป้าหมายจริง
- **var(x)**: `E_D[(g_D(x) − ḡ(x))²]` — ความกว้างหรือความแกว่งของโมเดลจากชุดข้อมูลต่างกัน

### Expected Value บนช่วง `[-1, 1]`

เนื่องจาก `x ~ Uniform[-1, 1]` ความหนาแน่นคือ `p(x) = 1/2` ดังนั้น:

```
E_x[h(x)] = ∫_{-1}^{1} h(x) · (1/2) dx
```

ส่วน `E_D[·]` คือค่าคาดหมายเหนือชุดข้อมูล `D = {(x₁, f(x₁)), (x₂, f(x₂))}` ที่สุ่มมา

---

## วิธีการคำนวณ

### 1. Analytical Method

- **Constant model**: คำนวณแบบปิดสมบูรณ์ได้โดยตรงจาก `E[f]` และ `E[f²]` แล้วใช้ trapezoidal rule ประมาณค่าอินทิเกรต
- **Linear / Linear through origin**: ใช้ numerical integration โดย:
  - สร้างกริด `x` บน `[-1, 1]`
  - สุ่มชุดข้อมูล `(x₁, x₂)` จำนวนมาก
  - คำนวณ `g_D(x)` สำหรับทุก `x` และทุกชุดข้อมูล
  - หา `ḡ(x)` และ `E_D[g_D(x)²]` จากค่าเฉลี่ย Monte Carlo
  - อินทิเกรต `bias²(x)` และ `var(x)` บน `x` ด้วย trapezoidal rule

### 2. Simulation Method

- สุ่มชุดข้อมูลจำนวนมาก (Monte Carlo)
- fit โมเดลบนแต่ละชุด
- ทำนายบนกริด test points
- คำนวณ `ḡ(x)`, `bias²`, และ `variance` โดยตรงจากตัวอย่าง

### 3. Learning Curve

- ทดลองกับ `n = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]`
- สำหรับแต่ละ `n` สุ่มหลายชุดข้อมูล แล้วหา `Ein` (MSE บนชุดฝึก) และ `Eout` (MSE บน test set)
- ทดลองกับ noise `σ = 0.0, 0.1, 0.3` โดยเพิ่ม `ε ~ N(0, σ²)` เข้าไปใน `y`

---

## ผลลัพธ์ (N = 2)

### Target: `f(x) = sin(πx)`

| Model | bias² (ana) | variance (ana) | Eout (ana) | Eout (sim) |
|-------|------------:|---------------:|-----------:|-----------:|
| Constant | 0.5000 | 0.2500 | 0.7500 | 0.7510 |
| Linear | 0.2060 | 1.6703 | 1.8763 | 1.8726 |
| Linear through origin | 0.2718 | 0.2372 | 0.5090 | 0.5052 |

### Target: `f(x) = x²`

| Model | bias² (ana) | variance (ana) | Eout (ana) | Eout (sim) |
|-------|------------:|---------------:|-----------:|-----------:|-----------:|
| Constant | 0.0889 | 0.0444 | 0.1333 | 0.1334 |
| Linear | 0.1998 | 0.3318 | 0.5315 | 0.5314 |
| Linear through origin | 0.2000 | 0.1147 | 0.3147 | 0.3164 |

### ข้อสังเกต

- **Constant model กับ `sin(πx)`**: bias² = 0.5, variance = 0.25 ตรงกับค่าในสไลด์ (`E[f] = 0`, `E[f²] = 0.5`)
- **Linear model กับ `sin(πx)`**: bias² ต่ำ (~0.21) แต่ variance สูงมาก (~1.67) สอดคล้องกับที่อาจารย์สอนไว้ว่า linear regression บนข้อมูล 2 จุดมี variance สูง
- **Linear through origin**: มักให้ variance ต่ำกว่า Linear ธรรมดา เพราะมีพารามิเตอร์น้อยกว่า แต่ bias² สูงกว่าเนื่องจากถูกบังคับผ่าน `(0, 0)`
- **Constant model กับ `x²`**: ทำงานได้ดีกว่าที่คาด (Eout ต่ำสุด) เพราะ `x²` สมมาตรบน `[-1, 1]` ค่าเฉลี่ยคงที่ `E[x²] = 1/3` ใกล้เคียงกับฟังก์ชันจริงเมื่อเฉลี่ยทั้งช่วง

---

## ไฟล์ภาพ (Plots)

### 1. `plots/average_fit.png`

ภาพรวม 2 × 3 แสดง:
- **แถวบน**: target `sin(πx)`
- **แถวล่าง**: target `x²`
- **คอลัมน์**: Constant, Linear, Linear through origin

ในแต่ละกราฟ:
- **เส้นเขียว**: ฟังก์ชันเป้าหมาย `f(x)`
- **เส้นประแดง**: `ḡ(x)` จาก simulation (แบบจำลองเฉลี่ย)
- **เส้นสีเทาบาง ๆ**: ตัวอย่าง hypothesis จากชุดข้อมูลสุ่ม 20 ชุด แสดงให้เห็นความแปรปรวนของแบบจำลอง

**สิ่งที่เห็นได้ชัด**: Linear model มีเส้นสีเทากระจายกว้างมาก (variance สูง) โดยเฉพาะกับ `sin(πx)` ในขณะที่ Constant model มีเส้นที่ค่อนข้างแน่น

### 2. `plots/learning_curve_sin(pi*x).png`

Learning curve ของ `sin(πx)` แยกเป็น 3 กราฟตามโมเดล แสดง:
- **Ein** (เส้นประ): ค่าคลาดเคลื่อนบนชุดฝึก
- **Eout** (เส้นทึบ): ค่าคลาดเคลื่อนบนข้อมูลใหม่
- สีต่าง ๆ แทนระดับ noise `σ = 0.0, 0.1, 0.3`

**สิ่งที่สังเกต**:
- ที่ `n = 2` Linear model มี `Ein ≈ 0` เพราะเส้นตรงผ่าน 2 จุดพอดี
- แต่ `Eout` ของ Linear สูงมากเมื่อ `n` น้อยและมี noise สูง (overfit)
- เมื่อ `n` เพิ่มขึ้น `Ein` และ `Eout` ลู่เข้าหากัน

### 3. `plots/learning_curve_x2.png`

Learning curve ของ `x²` ในรูปแบบเดียวกัน

**สิ่งที่สังเกต**:
- Constant model มี `Eout` ต่ำมากตั้งแต่ `n` น้อย สอดคล้องกับผล bias-variance
- Linear model มี `Eout` สูงช่วง `n` น้อยเมื่อมี noise เนื่องจาก variance สูง
- Linear through origin มีพฤติกรรมระหว่าง Constant และ Linear

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

ไม่จำเป็นต้องติดตั้ง `scipy` เพราะใช้ numerical integration ด้วย `numpy` อย่างเดียว

---

## โครงสร้างโฟลเดอร์

```
Assignment/Week3/
├── bias_variance_lab.py    # โค้ดหลัก
├── README.md               # คู่มือและอธิบายผลลัพธ์
└── plots/
    ├── average_fit.png
    ├── learning_curve_sin(pi*x).png
    └── learning_curve_x2.png
```

---

## อ้างอิง

- สไลด์ Week 3: `Generalization.pdf`
- Wikipedia: Expected value — https://en.wikipedia.org/wiki/Expected_value
