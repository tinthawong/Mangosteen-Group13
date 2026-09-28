# Mangosteen Ripeness AI

ระบบจำแนกความสุกของมังคุด 3 ระดับ (**Unripe / Ripe / Overripe**) ด้วยโมเดล TensorFlow Lite (int8) ที่รันบนบอร์ด ESP32-S3 โดยตรง แสดงภาพจากกล้องและผลทำนายแบบเรียลไทม์ผ่านหน้าเว็บ

## ผู้จัดทำ — Group13

| รหัสนักศึกษา | ชื่อ-สกุล |
|---|---|
| 67114587 | Thanakrit Inthawong |
| 67124651 | Pirapat Laohasaran |
| 67103150 | Chaiyaphat Onwanna |

## ฮาร์ดแวร์และซอฟต์แวร์

- บอร์ด LilyGO T-SIMCAM (ESP32-S3, PSRAM 8 MB) — ตั้งค่า pin map เป็นเวอร์ชัน V1.2 (`src/config.h`)
- PlatformIO + Arduino framework
- ไลบรารี `esp32-camera`, `TensorFlowLite_ESP32`

## การทำงาน

1. กล้องถ่ายภาพ JPEG ขนาด 640×480
2. ถอดรหัส JPEG เป็น RGB888 → ครอปตรงกลางเป็นสี่เหลี่ยมจัตุรัส → ย่อเป็น 96×96
3. Normalize ค่าสีเป็นช่วง −1 ถึง 1 แล้ว quantize เป็น int8 ส่งเข้าโมเดล
4. โมเดลให้ความน่าจะเป็นของ 3 คลาส เลือกคลาสที่ค่าสูงสุด
5. บอร์ดเปิด Wi-Fi Access Point พร้อมเว็บเซิร์ฟเวอร์
   - Port 80: หน้าเว็บ และ `/result` (JSON ผลทำนาย)
   - Port 81: `/stream` วิดีโอสตรีม MJPEG

## วิธีใช้งาน

1. เปิดโปรเจกต์ด้วย VS Code + PlatformIO
2. แก้ชื่อ/รหัส Wi-Fi และชื่อกลุ่มใน `src/config.h`
3. Build และ Upload ลงบอร์ด
4. ใช้มือถือหรือคอมเชื่อมต่อ Wi-Fi ของบอร์ด แล้วเปิด `http://192.168.4.1/`

## โครงสร้างไฟล์

```
src/
├── main.cpp        กล้อง, AI inference, web server
├── config.h        ตั้งค่า Wi-Fi, ชื่อกลุ่ม, pin map
└── model_data.h    โมเดล TFLite (int8) ในรูป C array
partitions.csv      ตาราง partition (app 6 MB)
platformio.ini      ตั้งค่าบอร์ดและไลบรารี
```

## การปรับปรุงจากเวอร์ชันต้นแบบ

- **แก้การเตรียมภาพเข้าโมเดล:** เวอร์ชันเดิมอ่านบัฟเฟอร์ JPEG เหมือนเป็นภาพ RGB565 ขนาด 96×96 ทำให้โมเดลได้ข้อมูลที่ไม่ใช่พิกเซลจริง แก้โดยถอดรหัส JPEG แล้วครอปและย่อภาพให้ถูกต้อง
- อ่านขนาด input จาก tensor ของโมเดลโดยตรง และตรวจชนิด/รูปร่าง tensor ตอนเริ่มต้น
- ป้องกันการอ่าน/เขียนผลทำนายพร้อมกันระหว่าง task (critical section)
- แยกพอร์ตควบคุมของเว็บเซิร์ฟเวอร์ตัวที่สองไม่ให้ชนกัน
- เปลี่ยน `delay(2000)` เป็นการจับเวลาแบบไม่บล็อก ทำให้ DNS/captive portal ตอบสนองดีขึ้น
- แสดงเวลา inference บนหน้าเว็บและ Serial Monitor
- ลบ `#define CAM_RESET_PIN` ที่ซ้ำกันใน `config.h`

## เครดิต

ต่อยอดจากโปรเจกต์ [Mangosteen3.1](https://github.com/BallThirakun/Mangosteen3.1) ซึ่งพัฒนาร่วมกับกลุ่มของ BallThirakun
