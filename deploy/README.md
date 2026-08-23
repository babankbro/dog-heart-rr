# การติดตั้งบนเซิร์ฟเวอร์

ทุกคำสั่งในไฟล์นี้รันบนเซิร์ฟเวอร์ ยกเว้นหัวข้อที่บอกว่ารันบนเครื่องตัวเอง

## 0. ก่อนอื่น — ปิดช่องทางรหัสผ่าน

รหัส root ที่ใช้อยู่ควรเปลี่ยน และควรเลิกใช้การล็อกอินด้วยรหัสผ่าน

บนเครื่องตัวเอง สร้างกุญแจแล้วส่งขึ้นไป

```bash
ssh-keygen -t ed25519 -C "ekg-deploy"
ssh-copy-id root@187.127.98.12
```

บนเซิร์ฟเวอร์ เปลี่ยนรหัสแล้วปิด password login

```bash
passwd
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && systemctl restart ssh
```

ทดสอบว่า `ssh root@187.127.98.12` เข้าได้ด้วยกุญแจ **ก่อน** ปิด session เดิม ไม่งั้นล็อกตัวเองออก

## 1. ติดตั้ง Docker

```bash
curl -fsSL https://get.docker.com | sh
```

ตรวจว่ามีที่ว่างพอ — image มี torch อยู่ข้างใน กินราว 4 GB และตอน `pip install` ใช้ RAM
สูงพอสมควร เครื่องที่มี RAM 1 GB มักถูก OOM kill ระหว่าง build

```bash
df -h /var/lib/docker && free -m
```

ถ้า RAM น้อย ให้เพิ่ม swap ชั่วคราวก่อน build

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
```

## 2. ดึงโค้ด

```bash
mkdir -p /root/ekg && cd /root/ekg
git clone https://github.com/babankbro/dog-heart-rr.git .
mkdir -p models data out
```

## 3. ส่ง weights ขึ้นไป

ไฟล์ `.pt` ไม่ได้อยู่ใน git (ไฟล์ละหลายสิบเมกะไบต์) ต้องส่งเอง — **รันบนเครื่องตัวเอง**

```bash
scp models/crop_best.pt models/point_ink_best.pt root@187.127.98.12:/root/ekg/models/
```

ถ้าจะยกภาพเดิมขึ้นไปด้วย (เป็นข้อมูลผู้ป่วย ส่งเท่าที่จำเป็น)

```bash
scp -r data/. root@187.127.98.12:/root/ekg/data/
```

## 4. ตั้งค่า

สร้างรหัสผ่านสำหรับหน้าเว็บ (แอปไม่มีระบบล็อกอินของตัวเอง ด่านนี้คือด่านเดียว)

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'รหัสที่ตั้งเอง'
```

เอา hash ที่ได้มาใส่ใน `.env`

```bash
cat > /root/ekg/.env <<'EOF'
SITE_ADDRESS=ekg.example.com
BASIC_AUTH_USER=vet
BASIC_AUTH_HASH=$2a$14$...วางค่าที่ได้จากคำสั่งข้างบน...
EOF
chmod 600 /root/ekg/.env
```

`SITE_ADDRESS` ตั้งได้ 2 แบบ

| ค่า | ผลลัพธ์ |
|-----|---------|
| `ekg.example.com` | caddy ขอใบรับรองให้เอง ได้ `https://ekg.example.com` — ต้องชี้ A record มาที่ `187.127.98.12` ก่อน |
| `:80` | ได้ `http://187.127.98.12` ล้วน ไม่มี HTTPS **รหัสผ่านจะวิ่งไปแบบไม่เข้ารหัส** ใช้ชั่วคราวเท่านั้น |

## 5. รัน

```bash
cd /root/ekg
docker compose -f docker-compose.prod.yml up -d --build
```

build ครั้งแรกใช้เวลาหลายนาที (โหลด torch) ดูความคืบหน้าและตรวจว่าขึ้นครบ

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f web
```

เปิดพอร์ตบนไฟร์วอลล์ถ้ามี

```bash
ufw allow 80,443/tcp
```

## 6. อัปเดตเวอร์ชันใหม่

```bash
cd /root/ekg && git pull && docker compose -f docker-compose.prod.yml up -d --build
```

---

## ทางเลือก: ลิงก์สาธารณะแบบไม่ต้องมีโดเมนและไม่ต้องเปิดพอร์ต

ถ้ายังไม่มีโดเมน หรือ ISP บล็อกพอร์ต 80/443 ใช้ Cloudflare Tunnel แทน caddy ได้
วิธีนี้ไม่ต้องเปิดพอร์ตขาเข้าเลย เซิร์ฟเวอร์เป็นฝ่ายต่อออกไปเอง

```bash
docker compose -f docker-compose.prod.yml up -d --build web
docker run --rm --network ekg_default cloudflare/cloudflared:latest \
  tunnel --url http://web:8000
```

จะได้ลิงก์ `https://xxxx.trycloudflare.com` พิมพ์ออกมาใน log

ข้อจำกัดที่ต้องรู้: ลิงก์แบบนี้เป็นของชั่วคราว หายเมื่อคำสั่งหยุด และ**ไม่มี basic auth
มากั้น** ใครมีลิงก์ก็เข้าดูภาพผู้ป่วยได้ทั้งหมด ถ้าจะใช้ยาวให้สมัคร Cloudflare Tunnel
แบบมีชื่อ แล้วเปิด Cloudflare Access คุมสิทธิ์
