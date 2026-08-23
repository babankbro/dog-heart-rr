# การติดตั้งบนเซิร์ฟเวอร์

ทุกคำสั่งในไฟล์นี้รันบนเซิร์ฟเวอร์ ยกเว้นหัวข้อที่บอกว่ารันบนเครื่องตัวเอง

## ทางลัด — สคริปต์เดียวจบ

`bootstrap.sh` ทำข้อ 1–5 ให้ทั้งหมด (ติดตั้ง Docker, เพิ่ม swap ถ้า RAM น้อย, clone,
สร้าง `.env`, build, เปิดไฟร์วอลล์, ตรวจว่าขึ้นจริง) รันซ้ำได้ ไม่พังของเดิม

```bash
curl -fsSL https://raw.githubusercontent.com/babankbro/dog-heart-rr/main/deploy/bootstrap.sh -o bootstrap.sh
SITE_ADDRESS=ekg.example.com WEB_USER=vet WEB_PASS='รหัสที่ตั้งเอง' bash bootstrap.sh
```

ยังไม่มีโดเมนให้ใช้ `SITE_ADDRESS=:80` แทน (ไม่มี HTTPS — อ่านข้อ 4 ก่อนตัดสินใจ)

`WEB_PASS` ถูกแปลงเป็น bcrypt hash แล้วทิ้ง ตัวรหัสไม่ถูกเขียนลงดิสก์ แต่จะติดอยู่ใน
history ของ shell ให้ล้างด้วย `history -d` หรือเติมช่องว่างหน้าคำสั่ง

### ไม่อยากตั้งรหัส เปิดสาธารณะเลย

ใช้ `NO_AUTH=1` แทน `WEB_USER`/`WEB_PASS`

```bash
SITE_ADDRESS=:80 NO_AUTH=1 bash bootstrap.sh
```

หมายความว่าใครก็ตามที่รู้ที่อยู่จะเข้าดูภาพ EKG ทั้งหมด ดาวน์โหลด CSV อัปโหลดภาพใหม่
และ**ลบข้อมูลได้** — ตัวแอปไม่มีระบบสิทธิ์ของตัวเอง ทุก endpoint เปิดเท่ากันหมด
เหมาะกับตอนสาธิตหรือทดสอบ ไม่เหมาะกับข้อมูลจริงที่ปล่อยทิ้งไว้

กลับมาใส่รหัสภายหลังได้ตลอด รันสคริปต์ซ้ำพร้อม `WEB_USER=... WEB_PASS=...`

กลไกอยู่ที่โฟลเดอร์ `deploy/auth/` — มีไฟล์ `*.caddy` = ถามรหัส, ว่าง = เปิดสาธารณะ
สลับด้วยมือก็ได้ แล้วสั่ง `docker compose -f docker-compose.prod.yml restart caddy`

ยังต้องทำเองอีกอย่างเดียวคือส่ง weights ขึ้นไป (ข้อ 3) สคริปต์จะเตือนให้ถ้ายังไม่มี

### ถ้าเจอ "address already in use" ที่พอร์ต 80

แปลว่ามีเว็บเซิร์ฟเวอร์อื่นครองอยู่ VPS หลายเจ้าลง nginx หรือ apache มาให้ตั้งแต่แรก
ดูก่อนว่าตัวไหน

```bash
ss -lptn 'sport = :80'
```

แล้วเลือกทางใดทางหนึ่ง

| ทาง | คำสั่ง | ข้อแลกเปลี่ยน |
|-----|--------|--------------|
| ปิดตัวเดิม | `systemctl disable --now nginx` | ตรงไปตรงมาที่สุด ใช้ได้ถ้าไม่ได้ใช้ nginx ทำอย่างอื่น |
| ย้ายพอร์ต | ใส่ `HTTP_PORT=8080 HTTPS_PORT=8443` หน้าคำสั่ง | ได้ `http://ไอพี:8080` แต่**ขอใบรับรอง HTTPS อัตโนมัติไม่ได้** เพราะ Let's Encrypt ยิงมาที่ 80/443 เท่านั้น |
| ให้ nginx เดิม proxy ต่อ | ตั้ง `HTTP_PORT=8080` แล้วเพิ่ม `proxy_pass http://127.0.0.1:8080;` ใน server block ของ nginx | เหมาะเมื่อเครื่องนี้มีเว็บอื่นอยู่ด้วยและอยากใช้โดเมนเดียวกัน |

สคริปต์เวอร์ชันปัจจุบันตรวจให้ตั้งแต่ก่อน build แล้ว จะบอกชื่อโปรเซสที่ครองพอร์ตให้เลย

ข้อ 0 เรื่อง SSH key ควรทำก่อนเสมอ ส่วนข้อ 1–5 ด้านล่างคือรายละเอียดของสิ่งที่สคริปต์ทำ
ไว้อ่านเวลาต้องแก้ทีละขั้น

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
