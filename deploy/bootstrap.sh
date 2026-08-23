#!/usr/bin/env bash
# ติดตั้งและสตาร์ทระบบบนเซิร์ฟเวอร์เปล่า รันซ้ำได้ ไม่พังของเดิม
#
#   SITE_ADDRESS=ekg.example.com WEB_USER=vet WEB_PASS='...' bash bootstrap.sh
#
# SITE_ADDRESS  ชื่อโดเมนที่ชี้มาที่เครื่องนี้ (ได้ HTTPS อัตโนมัติ)
#               หรือ :80 ถ้ายังไม่มีโดเมน (ไม่มี HTTPS รหัสผ่านวิ่งแบบไม่เข้ารหัส)
# WEB_USER      ชื่อผู้ใช้สำหรับเข้าหน้าเว็บ
# WEB_PASS      รหัสผ่านหน้าเว็บ — สคริปต์แปลงเป็น bcrypt hash ให้ ไม่เก็บตัวรหัสลงดิสก์
set -euo pipefail

REPO=https://github.com/babankbro/dog-heart-rr.git
APP_DIR=${APP_DIR:-/root/ekg}
COMPOSE="docker compose -f docker-compose.prod.yml"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mหยุด: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die 'ต้องรันด้วย root'

# ---------- 1. Docker ----------
if command -v docker >/dev/null 2>&1; then
	say "มี Docker อยู่แล้ว ($(docker --version))"
else
	say 'ติดตั้ง Docker'
	curl -fsSL https://get.docker.com | sh
fi
docker compose version >/dev/null 2>&1 || die 'ไม่มี docker compose v2'

# ---------- 2. ทรัพยากร ----------
# build ต้องคอมไพล์/ติดตั้ง torch ซึ่งกิน RAM เยอะ เครื่องเล็กจะโดน OOM kill กลางทาง
mem=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
swap=$(awk '/SwapTotal/{print int($2/1024)}' /proc/meminfo)
say "RAM ${mem} MB, swap ${swap} MB"
if [ "$mem" -lt 2000 ] && [ "$swap" -lt 1000 ] && [ ! -f /swapfile ]; then
	say 'RAM น้อย เพิ่ม swap 2 GB'
	fallocate -l 2G /swapfile
	chmod 600 /swapfile
	mkswap /swapfile >/dev/null
	swapon /swapfile
	grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi

avail=$(df -Pm /var/lib/docker 2>/dev/null | awk 'NR==2{print $4}' || df -Pm / | awk 'NR==2{print $4}')
[ "${avail:-0}" -ge 6000 ] || die "ที่ว่างเหลือ ${avail} MB ต้องมีอย่างน้อย 6000 MB (image มี torch อยู่ข้างใน)"

# ---------- 3. โค้ด ----------
if [ -d "$APP_DIR/.git" ]; then
	say "อัปเดตโค้ดที่ $APP_DIR"
	git -C "$APP_DIR" pull --ff-only
else
	say "clone ลง $APP_DIR"
	mkdir -p "$APP_DIR"
	git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"
mkdir -p models data out

# ---------- 4. weights ----------
# ไฟล์ .pt ไม่ได้อยู่ใน git ต้อง scp ขึ้นมาเอง ถ้าไม่มีระบบยังรันได้แต่ตกไปใช้ anchor อย่างเดียว
missing=''
for w in crop_best.pt point_ink_best.pt; do
	[ -f "models/$w" ] || missing="$missing $w"
done
if [ -n "$missing" ]; then
	printf '\n\033[1;33mเตือน: ไม่พบ weights:%s\033[0m\n' "$missing"
	printf 'ส่งขึ้นมาจากเครื่องตัวเองด้วย:\n'
	printf '  scp models/crop_best.pt models/point_ink_best.pt root@%s:%s/models/\n' \
		"$(hostname -I | awk '{print $1}')" "$APP_DIR"
	printf 'ระบบจะยังสตาร์ทได้ แต่จุด R จะมาจาก image processing อย่างเดียว\n'
fi

# ---------- 5. .env ----------
if [ -f .env ]; then
	say 'ใช้ .env เดิม (ลบทิ้งถ้าอยากตั้งใหม่)'
else
	: "${SITE_ADDRESS:?ต้องตั้ง SITE_ADDRESS เช่น ekg.example.com หรือ :80}"
	: "${WEB_USER:?ต้องตั้ง WEB_USER}"
	: "${WEB_PASS:?ต้องตั้ง WEB_PASS}"
	say 'สร้าง .env'
	hash=$(docker run --rm caddy:2-alpine caddy hash-password --plaintext "$WEB_PASS")
	umask 077
	cat >.env <<-EOF
		SITE_ADDRESS=$SITE_ADDRESS
		BASIC_AUTH_USER=$WEB_USER
		BASIC_AUTH_HASH=$hash
	EOF
	unset WEB_PASS
fi

# ---------- 6. build + start ----------
say 'build (ครั้งแรกใช้เวลาหลายนาที กำลังโหลด torch)'
$COMPOSE up -d --build

# ---------- 7. ไฟร์วอลล์ ----------
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q 'Status: active'; then
	say 'เปิดพอร์ต 80/443 บน ufw'
	ufw allow 80/tcp >/dev/null
	ufw allow 443/tcp >/dev/null
fi

# ---------- 8. ตรวจว่าขึ้นจริง ----------
say 'รอให้บริการพร้อม'
site=$(grep '^SITE_ADDRESS=' .env | cut -d= -f2-)
for _ in $(seq 30); do
	code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost/ || true)
	[ "$code" = '401' ] && break      # 401 คือถูกแล้ว แปลว่า caddy กั้นรหัสผ่านอยู่
	sleep 2
done

$COMPOSE ps
if [ "${code:-}" = '401' ]; then
	case "$site" in
	:*) url="http://$(hostname -I | awk '{print $1}')" ;;
	*) url="https://$site" ;;
	esac
	say "พร้อมใช้งาน: $url  (ล็อกอินด้วยผู้ใช้ที่ตั้งไว้)"
else
	printf '\n\033[1;31mยังไม่ตอบตามที่ควร (ได้ %s) ดู log ด้วย:\033[0m\n' "${code:-ไม่มี}"
	printf '  cd %s && %s logs --tail 50\n' "$APP_DIR" "$COMPOSE"
	exit 1
fi
