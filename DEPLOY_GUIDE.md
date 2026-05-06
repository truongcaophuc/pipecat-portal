# Hướng Dẫn Triển Khai (Deployment Guide)

Tài liệu này hướng dẫn cách đưa mã nguồn lên server Linux và chạy bằng Docker.

## 1. Chuẩn Bị
Đảm bảo server Linux của bạn đã cài đặt:
- **Docker**: [Hướng dẫn cài đặt](https://docs.docker.com/engine/install/)
- **Docker Compose**: [Hướng dẫn cài đặt](https://docs.docker.com/compose/install/)

## 2. Copy Mã Nguồn Lên Server
Bạn cần copy toàn bộ thư mục `voice-ai-pipeline` lên server. Có thể dùng `scp` hoặc `rsync` hoặc git clone.

Ví dụ dùng SCP (chạy từ máy tính cá nhân):
```bash
scp -r "d:\Poptech\Server_Voicebot\voice-ai-pipeline" user@<IP_SERVER>:/home/user/app
```

## 3. Cấu Hình Biến Môi Trường
Trước khi chạy, hãy kiểm tra lại các file cấu hình `.env`:

1.  **Backend**: `evn-demo/Evn Outage Demo/server/.env`
    - Đảm bảo các API Key và URL đã chính xác.

2.  **Frontend**: `evn-demo/Evn Outage Demo/web-tet/.env`
    - Đảm bảo `VITE_BOT_START_URL` trỏ về đúng địa chỉ websocket của server (IP hoặc Domain).
    - Ví dụ: `VITE_BOT_START_URL=wss://your-domain.com/ws` hoặc `ws://<IP_SERVER>:7881/ws`

## 4. Chạy Ứng Dụng với Docker Compose
Vào thư mục chứa file `docker-compose.prod.yml` và chạy lệnh:

```bash
cd /home/user/app/voice-ai-pipeline

# Build và chạy ngầm (detached mode)
docker compose -f docker-compose.prod.yml up -d --build
```

## 5. Kiểm Tra
- **Frontend**: Truy cập trình duyệt tại `http://<IP_SERVER>:5173`
- **Backend Port**: 7881 (Nên mở firewall port này nếu dùng ws:// trực tiếp).
- **Log**: Để xem log, dùng lệnh:
    ```bash
    docker compose -f docker-compose.prod.yml logs -f
    ```

## 6. Lưu ý về SSL (HTTPS/WSS)
Nếu bạn dùng `wss://` (Secure WebSocket), bạn cần có chứng chỉ SSL.
- Cách tốt nhất là dùng một **Reverse Proxy** (như Nginx cài trực tiếp trên host hoặc Traefik) để hứng port 80/443, trỏ domain về đó, và proxy pass vào port container (5173 và 7881).
