# Real-time Fashion Recommender System

## Kiến trúc hệ thống

### 1. Lớp Giao diện (Frontend / Presentation Layer)
Đảm nhiệm vai trò tương tác trực tiếp với người dùng, yêu cầu tốc độ phản hồi UI mượt mà và quản lý trạng thái phiên làm việc (session).

- Công nghệ: Next.js (App Router, React) + Tailwind CSS.

- Môi trường chạy: Localhost (Cổng 3000).

- Nhiệm vụ cốt lõi:

+ Giao diện: Render danh sách sản phẩm dạng Grid và khu vực "Gợi ý cá nhân hóa" (Sidebar/Panel) với các bộ lọc (Filters) động.

+ Client-side State Management: Tự động tạo và quản lý session_id (UUID) lưu trong Local Storage cho mỗi người dùng nặc danh.

+ Event-driven Updates: Bắt các sự kiện tìm kiếm (Search/Filter) và tương tác (Click) để gọi API Backend, đồng thời cập nhật UI tức thời (sử dụng Skeleton Loading trong lúc chờ API).

### 2. Lớp Máy chủ Xử lý (Backend / API Layer)
Đóng vai trò là "Bộ não" của hệ thống, xử lý toàn bộ logic tìm kiếm, cập nhật mô hình AI và gợi ý cá nhân hóa.

- Công nghệ: Python + FastAPI + PyTorch.

- Môi trường chạy: Máy tính cá nhân (Localhost, Cổng 8000).

- Kiến trúc In-Memory (RAM Cache): Khi khởi động (startup event), Backend tự động nạp các model (FashionCLIP text-encoder) và Dictionary chứa vector embedding/metadata vào RAM để đảm bảo tính toán ở độ trễ < 20ms.

- Các luồng API chính:

+ GET /search: Nhận query (text) và filters. Trích xuất vector 512-dim qua FashionCLIP và gọi Weaviate để thực hiện Hybrid Search (Vector + BM25). Trả về danh sách sản phẩm hiển thị.

+ POST /interact: Nhận session_id và article_id (sản phẩm user vừa click). Đẩy task chạy ngầm (Background Task) để: Cập nhật lịch sử click vào Redis, lấy mô hình Personal MLP của user từ Redis (hoặc tạo mới từ Base MLP), fine-tune model bằng thuật toán SGD (Triplet Loss), lưu đè model đã cập nhật lên Redis.

+ GET /recommend: Truy xuất Weaviate (Stage 1 - Candidate Retrieval) để lấy Top 100 ứng viên dựa trên vector EMA của user. Sau đó chiếu 100 ứng viên này qua mô hình Personal MLP hiện tại (Stage 2 - Re-ranking) và tính k-NN để trả về Top 10 sản phẩm phù hợp nhất.

### 3. Lớp Dữ liệu & Lưu trữ (Data & Storage Layer)
Nơi lưu trữ tri thức tĩnh (danh mục sản phẩm, embeddings) và trạng thái động (lịch sử user, trọng số model cá nhân).

#### 3.1. Vector Database (Weaviate Cloud)
Được cấu hình để phục vụ đồng thời truy vấn Semantic và Candidate Generation.

- Collection Product: Phục vụ Hybrid Search.

- Dữ liệu: Vector 512-dim (FashionCLIP) + Metadata (Tên, Giá, Loại, Màu sắc, Link ảnh).

- Tokenization: Hỗ trợ BM25 (Keyword matching) và Exact Filtering cho metadata.

- Collection ProductRec: Phục vụ Recommendation.

- Dữ liệu: Vector 64-dim (từ Student MLP) + article_id.

#### 3.2. State Management (Redis - Local Docker)
Lưu trữ "bộ não" cá nhân hóa với tốc độ truy xuất In-memory. Để khởi chạy local, sử dụng lệnh `docker-compose up -d redis`.

- rec:state:{session_id}: JSON document lưu trữ thông tin session (EMA vector, lịch sử tương tác, metadata).
- rec:model:{session_id}: Chuỗi nhị phân (Base64) chứa trọng số (weights) của mạng Personal MLP dành riêng cho từng user.
- rec:lock:{session_id}: Khóa ngắn hạn chống đụng độ khi chạy adaptation.

#### 3.3. Image Hosting (Cloudinary)
- Vai trò: CDN chuyên dụng để phân phối hình ảnh tĩnh.

- Cách hoạt động: Metadata lưu trữ đường dẫn secure_url. Trình duyệt (thông qua component <Image> của Next.js) sẽ tải ảnh trực tiếp từ Cloudinary, giúp giảm tải hoàn toàn băng thông cho Backend.

### 4. Luồng Tương tác Dữ liệu (System Data Flow)
- Trải nghiệm Khám phá (Search): User nhập "áo len đỏ" -> Next.js gọi GET /search -> FastAPI mã hóa text thành vector và gọi Weaviate Hybrid Search -> Trả về danh sách Grid sản phẩm.

- Học hỏi Thời gian thực (Interact): User click vào 1 chiếc áo trong Grid -> Next.js gọi ngầm POST /interact -> FastAPI lấy Personal MLP của user từ Redis, chạy SGD để học sở thích mới, và lưu ngược model lại Redis (thời gian < 50ms).

- Gợi ý Tức thời (Recommend): Ngay sau khi click, Next.js gọi GET /recommend -> FastAPI lấy Top 100 từ Weaviate, Re-rank qua mô hình vừa được fine-tune, và trả về Top 10 -> Cập nhật hiển thị lên khu vực "Gợi ý cá nhân hóa" (thời gian ~200ms).

## Kế Hoạch Triển Khai Kỹ Thuật

### 1. Phase 1: Xử lý Dữ liệu & Thiết lập Lưu trữ (Offline Phase)

Mục tiêu: Cấu hình Vector Database để hỗ trợ đồng thời 2 luồng: Semantic Search (đầu vào từ Text/Filter) và Recommendation (đầu vào từ User Graph).

#### 1.1. Cấu hình Weaviate Schema: Thiết lập 2 collection chuyên biệt:

- Product (Dùng cho Hybrid Search): * Lưu vector 512-dim từ FashionCLIP.

- Định nghĩa cấu trúc Metadata (Properties) rõ ràng: product_name, product_type, color_group, department và cấu hình Tokenization dạng word hoặc field để hỗ trợ thuật toán BM25 (Keyword matching) và Exact Match Filtering.

- ProductRec (Dùng cho Recommendations): * Lưu vector 64-dim từ Student MLP kèm article_id.

#### 1.2. Batch Import: * Upload ảnh lên Cloudinary lấy secure_url.

- Trích xuất embedding 512-dim (FashionCLIP) và 64-dim (Student MLP).

- Đẩy toàn bộ dữ liệu kèm metadata vào 2 collection trên Weaviate Cloud.

### 2. Phase 2: Phát triển Backend API (FastAPI - Localhost:8000)

Mục tiêu: Mở rộng "Bộ não" để xử lý luồng Search đa phương thức (Multimodal) với tốc độ cao.

#### 2.1. Khởi tạo State & Models trên RAM (@app.on_event("startup")):

- Nạp dictionary vector 64-dim để phục vụ luồng Re-ranking.

- Nạp mô hình FashionCLIP Text Encoder: Đưa model encode text vào VRAM/RAM để biến câu truy vấn của người dùng (ví dụ: "áo len đỏ") thành vector 512-dim với độ trễ < 20ms.

#### 2.2. Xây dựng API GET /search (Tính năng cốt lõi mới):

- Nhận Payload: query (string) và filters (dict metadata).

- Logic xử lý: * Encode query thành vector 512-dim qua mạng FashionCLIP.

- Gọi Weaviate Collection Product bằng hàm query.hybrid() kết hợp:

- Vector similarity (tìm các sản phẩm có ảnh khớp ý nghĩa semantic của text).

- BM25 (tìm các sản phẩm có metadata chứa từ khóa query).

- Where filter (lọc chính xác theo thuộc tính filters).

- Sử dụng tham số alpha (từ 0 đến 1) để cân bằng trọng số giữa Vector và Keyword.

- Tối ưu Payload: Chỉ trả về thông tin hiển thị (ID, Tên, Giá, Ảnh), tuyệt đối lược bỏ trường vector.

#### 2.3. Cập nhật API POST /interact & GET /recommend:

- Giữ nguyên luồng tính toán: Ghi nhận click từ kết quả của API /search -> Lưu state Redis -> Fine-tune Personal MLP qua SGD -> Query ProductRec -> Re-rank bằng không gian cá nhân hóa trả về Top 10.

### 3. Phase 3: Phát triển Frontend (Next.js - Localhost:3000)

Mục tiêu: Khai thác UI để kích hoạt Hybrid Search và điều phối song song luồng Recommendation.

#### 3.1. Nâng cấp Component Tìm kiếm:

- Thêm thanh Search Bar nhập text.

- Thêm cụm Filter UI (Dropdown/Checkbox) tương ứng với các trường metadata trên Weaviate (ví dụ: Chọn nhóm màu, Loại sản phẩm).

#### 3.2. Cập nhật luồng State & Data Fetching:

- Thay vì load trang chủ ngẫu nhiên: Gọi API GET /search với query rỗng để Weaviate trả về list default ban đầu.

- Khi User Submit Search/Filter: Gọi API GET /search với tham số tương ứng -> Render lại Grid kết quả chính.

- Khi User Click vào Item Search: * Gọi ngầm POST /interact.

- Gọi GET /recommend và render danh sách "You might also like" bằng các component UI cập nhật tức thời theo trọng số Personal MLP.

#### 4. Phase 4: Tích hợp & Kiểm thử Hiệu năng (Integration)

- Tối ưu hóa Weaviate Hybrid Cache: Đảm bảo luồng search cơ bản (không cá nhân hóa) phải được Weaviate trả về dưới 100ms.

- Pre-warm AI Models: Gọi giả lập 1 request vào API /search ngay khi start server để load PyTorch graph và FashionCLIP, triệt tiêu độ trễ Cold-start cho lượt tìm kiếm đầu tiên của user lúc Demo.

## Hướng dẫn sử dụng
```bash
# Khởi động Redis
docker-compose up -d redis

# Khởi động FastAPI
uvicorn app.main:app --reload --port 8000

# Khởi động Next.js
# Mở 1 terminal mới
cd frontend
npm run dev
```