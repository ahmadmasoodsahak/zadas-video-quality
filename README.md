# Canlı Kamera + Django Tabanlı AI Upscale

Bu proje, tarayıcıdan kamerayı açar ve kareleri Python/Django arka ucuna göndererek anlık olarak çözünürlüğü yükseltilmiş ve filtrelerle iyileştirilmiş görüntüyü geri döndürür.

- Sol panel: Orijinal canlı kamera görüntüsü
- Sağ panel: Sunucuda (Python/NumPy/Pillow) işlenmiş yükseltilmiş görüntü

İleride ONNX Runtime ile gerçek süper-çözünürlük (SR) modelleri (FSRCNN/ESPCN/ESRGAN türevleri) kolayca entegre edilebilir.

## Kurulum
1. Sanal ortam (önerilir):
   - Windows PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
2. Bağımlılıklar: `pip install -r requirements.txt`
3. Django migrate: `python manage.py migrate`
4. Sunucu: `python manage.py runserver`
5. Tarayıcıda `http://127.0.0.1:8000/` adresine gidin.

## Kullanım
- "Kamerayı Başlat" ile kamerayı açın ve izin verin.
- Hedef ölçek (1x, 1.5x, 2x) ve model seçenekleri ile çıkışı değiştirin:
  - Lite (Hızlı, düşük kaynak)
  - Quality (Daha iyi kalite)
  - Ultra (Maks detay, daha yavaş)
  - Smooth (Yumuşatma + hafif keskinlik)
  - Denoise (Gürültü azaltma + keskinlik)
- "Görüntüyü İndir" ile işlenmiş görüntünün anlık kaydını alın.

## Teknik detaylar
- Frontend: Kamera akışı, requestAnimationFrame ile gerçek zamanlı olarak kareleri JPEG Base64 şeklinde `/api/process/` uç noktasına gönderir (aşırı yükü önlemek için tek eşzamanlı istek ilkesi).
- Backend: Django view, görseli alır; Pillow ile ölçekler; NumPy ile konvolüsyon çekirdekleri uygular ve JPEG Base64 döndürür.
- Modeller: Klasik filtre preset'lerine ek olarak FSRCNN x2 ONNX modeli entegre edilmiştir. CUDA destekli ONNX Runtime (onnxruntime-gpu) varsa GPU (CUDA) ile hızlandırılır; aksi halde CPU'ya düşer.

## Geliştirme ve CUDA Kullanımı
- CUDA ile hızlandırma: requirements.txt içinde onnxruntime-gpu kullanılır. NVIDIA sürücüleri ve uyumlu CUDA/cuDNN kurulu olmalıdır.
- Doğrulama: Sunucu çalışırken `http://127.0.0.1:8000/api/info/` adresine gidin. `active_providers` içinde `CUDAExecutionProvider` görmelisiniz. Aksi halde CPU çalışır.
- Model: İlk kullanıldığında FSRCNN x2 ONNX modeli otomatik indirilir (static/models). Otomatik indirme başarısız olursa README'deki URL'den model dosyasını indirip `static/models/fsrcnn_x2.onnx` olarak koyun.
- Performans: Daha yüksek FPS için tarayıcıdan gönderilen JPEG kalitesini düşürün (templates/index.html → toDataURL kalite), veya giriş çözünürlüğünü azaltın. GPU etkinse SR adımı CUDA’da çalışır.

## Docker ile Çalıştırma (CUDA destekli)
Önkoşullar:
- NVIDIA GPU + sürücü
- NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker Compose v2

Adımlar:
1. İmajı inşa edin: `docker compose build`
2. Servisi GPU ile başlatın: `docker compose run --rm --gpus all web python3 - <<<'import onnxruntime as ort; print(ort.get_available_providers())'` komutuyla GPU erişimini hızlıca test edebilirsiniz (çıktıda CUDAExecutionProvider beklenir).
3. Servisi başlatın: `docker compose up` (Compose sürümünüz `--gpus all` bayrağını destekliyorsa `docker compose up --profile gpu` yerine doğrudan `--gpus all` kullanın ya da `docker compose --compatibility up`).
4. Tarayıcıdan `http://127.0.0.1:8000/` adresine gidin.
5. CUDA doğrulaması: `http://127.0.0.1:8000/api/info/` sayfasında `available_providers` ve `active_providers` içinde `CUDAExecutionProvider` görmelisiniz.

Zorunlu CUDA:
- Container, varsayılan olarak `REQUIRE_CUDA=1` ile başlar. Eğer CUDA tespit edilmezse container başlatma aşamasında hata ile çıkar. CPU ile çalıştırmak isterseniz `docker compose run -e REQUIRE_CUDA=0` veya compose dosyasında bu değişkeni 0 yapın.

Notlar:
- Container başlarken otomatik `migrate` çalışır ve ONNXRuntime mevcut sağlayıcıları log'a yazar.
- Model dosyaları container içinde `/app/static/models` altında tutulur ve `docker-compose.yml` ile `model_cache` volume'üne bağlanır.
- Eğer indirme engellenirse modeli manuel indirip `static/models/fsrcnn_x2.onnx` konumuna yerleştirin (volume sayesinde kalıcı olacaktır).
