from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
import io
from PIL import Image
import numpy as np
import os
import threading

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime başarıyla yüklendi")
    print(f"📦 ONNX Runtime versiyonu: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"🔌 Mevcut execution providers: {providers}")
    
    # CUDA provider kontrolü
    if 'CUDAExecutionProvider' in providers:
        print("🚀 CUDA provider mevcut - GPU hızlandırması kullanılabilir!")
        # CUDA device bilgilerini al
        try:
            device_info = ort.get_device()
            print(f"💻 Aktif device: {device_info}")
        except:
            print("⚠️ Device bilgisi alınamadı")
    else:
        print("⚠️ CUDA provider bulunamadı - sadece CPU kullanılabilir")
        
    print("=" * 50)
except Exception as e:
    print(f"❌ ONNX Runtime yüklenemedi: {e}")
    ort = None

# Basit Python tarafı filtre/süper çözünürlük örnekleri
# Not: Gerçek SR için onnxruntime ile bir model yükleyebilirsiniz. Burada performans ve basitlik için klasik işlemler var.

def index(request):
    return render(request, 'index.html')


def _np_from_base64(b64: str) -> np.ndarray:
    try:
        header, data = b64.split(',') if ',' in b64 else ('', b64)
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return np.array(img)
    except Exception:
        return None


def _to_base64_image(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')
    bio = io.BytesIO()
    img.save(bio, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(bio.getvalue()).decode('utf-8')


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Optimize edilmiş resize fonksiyonu"""
    if abs(scale - 1.0) < 1e-3:
        return img
        
    pil = Image.fromarray(img)
    w, h = pil.size
    new_w, new_h = int(w*scale), int(h*scale)
    
    # Küçük görüntüler için LANCZOS, büyük görüntüler için BILINEAR (daha hızlı)
    if w * h > 500000:  # 500k pixel üzeri
        resample = Image.BILINEAR
    else:
        resample = Image.LANCZOS
        
    pil = pil.resize((new_w, new_h), resample)
    return np.array(pil)

# Optimize edilmiş konvolüsyon (scipy alternatifi)
def _apply_kernel(img: np.ndarray, kernel: np.ndarray, factor: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """OpenCV kullanarak optimize edilmiş konvolüsyon"""
    try:
        import cv2
        # OpenCV filter2D kullan (çok daha hızlı)
        kernel_cv = kernel.astype(np.float32)
        
        # Her kanal için ayrı ayrı işle
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            result[:,:,c] = cv2.filter2D(img[:,:,c].astype(np.float32), -1, kernel_cv)
        
        result = result * factor + bias
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
        
    except ImportError:
        print("⚠️ OpenCV bulunamadı, yavaş NumPy konvolüsyonu kullanılıyor...")
        # Fallback to slow numpy implementation
        kh, kw = kernel.shape
        pad_h, pad_w = kh//2, kw//2
        
        # simetrik padding
        padded = np.pad(img, ((pad_h, pad_h),(pad_w, pad_w),(0,0)), mode='reflect')
        h, w, _ = img.shape
        out = np.zeros_like(img, dtype=np.float32)
        
        # Sadece merkez bölgeyi işle (performans için)
        step = max(1, min(h, w) // 200)  # Büyük görüntülerde sampling
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                ys, ye = y, min(y + step, h)
                xs, xe = x, min(x + step, w)
                
                for py in range(ys, ye):
                    for px in range(xs, xe):
                        region = padded[py:py+kh, px:px+kw, :]
                        out[py, px, 0] = np.sum(region[:,:,0] * kernel)
                        out[py, px, 1] = np.sum(region[:,:,1] * kernel)
                        out[py, px, 2] = np.sum(region[:,:,2] * kernel)
        
        out = out * factor + bias
        out = np.clip(out, 0, 255)
        return out.astype(np.uint8)

class ONNXUpscaler:
    _lock = threading.Lock()
    _session = None
    _providers = []
    _model_scale = 2  # FSRCNN x2 varsayılan

    @classmethod
    def _get_model_path(cls):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'fsrcnn_x2.onnx')
        
        if not os.path.exists(model_path):
            print(f"📥 Model dosyası bulunamadı, indiriliyor: {model_path}")
            try:
                import urllib.request
                # Gerçek çalışan FSRCNN model URL'si
                urls_to_try = [
                    'https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx',
                    'https://huggingface.co/rocca/super-resolution-onnx/resolve/main/fsrcnn-small.onnx'
                ]
                
                for url in urls_to_try:
                    try:
                        print(f"🌐 Deneniyor: {url}")
                        urllib.request.urlretrieve(url, model_path)
                        print(f"✅ Model başarıyla indirildi!")
                        break
                    except Exception as e:
                        print(f"❌ Bu URL başarısız: {e}")
                        continue
                        
                if not os.path.exists(model_path):
                    print("⚠️ Model indirilemedi, basit bir placeholder oluşturuluyor...")
                    # Basit bir fallback - küçük dummy model
                    return None
                    
            except Exception as e:
                print(f"❌ Model indirme hatası: {e}")
                return None
                
        print(f"📁 Model dosyası hazır: {model_path}")
        return model_path

    @classmethod
    def session(cls):
        if ort is None:
            print("❌ ONNX Runtime bulunamadı, CUDA upscaling devre dışı")
            return None, []
        with cls._lock:
            if cls._session is not None:
                print(f"♻️ Mevcut ONNX session kullanılıyor (providers: {cls._providers})")
                return cls._session, cls._providers
            
            model_path = cls._get_model_path()
            if not model_path or not os.path.exists(model_path):
                print("❌ ONNX model dosyası bulunamadı")
                return None, []
            
            # Öncelikli olarak CUDA, sonra CPU
            providers = []
            available_providers = ort.get_available_providers() if hasattr(ort, 'get_available_providers') else []
            print(f"🔍 Mevcut tüm providers: {available_providers}")
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                print("✅ CUDA provider eklendi")
            else:
                print("⚠️ CUDA provider bulunamadı")
                
            providers.append('CPUExecutionProvider')
            print(f"📋 Kullanılacak provider sırası: {providers}")
            
            sess_options = ort.SessionOptions()
            try:
                cls._session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
                cls._providers = cls._session.get_providers()
                print(f"🎯 ONNX session başarıyla oluşturuldu!")
                print(f"🔧 Aktif providers: {cls._providers}")
                return cls._session, cls._providers
            except Exception as e:
                print(f"❌ ONNX session oluşturulamadı: {e}")
                return None, []

    @classmethod
    def upscale_x2(cls, img_rgb: np.ndarray) -> tuple[np.ndarray, list]:
        """Optimize edilmiş ONNX x2 upscaling"""
        print(f"🖼️ Upscale işlemi başlatılıyor - görüntü boyutu: {img_rgb.shape}")
        
        # Çok büyük görüntüleri önce küçült (memory ve performans için)
        original_shape = img_rgb.shape
        if original_shape[0] > 1080 or original_shape[1] > 1920:
            print("📏 Görüntü çok büyük, önce küçültülüyor...")
            scale_down = min(1080/original_shape[0], 1920/original_shape[1])
            img_rgb = _resize(img_rgb, scale_down)
            print(f"📐 Yeni boyut: {img_rgb.shape}")
        
        session, providers = cls.session()
        if session is None:
            print("❌ ONNX session bulunamadı, CPU upscaling'e geçiliyor")
            return None, providers
            
        print(f"🚀 ONNX ile upscaling başlıyor (providers: {providers})")
        
        try:
            # Görüntüyü model için hazırla
            img_float = img_rgb.astype(np.float32) / 255.0
            
            # Model input shape kontrolü
            input_shape = session.get_inputs()[0].shape
            print(f"🔍 Model beklenen input shape: {input_shape}")
            
            # Eğer model 1-channel (Y) bekliyorsa RGB'yi Y'ye çevir
            if len(input_shape) == 4 and input_shape[1] == 1:
                # RGB to Y (luminance)
                img_y = np.dot(img_float, [0.299, 0.587, 0.114])
                chw = img_y[None, None, :, :]  # NCHW format
                print("🔄 RGB -> Y dönüşümü yapıldı")
            else:
                # RGB olarak bırak
                chw = np.transpose(img_float, (2,0,1))[None, ...]  # NCHW format
                print("🔄 RGB formatı korundu")
            
            print(f"📐 Model girdi şekli: {chw.shape}")
            
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            print(f"🔌 Model girdi: {input_name}, çıktı: {output_name}")
            
            # ONNX inference
            out = session.run([output_name], {input_name: chw})[0]
            print(f"✅ ONNX inference tamamlandı - çıktı şekli: {out.shape}")
            
            # Output'u işle
            if out.shape[1] == 1:  # Y channel output
                # Y kanalını RGB'ye geri çevir
                out_y = np.clip(out[0, 0] * 255.0, 0, 255).astype(np.uint8)
                # Basit grayscale to RGB
                out_rgb = np.stack([out_y, out_y, out_y], axis=2)
                print("🔄 Y -> RGB dönüşümü yapıldı")
            else:
                # RGB output
                out_rgb = np.clip(np.transpose(out[0], (1,2,0)) * 255.0, 0, 255).astype(np.uint8)
                print("🔄 RGB output işlendi")
            
            print(f"🎯 Final görüntü boyutu: {out_rgb.shape}")
            return out_rgb, providers
            
        except Exception as e:
            print(f"❌ ONNX upscaling hatası: {e}")
            import traceback
            traceback.print_exc()
            return None, providers


kernels = {
    'sharpenLite': (np.array([[0, -0.2, 0], [-0.2, 2, -0.2], [0, -0.2, 0]]), 1.0, 0.0),
    'sharpenStrong': (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), 1.0, 0.0),
    'edgeBoost': (np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]), 0.2, 0.0),
    'gaussianBlur3': (np.array([[1,2,1],[2,4,2],[1,2,1]])/16.0, 1.0, 0.0),
    'boxBlur3': (np.ones((3,3))/9.0, 1.0, 0.0),
}


def _process_numpy(img: np.ndarray, scale: float, mode: str, grayscale: bool = False, contrast: float = 1.0) -> np.ndarray:
    print(f"🎨 İşlem başlıyor - mod: {mode}, ölçek: {scale}, gri: {grayscale}, kontrast: {contrast}")
    
    # Performans kontrolü - çok büyük görüntüleri limitele
    h, w = img.shape[:2]
    is_large = (h * w) > 1000000  # 1M pixel üzeri
    if is_large:
        print(f"⚡ Büyük görüntü tespit edildi ({h}x{w}), hızlı mod aktif")
    
    # Eğer ONNX SR seçildiyse önce x2 SR uygula
    if mode == 'fsrcnn':
        print("🔥 FSRCNN modu seçildi, ONNX upscaling deneniyor...")
        sr, providers = ONNXUpscaler.upscale_x2(img)
        if sr is not None:
            print("✅ ONNX upscaling başarılı!")
            out = sr
            # istenen ölçek 2 değilse ek yeniden örnekleme uygula
            if abs(scale - 2.0) > 1e-3:
                print(f"📏 Ek ölçekleme uygulanıyor: {scale / 2.0}")
                out = _resize(out, scale / 2.0)
        else:
            # GPU/Model yoksa klasik ölçeklemeye düş
            print("⚠️ ONNX başarısız, klasik ölçeklemeye geçiliyor")
            out = _resize(img, scale)
    else:
        # ölçekle
        print(f"📐 Klasik ölçekleme uygulanıyor: {scale}")
        out = _resize(img, scale)
    if grayscale:
        g = np.dot(out[...,:3], [0.299, 0.587, 0.114])
        out = np.stack([g, g, g], axis=2).astype(np.uint8)
    # Kontrast ayarı (basit): (x - 128)*c + 128
    if contrast and abs(contrast - 1.0) > 1e-3:
        out = np.clip((out.astype(np.float32) - 128.0) * float(contrast) + 128.0, 0, 255).astype(np.uint8)

    # Filtre uygulama - büyük görüntülerde basitleştir
    if mode == 'lite':
        if not is_large:
            k, f, b = kernels['sharpenLite']
            out = _apply_kernel(out, k, f, b)
        else:
            print("⚡ Büyük görüntü - lite filtre atlandı")
    elif mode == 'quality':
        k, f, b = kernels['edgeBoost']
        out = _apply_kernel(out, k, f, b)
        if not is_large:  # Sadece küçük görüntülerde ikinci filtre
            k, f, b = kernels['sharpenStrong']
            out = _apply_kernel(out, k, f, b)
    elif mode == 'ultra':
        if not is_large:
            k, f, b = kernels['gaussianBlur3']
            out = _apply_kernel(out, k, f, b)
            k, f, b = kernels['edgeBoost']
            out = _apply_kernel(out, k, f, b)
            k, f, b = kernels['sharpenStrong']
            out = _apply_kernel(out, k, f, b)
        else:
            print("⚡ Büyük görüntü - ultra filtreler basitleştirildi")
            k, f, b = kernels['edgeBoost']
            out = _apply_kernel(out, k, f, b)
    elif mode == 'smooth':
        k, f, b = kernels['gaussianBlur3']
        out = _apply_kernel(out, k, f, b)
        if not is_large:
            k, f, b = kernels['sharpenLite']
            out = _apply_kernel(out, k, f, b)
    elif mode == 'denoise':
        k, f, b = kernels['boxBlur3']
        out = _apply_kernel(out, k, f, b)
        if not is_large:
            k, f, b = kernels['sharpenLite']
            out = _apply_kernel(out, k, f, b)
    else:
        k, f, b = kernels['edgeBoost']
        out = _apply_kernel(out, k, f, b)
        if not is_large:
            k, f, b = kernels['sharpenStrong']
            out = _apply_kernel(out, k, f, b)
    return out


@csrf_exempt
def process_frame(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('POST bekleniyor')
    data = request.POST
    b64 = data.get('image')
    scale = float(data.get('scale', '2'))
    mode = data.get('model', 'quality')
    grayscale = data.get('grayscale', '0') == '1'
    contrast = float(data.get('contrast', '1.0'))
    npimg = _np_from_base64(b64)
    if npimg is None:
        return HttpResponseBadRequest('Geçersiz görüntü')
    out = _process_numpy(npimg, scale, mode, grayscale=grayscale, contrast=contrast)
    b64_out = _to_base64_image(out)
    return JsonResponse({'image': b64_out})

@csrf_exempt
def info(request):
    providers = []
    active = []
    cuda_available = False
    gpu_info = {}
    
    print("🔍 Sistem bilgileri toplanıyor...")
    
    if ort is not None:
        try:
            providers = getattr(ort, 'get_available_providers', lambda: [])()
            print(f"📋 Tüm providers: {providers}")
            
            # CUDA provider kontrolü
            cuda_available = 'CUDAExecutionProvider' in providers
            print(f"🚀 CUDA mevcut: {cuda_available}")
            
        except Exception as e:
            print(f"❌ Provider bilgisi alınamadı: {e}")
            providers = []
            
        # Session bilgilerini al - basit şekilde
        try:
            sess, active = ONNXUpscaler.session()
            active = active or []
            print(f"🎯 Aktif providers: {active}")
        except Exception as e:
            print(f"⚠️ Session bilgisi alınamadı: {e}")
            active = []
    else:
        print("❌ ONNX Runtime yüklü değil")
    
    # GPU bilgilerini basit şekilde topla
    gpu_info = {'status': 'basic_info_only'}
    
    result = {
        'onnxruntime': ort is not None,
        'onnxruntime_version': getattr(ort, '__version__', 'unknown') if ort else None,
        'available_providers': providers,
        'active_providers': active,
        'cuda_available': cuda_available,
        'gpu_info': gpu_info
    }
    
    print(f"📊 Döndürülen bilgiler: {result}")
    return JsonResponse(result)
