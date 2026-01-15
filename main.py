# --- GEREKLİ TÜM KÜTÜPHANELER ---
import feedparser
import requests
import time
import re
import json
import os
import logging
import threading
import hashlib
import concurrent.futures
from datetime import datetime
from flask import Flask, request, render_template_string, jsonify
from transformers import pipeline
from googlesearch import search
import pytesseract
from PIL import Image

# --- UYGULAMA KURULUMU ---
app = Flask(__name__)

# --- TESSERACT-OCR MOTORUNUN YOLU ---
# Lütfen Tesseract-OCR'ı kurduğunuz yolun bu olduğundan emin olun.
TESSERACT_PATH = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print(f"UYARI: Tesseract-OCR yürütülebilir dosyası şu yolda bulunamadı: {TESSERACT_PATH}")
    print("Lütfen TESSERACT_PATH değişkenini doğru kurulum yoluyla güncelleyin.")

# --- BASİT YAPILANDIRMA ---
class Config:
    CACHE_DURATION = 1800  # 30 dakika
    MIN_KEYWORDS = 3
    JACCARD_THRESHOLD = 0.08
    MAX_RSS_BONUS = 25
    SENTIMENT_CONFIDENCE_THRESHOLD = 0.8
    SERVER_PORT = 5000
    DEBUG = False # Prodüksiyon için False olmalı
    MAX_WORKERS = 5 # Model yükleme ve RSS için
    REQUEST_TIMEOUT = 10

CONFIG = Config()

# --- PARAMETRELER VE LİSTELER ---
# (Her iki dosyadan daha kapsamlı listeler birleştirildi)
TÜRKÇE_GÜVENİLİR_RSS_FEEDS = {
    "Anadolu Ajansı (Gündem)": "https://www.aa.com.tr/tr/rss/default?cat=gundem",
    "Anadolu Ajansı (Ekonomi)": "https://www.aa.com.tr/tr/rss/default?cat=ekonomi",
    "TRT Haber (Gündem)": "https://www.trthaber.com/xml_kategori.php?kategori=gundem",
    "NTV (Türkiye)": "https://www.ntv.com.tr/turkiye.rss",
    "Habertürk (Gündem)": "https://www.haberturk.com/rss/gundem.xml",
    "Hürriyet (Gündem)": "https://www.hurriyet.com.tr/rss/gundem",
    "Milliyet (Gündem)": "https://www.milliyet.com.tr/rss/rssnew/gundemrss.xml",
    "Sözcü Gündem": "https://www.sozcu.com.tr/feed/?cat=gundem",
}

SANSASYONEL_KELİMELER = [
    'şok!', 'skandal!', 'inanılmaz!', 'flaş!', 'son dakika!', 
    'gizli gerçek', 'büyük sır', 'herkes şokta', 'olay oldu', 'korkunç',
    'çılgına döndü', 'ifşa', 'bomba', 'şoke etti', 'şaşırtan'
]

STOP_WORDS_TR = {
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 
    'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 
    'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'ile', 'ise', 'için', 
    'ki', 'kim', 'mı', 'mi', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 
    'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 
    'tüm', 've', 'veya', 'ya', 'yani', 'zaten', 'bir', 'iki', 'üç', 'dört', 'beş'
}

# --- ÖNBELLEK YÖNETİMİ (main.py'den) ---
class CacheManager:
    def __init__(self):
        self.rss_cache = []
        self.last_rss_fetch_time = 0
        self.analysis_cache = {} # Metin analiz sonuçlarını hash ile saklar

cache_manager = CacheManager()

# --- NLP MODELİ YÖNETİMİ (Geliştirilmiş) ---
class ModelManager:
    def __init__(self):
        self.sentiment_pipeline = None
        self.ai_image_detector = None
        self.sentiment_model_loaded = False
        self.ai_model_loaded = False
        
    def load_models(self):
        """Tüm modelleri ayrı thread'lerde asenkron olarak yükle"""
        print("Model yükleme işlemleri başlatılıyor...")
        
        # Görev 1: Duygu Analizi
        thread1 = threading.Thread(target=self._load_sentiment_model)
        thread1.daemon = True
        thread1.start()
        
        # Görev 2: YZ Görüntü Tespiti
        thread2 = threading.Thread(target=self._load_ai_image_model)
        thread2.daemon = True
        thread2.start()

    def _load_sentiment_model(self):
        try:
            print("Duygu analizi modeli yükleniyor (savasy/bert-base-turkish-sentiment-cased)...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="savasy/bert-base-turkish-sentiment-cased"
            )
            self.sentiment_model_loaded = True
            print("✓ Duygu analizi modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"✗ Duygu analizi modeli yüklenemedi: {e}")
            self.sentiment_pipeline = None

    def _load_ai_image_model(self):
        try:
            print("YZ resim tespit modeli yükleniyor (dima806/ai_vs_real_image_detection)...")
            self.ai_image_detector = pipeline(
                "image-classification", 
                model="dima806/ai_vs_real_image_detection"
            )
            self.ai_model_loaded = True
            print("✓ YZ resim tespit modeli başarıyla yüklendi.")
        except Exception as e:
            print(f"✗ YZ resim tespit modeli yüklenemedi: {e}")
            self.ai_image_detector = None

model_manager = ModelManager()

# --- YARDIMCI FONKSİYONLAR (main.py'den) ---
def text_preprocessing(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text.strip())

def get_keywords(text):
    text_lower = text.lower()
    text_cleaned = re.sub(r'[^\w\s]', '', text_lower)
    words = text_cleaned.split()
    return {word for word in words if word not in STOP_WORDS_TR and len(word) > 3}

def calculate_jaccard_similarity(set1, set2):
    if not set1 or not set2: return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def generate_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# --- RSS YÖNETİMİ (main.py'nin eşzamanlı (concurrent) yöntemi) ---
def fetch_single_rss_feed(source_name, url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, timeout=CONFIG.REQUEST_TIMEOUT, headers=headers)
        feed = feedparser.parse(response.content)
        items = []
        for entry in feed.entries[:10]: # Her kaynaktan son 10 haber
            content = f"{entry.get('title', '')} {entry.get('summary', '')}"
            items.append({
                "keywords": get_keywords(content),
                "source": source_name,
                "title": entry.get('title', 'Başlık Yok'),
                "link": entry.get('link', '#'),
            })
        return source_name, items
    except Exception as e:
        return source_name, []

def fetch_all_rss_feeds():
    global cache_manager
    current_time = time.time()
    if (current_time - cache_manager.last_rss_fetch_time < CONFIG.CACHE_DURATION and 
        cache_manager.rss_cache):
        print("RSS verisi önbellekten kullanıldı.")
        return cache_manager.rss_cache
        
    print(f"RSS verisi önbelleği yenileniyor... ({len(TÜRKÇE_GÜVENİLİR_RSS_FEEDS)} kaynak)")
    all_rss_content = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS) as executor:
        future_to_source = {
            executor.submit(fetch_single_rss_feed, source_name, url): source_name 
            for source_name, url in TÜRKÇE_GÜVENİLİR_RSS_FEEDS.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_source):
            source_name, items = future.result()
            if items:
                all_rss_content.extend(items)
                print(f"✓ {source_name}: {len(items)} haber")
            else:
                print(f"✗ {source_name}: Haber alınamadı.")
    
    cache_manager.rss_cache = all_rss_content
    cache_manager.last_rss_fetch_time = current_time
    print(f"✓ Önbellek {len(all_rss_content)} haber ile yenilendi.")
    return all_rss_content

# --- METİN ANALİZ FONKSİYONLARI (main.py'nin gelişmiş motoru) ---
def analyze_text_length(text, score, log):
    word_count = len(text.split())
    if word_count < 20:
        score -= 15
        log.append({"type": "log-minus", "message": f"<b>-15 Puan:</b> Metin çok kısa ({word_count} kelime). Detay eksikliği güvenilirliği azaltır."})
    elif word_count > 100:
        score += 10
        log.append({"type": "log-plus", "message": f"<b>+10 Puan:</b> Metin yeterince detaylı ({word_count} kelime)."})
    else:
        log.append({"type": "log-info", "message": f"<b>+0 Puan:</b> Metin uzunluğu uygun ({word_count} kelime)."})
    return score, log

def analyze_sensationalism(text, score, log):
    text_lower = text.lower()
    found_words = [word for word in SANSASYONEL_KELİMELER if word in text_lower]
    if found_words:
        penalty = min(len(found_words) * 8, 30)
        score -= penalty
        log.append({"type": "log-minus", "message": f"<b>-{penalty} Puan:</b> Metinde {len(found_words)} sansasyonel ifade bulundu: {', '.join(found_words[:3])}"})
    else:
        score += 5
        log.append({"type": "log-plus", "message": f"<b>+5 Puan:</b> Metin tarafsız ve nesnel bir dil kullanıyor."})
    return score, log

def analyze_sentiment(text, score, log):
    if not model_manager.sentiment_model_loaded or model_manager.sentiment_pipeline is None:
        log.append({"type": "log-info", "message": "Duygu analizi modeli henüz yükleniyor, bu adım atlandı."})
        return score, log
    try:
        truncated_text = text[:512]
        result = model_manager.sentiment_pipeline(truncated_text)[0]
        label, confidence = result['label'], result['score']
        
        taziye_kelimeler = ['taziye', 'başsağlığı', 'vefat', 'ölüm', 'merhume', 'merhum']
        if any(kelime in text.lower() for kelime in taziye_kelimeler):
            log.append({"type": "log-info", "message": f"<b>+0 Puan:</b> Taziye mesajı olduğu için duygu analizi dikkate alınmadı."})
        elif label == 'negative' and confidence > CONFIG.SENTIMENT_CONFIDENCE_THRESHOLD:
            score -= 12
            log.append({"type": "log-minus", "message": f"<b>-12 Puan:</b> Metin güçlü negatif duygu içeriyor (%{confidence*100:.0f} güven)."})
        elif label == 'positive' and confidence > 0.85:
            score -= 8
            log.append({"type": "log-minus", "message": f"<b>-8 Puan:</b> Metin aşırı pozitif/övgü dolu (%{confidence*100:.0f} güven)."})
        else:
            score += 8
            log.append({"type": "log-plus", "message": f"<b>+8 Puan:</b> Metnin duygu dengesi uygun görünüyor."})
    except Exception as e:
        log.append({"type": "log-info", "message": f"Duygu analizi hatası: {str(e)[:100]}..."})
    return score, log

def analyze_rss_feeds(text, score, log):
    try:
        all_rss_items = fetch_all_rss_feeds()
        if not all_rss_items:
            log.append({"type": "log-info", "message": "Güvenilir RSS kaynaklarına ulaşılamadı."})
            return score, log
            
        keywords_user = get_keywords(text)
        if len(keywords_user) < CONFIG.MIN_KEYWORDS:
            log.append({"type": "log-info", "message": f"RSS karşılaştırması için yeterli anahtar kelime yok (en az {CONFIG.MIN_KEYWORDS} gerekli)."})
            return score, log

        best_match, highest_similarity, match_count = None, 0.0, 0
        for item in all_rss_items:
            similarity = calculate_jaccard_similarity(keywords_user, item["keywords"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = item
            if similarity > CONFIG.JACCARD_THRESHOLD:
                match_count += 1
        
        if highest_similarity > CONFIG.JACCARD_THRESHOLD:
            score_bonus = min(int(highest_similarity * 100), CONFIG.MAX_RSS_BONUS)
            score += score_bonus
            log.append({"type": "log-plus", "message": f"<b>+{score_bonus} Puan:</b> Konu {match_count} güvenilir kaynakta doğrulandı (en yüksek benzerlik: %{highest_similarity*100:.1f}). Kaynak: <a href='{best_match['link']}' target='_blank'>{best_match['source']}</a>"})
        else:
            log.append({"type": "log-info", "message": f"<b>+0 Puan:</b> Bu konu güncel RSS kaynaklarında bulunamadı (en yüksek benzerlik: %{highest_similarity*100:.1f})."})
    except Exception as e:
        log.append({"type": "log-info", "message": f"RSS analizi hatası: {str(e)[:100]}..."})
    return score, log

def analyze_fact_check(text, score, log):
    try:
        if len(text.split()) < 10:
            log.append({"type": "log-info", "message": "Metin çok kısa, fact-check kontrolü atlandı."})
            return score, log
        query = f'site:teyit.org "{text[:80]}"'
        search_results = list(search(query, num_results=1))
        if search_results:
            score -= 25
            log.append({"type": "log-minus", "message": f"<b>-25 Puan:</b> Bu iddia fact-check sitelerinde incelenmiş. <a href='{search_results[0]}' target='_blank'>İncelemeyi görüntüle</a>"})
        else:
            score += 5
            log.append({"type": "log-plus", "message": "<b>+5 Puan:</b> Fact-check sitelerinde bu iddiayla ilgili kayıt bulunamadı."})
    except Exception as e:
        log.append({"type": "log-info", "message": "<b>+0 Puan:</b> Fact-check kontrolü geçici olarak devre dışı (Google arama limiti aşılmış olabilir)."})
    return score, log

# --- ANA METİN ANALİZ MOTORU (main.py'den) ---
def analyze_text(text):
    start_time = time.time()
    cleaned_text = text_preprocessing(text)
    if not cleaned_text:
        return {'error': 'Geçersiz veya boş metin girdiniz.', 'analysis_type': 'text'}
    
    text_hash = generate_text_hash(cleaned_text)
    if text_hash in cache_manager.analysis_cache:
        cached_result = cache_manager.analysis_cache[text_hash]
        cached_result['cached'] = True
        return cached_result
    
    score, analysis_log = 50, []
    analysis_steps = [
        ("Metin Uzunluğu Analizi", analyze_text_length),
        ("Sansasyonel Dil Kontrolü", analyze_sensationalism),
        ("Duygu Analizi", analyze_sentiment),
        ("RSS Doğrulama", analyze_rss_feeds),
        ("Fact-Check Kontrolü", analyze_fact_check)
    ]
    
    for step_name, analysis_func in analysis_steps:
        try:
            score, analysis_log = analysis_func(cleaned_text, score, analysis_log)
        except Exception as e:
            analysis_log.append({"type": "log-info", "message": f"'{step_name}' adımında hata: {str(e)[:100]}..."})
    
    final_score = max(0, min(100, int(score)))
    processing_time = time.time() - start_time
    analysis_log.append({"type": "log-info", "message": f"Analiz {processing_time:.2f} saniyede tamamlandı."})
    
    result = {
        'score': final_score,
        'log': analysis_log,
        'original_text': text,
        'processing_time': processing_time,
        'word_count': len(text.split()),
        'char_count': len(text),
        'analysis_type': 'text' # HTML'in doğru sekmeyi açması için
    }
    
    cache_manager.analysis_cache[text_hash] = result
    if len(cache_manager.analysis_cache) > 100:
        cache_manager.analysis_cache.clear() # Önbelleği çok büyütme
    
    return result

# --- FOTOĞRAF ANALİZ FONKSİYONLARI (gerçeklik radarı.py'den) ---

def analyze_photo_ocr(files):
    """
    Yüklenen fotoğraftan OCR ile metin okur ve Gelişmiş analyze_text'i çağırır.
    """
    print("Fotoğraf METİN analizi (OCR) başlatıldı...")
    if 'photo_file_ocr' not in files or not files['photo_file_ocr'].filename:
        raise ValueError("Lütfen metin analizi için bir fotoğraf dosyası seçin.")
        
    file = files['photo_file_ocr']
    try:
        img = Image.open(file.stream)
    except Exception as e:
        raise ValueError(f"Geçersiz resim dosyası: {e}")

    try:
        text_from_image = pytesseract.image_to_string(img, lang='tur+eng')
    except Exception as e:
        raise RuntimeError(f"Tesseract-OCR hatası (Kurulumu ve TESSERACT_PATH'i kontrol edin): {e}")

    if not text_from_image or not text_from_image.strip():
        return {
            'score': 50,
            'log': [{"type": "log-info", "message": "Fotoğraf başarıyla yüklendi ancak içinde okunabilir bir metin tespit edilemedi."}],
            'original_text': '(Yüklenen Fotoğraf)',
            'analysis_type': 'photo_ocr'
        }
    
    # METNİ ANALİZ ETMEK İÇİN 'main.py'NİN GELİŞMİŞ MOTORUNU KULLAN
    results = analyze_text(text_from_image)
    
    results['log'].insert(0, {
        "type": "log-info",
        "message": f"<b>Fotoğraftan Okunan Metin:</b> \"{text_from_image[:150]}...\""
    })
    results['analysis_type'] = 'photo_ocr' # Tipi override et
    return results

def analyze_ai_generation(files):
    """
    Yüklenen fotoğrafın YZ ile üretilip üretilmediğini tespit eder.
    (ModelManager entegrasyonu yapıldı)
    """
    print("Fotoğraf AI analizi başlatıldı...")
    if 'photo_file_ai' not in files or not files['photo_file_ai'].filename:
        raise ValueError("Lütfen Yapay Zeka tespiti için bir fotoğraf dosyası seçin.")
        
    if not model_manager.ai_model_loaded or model_manager.ai_image_detector is None:
        raise RuntimeError("Yapay zeka resim tespit modeli henüz yükleniyor veya yüklenemedi. Lütfen biraz bekleyip tekrar deneyin.")
        
    file = files['photo_file_ai']
    try:
        img = Image.open(file.stream)
    except Exception as e:
        raise ValueError(f"Geçersiz resim dosyası: {e}") 

    try:
        # Modeli ModelManager üzerinden çağır
        predictions = model_manager.ai_image_detector(img) 

        ai_score, human_score = 0.0, 0.0
        for p in predictions:
            if p['label'] == 'FAKE': ai_score = p['score']
            elif p['label'] == 'REAL': human_score = p['score']
        print(f"AI Tespit Ham Skorlar -> FAKE: {ai_score:.4f}, REAL: {human_score:.4f}")

    except Exception as e:
        raise RuntimeError(f"AI resim tespiti sırasında hata oluştu: {e}")

    HIGH_THRESHOLD, MEDIUM_THRESHOLD = 0.85, 0.50
    analysis_log, final_label, css_class = [], "", "log-info"
    
    if ai_score > HIGH_THRESHOLD:
        final_label = f"Yüksek Olasılıkla Yapay Zeka (%{ai_score*100:.0f})"
        css_class = "log-minus"
        analysis_log.append({"type": css_class, "message": f"<b>Tespit (>{HIGH_THRESHOLD*100:.0f}%):</b> Fotoğrafın <b>%{ai_score*100:.0f}</b> olasılıkla bir yapay zeka tarafından üretildiği ('FAKE') tespit edildi."})
    elif ai_score > MEDIUM_THRESHOLD:
        final_label = f"Orta Olasılıkla Yapay Zeka (%{ai_score*100:.0f})"
        css_class = "log-info"
        analysis_log.append({"type": css_class, "message": f"<b>Tespit (%{MEDIUM_THRESHOLD*100:.0f} - %{HIGH_THRESHOLD*100:.0f}):</b> Fotoğrafın <b>%{ai_score*100:.0f}</b> olasılıkla yapay zeka tarafından üretilmiş olabileceği düşünülüyor, ancak kesinlik sınırı aşılamadı."})
    elif human_score > HIGH_THRESHOLD: 
        final_label = f"Yüksek Olasılıkla Yapay Zeka Değil (%{human_score*100:.0f})"
        css_class = "log-plus"
        analysis_log.append({"type": css_class, "message": f"<b>Tespit (>{HIGH_THRESHOLD*100:.0f}%):</b> Fotoğrafın <b>%{human_score*100:.0f}</b> olasılıkla yapay zeka tarafından üretilmediği ('REAL') tespit edildi."})
    elif human_score > MEDIUM_THRESHOLD:
        final_label = f"Orta Olasılıkla Yapay Zeka Değil (%{human_score*100:.0f})"
        css_class = "log-info"
        analysis_log.append({"type": css_class, "message": f"<b>Tespit (%{MEDIUM_THRESHOLD*100:.0f} - %{HIGH_THRESHOLD*100:.0f}):</b> Fotoğrafın <b>%{human_score*100:.0f}</b> olasılıkla yapay zeka tarafından üretilmediği ('REAL') düşünülüyor, ancak kesinlik sınırı aşılamadı."})
    else:
        final_label = "Tespit Edilemedi / Belirsiz"
        css_class = "log-info"
        analysis_log.append({"type": css_class, "message": "<b>Tespit:</b> Model, fotoğrafın yapay zeka mı yoksa insan yapımı mı olduğu konusunda net bir karara varamadı."})

    analysis_log.append({"type": "log-info", "message": f"Model Ham Skorları | Yapay Zeka ('FAKE'): %{ai_score*100:.1f} | İnsan ('REAL'): %{human_score*100:.1f}"})

    return {
        'analysis_type': 'photo_ai', 
        'ai_label': final_label,
        'log': analysis_log,
        'css_class': css_class
    }

# --- BİRLEŞTİRİLMİŞ WEB ARAYÜZÜ (HTML/CSS) ---
# (main.py'nin modern tasarımı + gerçeklik radarı.py'nin sekme yapısı)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gelişmiş Dezenformasyon Analiz Aracı</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 900px;
            margin: 20px auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.2em; }
        .content { padding: 30px; }
        
        /* Ana Sekme Stilleri */
        .tab-container { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }
        .tab-button {
            padding: 15px 25px; cursor: pointer; background-color: #f0f0f0;
            border: none; font-size: 16px; font-weight: 500;
            border-radius: 8px 8px 0 0; margin-bottom: -2px; transition: all 0.3s ease;
        }
        .tab-button:hover { background-color: #e0e0e0; }
        .tab-button.active {
            background-color: #fff; border: 2px solid #ddd;
            border-bottom: 2px solid #fff; font-weight: bold; color: #3498db;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.5s; }
        
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        /* Alt Sekme Stilleri */
        .sub-tab-container { display: flex; border-bottom: 1px solid #ccc; margin-bottom: 20px; margin-top: 10px; }
        .sub-tab-button {
            padding: 10px 15px; cursor: pointer; background: none; border: none;
            font-size: 15px; border-bottom: 3px solid transparent; transition: all 0.3s ease;
        }
        .sub-tab-button.active { font-weight: bold; color: #3498db; border-bottom: 3px solid #3498db; }
        .sub-tab-content { display: none; }
        .sub-tab-content.active { display: block; }

        /* Modern Form Stilleri */
        form { display: flex; flex-direction: column; gap: 15px; }
        textarea {
            width: 100%; height: 180px; padding: 15px; font-size: 16px;
            border: 2px solid #e0e0e0; border-radius: 10px; resize: vertical;
            transition: border-color 0.3s; font-family: inherit;
        }
        textarea:focus { outline: none; border-color: #3498db; }
        
        input[type="url"], input[type="file"] {
            width: 100%; padding: 12px; font-size: 16px; border: 2px solid #e0e0e0;
            border-radius: 10px; font-family: inherit; transition: border-color 0.3s;
        }
        input[type="file"] { padding: 10px; }
        input[type="url"]:focus, input[type="file"]:focus { outline: none; border-color: #3498db; }

        .btn {
            background: #3498db; color: white; border: none; padding: 15px 30px;
            font-size: 18px; font-weight: bold; border-radius: 10px;
            cursor: pointer; transition: all 0.3s; width: 100%;
        }
        .btn:hover { background: #2980b9; transform: translateY(-2px); }
        .btn:disabled { background: #95a5a6; cursor: not-allowed; }

        /* Loading Spinner */
        .loading { display: none; text-align: center; padding: 40px 0; }
        .loading-spinner {
            border: 5px solid #f3f4f6; border-top: 5px solid #3498db;
            border-radius: 50%; width: 50px; height: 50px;
            animation: spin 1s linear infinite; margin: 0 auto 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Sonuç Stilleri (main.py'den) */
        .results {
            margin-top: 30px; background: #ecf0f1; border-radius: 10px;
            padding: 25px; animation: fadeIn 0.5s;
        }
        .score-container { text-align: center; margin-bottom: 30px; }
        .score { font-size: 4.5em; font-weight: bold; line-height: 1; margin: 20px 0; }
        .score-label { font-size: 1.4em; font-weight: bold; margin-top: 10px; }
        
        .score-red { color: #e74c3c; } .score-orange { color: #f39c12; }
        .score-yellow { color: #f1c40f; } .score-green { color: #2ecc71; }
        
        .analysis-log { list-style: none; }
        .analysis-log li {
            padding: 15px; margin-bottom: 10px; border-radius: 8px;
            border-left: 5px solid;
        }
        .log-plus { border-left-color: #2ecc71; background: rgba(46, 204, 113, 0.1); }
        .log-minus { border-left-color: #e74c3c; background: rgba(231, 76, 60, 0.1); }
        .log-info { border-left-color: #f39c12; background: rgba(243, 156, 18, 0.1); }
        
        .error {
            background: #e74c3c; color: white; padding: 15px;
            border-radius: 8px; text-align: center; font-weight: bold;
        }
        
        /* AI Tespit Stili */
        .ai-label { font-size: 2.5em; font-weight: bold; text-align: center; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gelişmiş Dezenformasyon Analiz Aracı</h1>
        </div>
        
        <div class="content">
            <div class="tab-container">
                <button class="tab-button" onclick="openTab(event, 'Metin')">Metin Analizi</button>
                <button class="tab-button" onclick="openTab(event, 'Fotograf')">Fotoğraf Analizi</button>
                <button class="tab-button" onclick="openTab(event, 'Link')">Link Analizi</button>
            </div>

            <div id="Metin" class="tab-content">
                <form action="/" method="POST" id="textForm">
                    <input type="hidden" name="analysis_type" value="text">
                    <textarea name="text_to_analyze" placeholder="Analiz etmek istediğiniz metni buraya yapıştırın...">{% if results and results.original_text and results.analysis_type == 'text' %}{{ results.original_text }}{% endif %}</textarea>
                    <button type="submit" class="btn" id="textBtn">Metni Analiz Et</button>
                </form>
            </div>

            <div id="Link" class="tab-content">
                <form action="/" method="POST" id="linkForm">
                    <input type="hidden" name="analysis_type" value="link">
                    <input type="url" name="url_to_analyze" placeholder="https://example.com/haber-linki">
                    <button type="submit" class="btn" id="linkBtn" disabled>Analiz Et (Henüz Aktif Değil)</button>
                </form>
            </div>

            <div id="Fotograf" class="tab-content">
                <div class="sub-tab-container">
                    <button class="sub-tab-button" onclick="openSubTab(event, 'FotografMetni')">Fotoğraf Metni Analizi (OCR)</button>
                    <button class="sub-tab-button" onclick="openSubTab(event, 'FotografAI')">Yapay Zeka Tespiti</button>
                </div>

                <div id="FotografMetni" class="sub-tab-content">
                    <form action="/" method="POST" enctype="multipart/form-data" id="ocrForm">
                        <input type="hidden" name="analysis_type" value="photo_ocr">
                        <label for="photo-upload-ocr">Fotoğraftaki metinleri analiz etmek için yükleyin:</label>
                        <input type="file" id="photo-upload-ocr" name="photo_file_ocr" accept="image/png, image/jpeg, image/webp">
                        <button type="submit" class="btn" id="ocrBtn">Metni Analiz Et</button>
                    </form>
                </div>

                <div id="FotografAI" class="sub-tab-content">
                    <form action="/" method="POST" enctype="multipart/form-data" id="aiForm">
                        <input type="hidden" name="analysis_type" value="photo_ai">
                        <label for="photo-upload-ai">Fotoğrafın YZ ile üretilip üretilmediğini tespit etmek için yükleyin:</label>
                        <input type="file" id="photo-upload-ai" name="photo_file_ai" accept="image/png, image/jpeg, image/webp">
                        <button type="submit" class="btn" id="aiBtn">YZ Tespiti Yap</button>
                    </form>
                </div>
            </div>

            <div class="loading" id="loadingSection">
                <div class="loading-spinner"></div>
                <p>Analiz yapılıyor, lütfen bekleyin...</p>
            </div>

            {% if results %}
                <div class="results" id="resultsSection">
                    {% if results.error %}
                        <p class="error">{{ results.error }}</p>
                    
                    {% elif results.analysis_type == 'photo_ai' %}
                        <h3 style="text-align:center;">Yapay Zeka Tespit Sonucu</h3>
                        {% set color_class = 'score-yellow' %}
                        {% if results.css_class == 'log-minus' %}{% set color_class = 'score-red' %}
                        {% elif results.css_class == 'log-plus' %}{% set color_class = 'score-green' %}
                        {% elif results.css_class == 'log-info' %}{% set color_class = 'score-orange' %}
                        {% endif %}
                        
                        <div class="ai-label {{ color_class }}">{{ results.ai_label }}</div>
                        <ul class="analysis-log">
                            {% for item in results.log %}<li class="{{ item.type }}">{{ item.message | safe }}</li>{% endfor %}
                        </ul>

                    {% elif results.score is defined %}
                        <div class="score-container">
                            {% if results.score <= 33 %}
                                <div class="score score-red">{{ results.score }}</div>
                                <div class="score-label">❌ Düşük Güvenilirlik</div>
                            {% elif results.score <= 50 %}
                                <div class="score score-orange">{{ results.score }}</div>
                                <div class="score-label">⚠️ Orta Güvenilirlik - Şüpheli</div>
                            {% elif results.score <= 75 %}
                                <div class="score score-yellow">{{ results.score }}</div>
                                <div class="score-label">🔍 Orta-Güvenilir</div>
                            {% else %}
                                <div class="score score-green">{{ results.score }}</div>
                                <div class="score-label">✅ Yüksek Güvenilirlik</div>
                            {% endif %}
                        </div>
                        <h3>📊 Detaylı Analiz Raporu:</h3>
                        <ul class="analysis-log">
                            {% for item in results.log %}<li class="{{ item.type }}">{{ item.message | safe }}</li>{% endfor %}
                        </ul>
                    {% endif %}
                </div>
            {% endif %}
        </div> </div> <script>
        // --- Sekme Yönetimi JS (gerçeklik radarı.py'den) ---
        function openTab(evt, tabName) {
            var i, tabcontent, tabbuttons;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) tabcontent[i].style.display = "none";
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
            
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            
            if (tabName === 'Fotograf') {
                var subTabs = document.getElementById('Fotograf').getElementsByClassName('sub-tab-content');
                for (var j = 0; j < subTabs.length; j++) subTabs[j].style.display = "none";
                var subButtons = document.getElementById('Fotograf').getElementsByClassName('sub-tab-button');
                for (var j = 0; j < subButtons.length; j++) subButtons[j].className = subButtons[j].className.replace(" active", "");
                
                document.getElementById('FotografMetni').style.display = "block";
                document.querySelector('#Fotograf .sub-tab-button').className += " active";
            }
        }

        function openSubTab(evt, subTabName) {
            var i, subtabcontent, subtabbuttons;
            var parentTab = evt.currentTarget.closest('.tab-content');
            subtabcontent = parentTab.getElementsByClassName("sub-tab-content");
            for (i = 0; i < subtabcontent.length; i++) subtabcontent[i].style.display = "none";
            subtabbuttons = parentTab.getElementsByClassName("sub-tab-button");
            for (i = 0; i < subtabbuttons.length; i++) subtabbuttons[i].className = subtabbuttons[i].className.replace(" active", "");
            
            document.getElementById(subTabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        // --- Yüklenme Ekranı JS (main.py'den) ---
        function showLoadingSpinner(e) {
            const loading = document.getElementById('loadingSection');
            const results = document.getElementById('resultsSection');
            
            // Tüm butonları devre dışı bırak
            document.querySelectorAll('.btn').forEach(btn => {
                btn.disabled = true;
                btn.innerHTML = '⏳ Analiz Yapılıyor...';
            });
            
            loading.style.display = 'block';
            if (results) results.style.display = 'none';
        }
        
        // Event listener'ları tüm formlara ekle
        document.getElementById('textForm').addEventListener('submit', showLoadingSpinner);
        document.getElementById('ocrForm').addEventListener('submit', showLoadingSpinner);
        document.getElementById('aiForm').addEventListener('submit', showLoadingSpinner);
        
        // --- Sayfa Yükleme JS (Birleştirilmiş) ---
        document.addEventListener("DOMContentLoaded", function() {
             var defaultTabButton = document.querySelector('.tab-button');
             var activeTabName = 'Metin'; // Varsayılan

            {% if results and results.analysis_type %}
                var type = '{{ results.analysis_type }}';
                if (type === 'text') activeTabName = 'Metin';
                else if (type === 'link') activeTabName = 'Link';
                else if (type === 'photo_ocr' || type === 'photo_ai') activeTabName = 'Fotograf';
            {% endif %}
            
            var buttons = document.getElementsByClassName('tab-button');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.includes(activeTabName)) {
                    defaultTabButton = buttons[i];
                    break;
                }
            }
            defaultTabButton.click();

            {% if results and results.analysis_type %}
                var type = '{{ results.analysis_type }}';
                if (type === 'photo_ocr') {
                    document.querySelector('.sub-tab-button[onclick*="FotografMetni"]').click();
                } else if (type === 'photo_ai') {
                    document.querySelector('.sub-tab-button[onclick*="FotografAI"]').click();
                }
            {% endif %}
        });
    </script>
</body>
</html>
"""

# --- FLASK ROUTES (Birleştirilmiş) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type', 'text')
        
        try:
            if analysis_type == 'text':
                text_to_analyze = request.form.get('text_to_analyze')
                if not text_to_analyze or not text_to_analyze.strip():
                    raise ValueError("Lütfen analiz edilecek bir metin girin.")
                results = analyze_text(text_to_analyze)

            elif analysis_type == 'photo_ocr':
                results = analyze_photo_ocr(request.files)

            elif analysis_type == 'photo_ai':
                results = analyze_ai_generation(request.files)

            elif analysis_type == 'link':
                raise ValueError("Link analizi özelliği henüz aktif değildir.")
            
        except Exception as e:
            print(f"HATA OLUŞTU: {e}")
            results = {'error': str(e), 'analysis_type': analysis_type}
    
    return render_template_string(HTML_TEMPLATE, results=results)

@app.route('/health')
def health_check():
    """Sağlık kontrol endpoint'i (main.py'den)"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sentiment_model_loaded': model_manager.sentiment_model_loaded,
        'ai_model_loaded': model_manager.ai_model_loaded,
        'rss_cache_size': len(cache_manager.rss_cache),
        'analysis_cache_size': len(cache_manager.analysis_cache)
    }

# --- UYGULAMA BAŞLATMA (main.py'nin gelişmiş yöntemi) ---
def initialize_app():
    """Uygulamayı başlangıç ayarlarıyla başlat"""
    print("\n" + "="*50)
    print("🚀 Gelişmiş Dezenformasyon Analiz Aracı")
    print("="*50)
    
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    
    # Modelleri asenkron yükle
    model_manager.load_models()
    
    # Başlangıç RSS verisini önceden getir
    print("📰 RSS beslemeleri önceden yükleniyor...")
    try:
        fetch_all_rss_feeds()
    except Exception as e:
        print(f"⚠️  RSS yükleme hatası: {e}")
    
    print(f"✅ Uygulama hazır! http://127.0.0.1:{CONFIG.SERVER_PORT} adresinden erişebilirsiniz.")
    print("="*50 + "\n")

if __name__ == '__main__':
    initialize_app()
    app.run(
        debug=CONFIG.DEBUG, 
        port=CONFIG.SERVER_PORT,
        host='0.0.0.0' # 0.0.0.0 yerine 127.0.0.1 de kullanabilirsiniz
    )