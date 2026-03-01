import { useState, useRef, useEffect } from 'react';
import * as ocr from '@paddlejs-models/ocr';
import { Camera, CheckCircle2, XCircle, Loader2 } from 'lucide-react';

const validRanges = [
  [67250001, 67700000],
  [69050001, 69500000],
  [69500001, 69950000],
  [69950001, 70400000],
  [70400001, 70850000],
  [70850001, 71300000],
  [76310012, 85139995],
  [86400001, 86850000],
  [90900001, 91350000],
  [91800001, 92250000],
  [87280145, 91646549],
  [96650001, 97100000],
  [99800001, 100250000],
  [100250001, 100700000],
  [109250001, 109700000],
  [110600001, 111050000],
  [111050001, 111500000],
  [111950001, 112400000],
  [112400001, 112850000],
  [112850001, 113300000],
  [114200001, 114650000],
  [114650001, 115100000],
  [115100001, 115550000],
  [118700001, 119150000],
  [119150001, 119600000],
  [120500001, 120950000],
  [77100001, 77550000],
  [78000001, 78450000],
  [78900001, 96350000],
  [96350001, 96800000],
  [96800001, 97250000],
  [98150001, 98600000],
  [104900001, 105350000],
  [105350001, 105800000],
  [106700001, 107150000],
  [107600001, 108050000],
  [108050001, 108500000],
  [109400001, 109850000],
];

export default function App() {
  const [isInitializing, setIsInitializing] = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<{ isValid: boolean; parsed: number }[] | null>(null);
  const [recognizedTexts, setRecognizedTexts] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const isOcrInitialized = useRef(false);

  useEffect(() => {
    if (isOcrInitialized.current) return;
    isOcrInitialized.current = true;
    
    async function initOcr() {
      const originalFetch = window.fetch;
      
      const ESTIMATED_TOTAL_BYTES = 3500000; 
      let loadedBytes = 0;

      window.fetch = async (...args) => {
        const reqStrOrObj = args[0];
        const init = args[1] as RequestInit | undefined;
        
        let isGet = true;
        if (reqStrOrObj instanceof Request) {
          isGet = reqStrOrObj.method.toUpperCase() === 'GET';
        }
        if (init && init.method) {
          isGet = init.method.toUpperCase() === 'GET';
        }

        if (isGet) {
          try {
            const cache = await caches.open('paddlejs-models-v1');
            const cachedResponse = await cache.match(reqStrOrObj as RequestInfo);
            if (cachedResponse) {
               return cachedResponse;
            }
          } catch (e) {
            console.warn("Error reading from cache:", e);
          }
        }

        const response = await originalFetch(...args);
        
        if (!response.body) return response;

        const responseClone = response.clone();
        
        if (isGet && response.ok) {
           const responseToCache = response.clone();
           caches.open('paddlejs-models-v1')
             .then(cache => cache.put(reqStrOrObj as RequestInfo, responseToCache))
             .catch(e => console.warn("Error saving to cache:", e));
        }

        const reader = responseClone.body!.getReader();

        const stream = new ReadableStream({
          async start(controller) {
            let doneReading = false;
            while (!doneReading) {
              const { done, value } = await reader.read();
              if (done) {
                doneReading = true;
                break;
              }
              
              if (value) {
                 loadedBytes += value.byteLength;
                 const progress = Math.min(99, Math.round((loadedBytes / ESTIMATED_TOTAL_BYTES) * 100));
                 setLoadProgress(progress);
              }
              controller.enqueue(value);
            }
            controller.close();
          }
        });

        return new Response(stream, {
          headers: response.headers,
          status: response.status,
          statusText: response.statusText,
        });
      };

      try {
        await ocr.init();
        setLoadProgress(100);
        setTimeout(() => setIsInitializing(false), 500);
      } catch (error) {
        console.error("Error al inicializar OCR:", error);
      } finally {
        window.fetch = originalFetch;
      }
    }
    initOcr();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImageSrc(url);
      setResults(null);
      setRecognizedTexts([]);
      processImage(url);
    }
  };

  const processImage = async (url: string) => {
    if (!url) return;
    setIsProcessing(true);
    
    const img = new Image();
    img.src = url;
    img.onload = async () => {
      try {
        const res = await ocr.recognize(img);
        if (res && res.text) {
          setRecognizedTexts(res.text);
          checkNumbers(res.text);
        } else {
          setRecognizedTexts([]);
          setResults(null);
        }
      } catch (err) {
        console.error("Error de OCR:", err);
      } finally {
        setIsProcessing(false);
      }
    };
  };

  const checkNumbers = (texts: string[]) => {
    const allText = texts.join(" ");
    const rawMatches = allText.match(/(?:\d\s*){8,9}/g);
    
    if (!rawMatches) {
        setResults([]);
        return;
    }

    const uniqueMatches = [...new Set(rawMatches.map(m => m.replace(/\s+/g, '')))];
    const newResults: { isValid: boolean; parsed: number }[] = [];

    for (const match of uniqueMatches) {
      const num = parseInt(match, 10);
      let isInvalid = false;
      for (const [min, max] of validRanges) {
        if (num >= min && num <= max) {
          isInvalid = true;
          break;
        }
      }
      newResults.push({ isValid: !isInvalid, parsed: num });
    }
    
    setResults(newResults);
  };

  return (
    <div className="container">
      <header className="glass-header">
        <h1>Escáner de Billetes</h1>
        <p>Escanea tus billetes con la cámara o súbelos.</p>
      </header>

      <main className="content">
        {isInitializing ? (
          <div className="status-box initializing">
            <Loader2 className="spinner" size={32} />
            <p>Descargando Modelo de IA<br/><small>(puede tardar un momento)</small></p>
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${loadProgress}%` }}></div>
            </div>
            <p className="progress-text">{loadProgress}%</p>
          </div>
        ) : (
           <div className="upload-container">
            <input 
              type="file" 
              accept="image/*" 
              capture="environment"
              onChange={handleFileChange} 
              ref={fileInputRef}
              className="hidden-input"
            />
            <button className="primary-btn" onClick={() => fileInputRef.current?.click()}>
              <Camera size={24} />
              <span>Capturar / Subir Billete</span>
            </button>
          </div>
        )}

        {imageSrc && (
          <div className="preview-card">
            <img src={imageSrc} alt="Billete escaneado" className="scanned-image" ref={imageRef} />
            
            {isProcessing && (
              <div className="processing-overlay">
                 <Loader2 className="spinner" size={48} />
                 <span>Escaneando números...</span>
              </div>
            )}
          </div>
        )}

        {!isProcessing && results !== null && results.length > 0 && (
          <div className="results-container" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%' }}>
            {results.map((result, index) => (
              <div key={index} className={`result-card ${result.isValid ? "valid" : "invalid"}`}>
                {result.isValid ? (
                  <>
                    <CheckCircle2 size={48} className="icon-valid" />
                    <h2>Billete Válido</h2>
                    <p>El número {result.parsed} es válido y no pertenece a los rangos reportados.</p>
                  </>
                ) : (
                  <>
                    <XCircle size={48} className="icon-invalid" />
                    <h2>Billete Inválido</h2>
                    <p>¡Cuidado! El número {result.parsed} pertenece a un lote reportado falso o robado.</p>
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {!isProcessing && results !== null && results.length === 0 && recognizedTexts.length > 0 && (
           <div className="result-card invalid">
              <XCircle size={48} className="icon-invalid" />
              <h2>Patrón no encontrado</h2>
              <p>No se detectó ningún número de 8 a 9 dígitos en la imagen.</p>
           </div>
        )}

      </main>
    </div>
  );
}
