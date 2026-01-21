package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
)

// requirementsTxt content
const requirementsTxt = `fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart
faster-whisper>=1.0.0
pydantic>=2.6.0
redis
requests
psutil
`

// mainPy content
const mainPy = `import os
import sys
import logging
import time
import psutil
import json
import shutil
import contextlib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form, Request
from pydantic import BaseModel
import redis
from faster_whisper import WhisperModel, download_model

# Force standard streams to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("dex-stt-service")

START_TIME = time.time()

# Global variables
model = None
redis_client = None

# Constants
MODEL_SIZE = "large-v3-turbo"
# Dexter standard model path
MODEL_DIR = os.path.expanduser("~/Dexter/models/whisper/large-v3-turbo")

# Device Configuration
# Default to "cpu" to save VRAM, but allow override via env var
DEVICE = os.getenv("DEX_STT_DEVICE", "cpu")
logger.info(f"STT Service configured to use device: {DEVICE}")

def load_model():
    global model
    logger.info(f"Loading Whisper model ({MODEL_SIZE}) from {MODEL_DIR}...")
    
    # Configure compute type based on device
    if DEVICE == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8" # Best for CPU
    
    try:
        # Check if model exists locally
        if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
            logger.info("Model not found locally. Downloading...")
            # Use specific high-quality conversion
            model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
            try:
                download_model(model_id, output_dir=MODEL_DIR)
                logger.info("Model download complete.")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                # Try falling back to default large-v3 if custom fails
                logger.info("Attempting fallback to default large-v3...")
                download_model("large-v3", output_dir=MODEL_DIR)

        try:
            model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=compute_type)
            logger.info(f"Initialized Whisper on {DEVICE.upper()} with {compute_type} precision")
        except Exception as e:
            if DEVICE == "cuda":
                logger.warning(f"CUDA initialization failed ({e}), falling back to CPU...")
                model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")
                logger.info("Initialized Whisper on CPU (Fallback)")
            else:
                raise e
            
        logger.info("Whisper model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load Whisper model: {e}")
        # We don't exit here to allow the service to start and report health error
        model = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    try:
        redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)
        redis_client.ping()
        logger.info("Connected to Redis.")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        redis_client = None
        
    load_model()
    yield
    if redis_client:
        redis_client.close()
    logger.info("STT Service shutdown complete.")

app = FastAPI(title="Dexter STT Service", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "model": MODEL_SIZE}

@app.get("/service")
async def service_status():
    process = psutil.Process(os.getpid())
    uptime_seconds = time.time() - START_TIME
    
    m, s = divmod(uptime_seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    
    parts = []
    if d > 0: parts.append(f"{int(d)}d")
    if h > 0: parts.append(f"{int(h)}h")
    if m > 0: parts.append(f"{int(m)}m")
    parts.append(f"{float(s):.3f}s")
    uptime_str = "".join(parts)

    # Try Environment Variables (Injected by Go Wrapper)
    branch = os.getenv("DEX_BRANCH", "unknown")
    commit = os.getenv("DEX_COMMIT", "unknown")
    version_str = os.getenv("DEX_VERSION", "0.0.0")
    build_date = os.getenv("DEX_BUILD_DATE", "unknown")
    arch = os.getenv("DEX_ARCH", "unknown")
    build_hash = os.getenv("DEX_BUILD_HASH", "unknown")

    version_parts = version_str.split('.')
    major = version_parts[0] if len(version_parts) > 0 else "0"
    minor = version_parts[1] if len(version_parts) > 1 else "0"
    patch = version_parts[2] if len(version_parts) > 2 else "0"

    full_version_str = f"{version_str}.{branch}.{commit}.{build_date}.{arch}"

    return {
        "version": {
            "str": full_version_str,
            "obj": {
                "major": major,
                "minor": minor,
                "patch": patch,
                "branch": branch,
                "commit": commit,
                "build_date": build_date,
                "arch": arch,
                "build_hash": build_hash
            }
        },
        "health": {
            "status": "ok" if model is not None else "error",
            "uptime": uptime_str
        },
        "metrics": {
            "cpu": { "avg": process.cpu_percent(interval=None) },
            "memory": { "avg": process.memory_info().rss / 1024 / 1024 }
        }
    }

class TranscribeRequest(BaseModel):
    redis_key: Optional[str] = None
    file_path: Optional[str] = None

@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    global model
    if model is None:
        # Try reloading
        load_model()
        if model is None:
            raise HTTPException(status_code=503, detail="STT model not loaded")

    audio_source = None
    cleanup_path = None
    r_key = None
    f_path = None

    # 1. Try to extract params from various sources
    # Check JSON body first
    if request.headers.get("content-type") == "application/json":
        try:
            body = await request.json()
            r_key = body.get("redis_key")
            f_path = body.get("file_path")
        except:
            pass
    
    # Check Form data if not in JSON
    if not r_key and not f_path:
        try:
            form = await request.form()
            r_key = form.get("redis_key")
            f_path = form.get("file_path")
        except:
            pass

    try:
        # 2. Determine Source (Priority: Local Path > Upload > Redis)
        if f_path:
            # Validate path exists and is safe (basic check)
            if os.path.exists(f_path):
                audio_source = f_path
                # We do NOT cleanup external paths by default, assumes caller handles it or it's a temp file we own?
                # Actually, for this optimization, the producer creates it in tmp, consumer reads it.
                # Who deletes it? Usually the consumer if it's a hand-off.
                # Let's assume we clean it up to prevent disk fill-up since it's a temp hand-off.
                cleanup_path = f_path
            else:
                logger.warning(f"Provided file_path not found: {f_path}")
                # Fallthrough to other methods if valid
        
        if not audio_source and file:
            # Save to temp file
            temp_filename = f"upload_{int(time.time())}_{file.filename}"
            temp_path = os.path.join("/tmp", temp_filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            audio_source = temp_path
            cleanup_path = temp_path
            
        if not audio_source and r_key:
            if not redis_client:
                raise HTTPException(status_code=500, detail="Redis unavailable")
            
            audio_data = redis_client.get(r_key)
            if not audio_data:
                raise HTTPException(status_code=404, detail=f"Redis key not found: {r_key}")
            
            # Sanitize key for filename
            safe_key = "".join([c if c.isalnum() or c in "._-" else "_" for c in r_key])
            temp_path = os.path.join("/tmp", f"stt_{safe_key}.wav")
            with open(temp_path, "wb") as f:
                f.write(audio_data)
            audio_source = temp_path
            cleanup_path = temp_path
            
        if not audio_source:
            raise HTTPException(status_code=400, detail="No audio source provided (file_path, file, or redis_key)")

        # 3. Transcribe
        logger.info(f"Transcribing {audio_source}...")
        segments, info = model.transcribe(
            audio_source,
            beam_size=5,
            language="en", # Force English for now
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text_parts = []
        hallucinations = ["thank you", "thanks", "you", "bye", "thank you.", "thanks.", "you.", "bye."]

        for segment in segments:
            text_clean = segment.text.strip().lower()
            
            # Anti-Hallucination Logic
            is_hallucination = False
            if text_clean in hallucinations:
                is_hallucination = True
            if "thank you" in text_clean and len(text_clean) < 25:
                is_hallucination = True
            
            if is_hallucination and segment.avg_logprob < -0.25:
                continue
                
            text_parts.append(segment.text)

        full_text = "".join(text_parts).strip()
        logger.info(f"Transcription: {full_text}")

        return {"text": full_text, "language": info.language, "probability": info.language_probability}

    except HTTPException:
        # Re-raise HTTP exceptions to maintain status codes
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            os.remove(cleanup_path)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8202")) # STT on 8202
    
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    
    uvicorn.run(app, host=host, port=port, log_config=log_config)
`

var (
	version   = "0.0.0"
	branch    = "unknown"
	commit    = "unknown"
	buildDate = "unknown"
	buildYear = "unknown"
	buildHash = "unknown"
	arch      = "unknown"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "version" {
		v := version
		if v == "0.0.0" || v == "" {
			v = os.Getenv("DEX_VERSION")
		}
		if v == "" {
			v = "0.0.0"
		}
		b := branch
		if b == "unknown" || b == "" {
			b = os.Getenv("DEX_BRANCH")
		}
		c := commit
		if c == "unknown" || c == "" {
			c = os.Getenv("DEX_COMMIT")
		}
		fmt.Printf("%s.%s.%s.%s.%s.%s.%s\n", v, b, c, buildDate, buildYear, buildHash, arch)
		return
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("Failed to get user home directory: %v", err)
	}

	serviceDir := filepath.Join(homeDir, "Dexter", "services", "dex-stt-service")
	if err := os.MkdirAll(serviceDir, 0755); err != nil {
		log.Fatalf("Failed to create service directory: %v", err)
	}

	if err := os.WriteFile(filepath.Join(serviceDir, "requirements.txt"), []byte(requirementsTxt), 0644); err != nil {
		log.Fatalf("Failed to write requirements.txt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(serviceDir, "main.py"), []byte(mainPy), 0644); err != nil {
		log.Fatalf("Failed to write main.py: %v", err)
	}

	// Load options.json to get configuration
	device := "cpu" // Default
	optionsPath := filepath.Join(homeDir, "Dexter", "config", "options.json")
	if data, err := os.ReadFile(optionsPath); err == nil {
		var opts struct {
			Services map[string]map[string]interface{} `json:"services"`
		}
		if err := json.Unmarshal(data, &opts); err == nil {
			if svc, ok := opts.Services["stt"]; ok {
				if val, ok := svc["device"].(string); ok {
					device = val
				}
			}
		}
	}

	// Use shared Dexter Python 3.10 environment
	pythonEnvDir := filepath.Join(homeDir, "Dexter", "python3.10")
	pythonBin := filepath.Join(pythonEnvDir, "bin", "python")
	pipBin := filepath.Join(pythonEnvDir, "bin", "pip")

	// Ensure the shared environment exists
	if _, err := os.Stat(pythonBin); os.IsNotExist(err) {
		log.Fatalf("Shared Python 3.10 environment not found at %s. Run 'dex verify' or 'dex build' to fix.", pythonBin)
	}

	log.Println("Ensuring pip is up-to-date...")
	pipUpdateCmd := exec.Command(pipBin, "install", "--upgrade", "pip")
	_ = pipUpdateCmd.Run()

	log.Println("Installing dependencies into shared environment...")
	pipCmd := exec.Command(pipBin, "install", "-r", "requirements.txt")
	pipCmd.Dir = serviceDir
	pipCmd.Stdout = os.Stdout
	pipCmd.Stderr = os.Stderr
	if err := pipCmd.Run(); err != nil {
		log.Printf("Warning: Failed to install dependencies: %v", err)
	}

	log.Println("Starting Dexter STT Service...")

	pythonCmd := exec.Command(pythonBin, "main.py")
	pythonCmd.Dir = serviceDir

	v := version
	if v == "0.0.0" || v == "" {
		v = os.Getenv("DEX_VERSION")
	}

	b := branch
	if b == "unknown" || b == "" {
		b = os.Getenv("DEX_BRANCH")
	}

	c := commit
	if c == "unknown" || c == "" {
		c = os.Getenv("DEX_COMMIT")
	}

	pythonCmd.Env = append(os.Environ(),
		fmt.Sprintf("DEX_VERSION=%s", v),
		fmt.Sprintf("DEX_BRANCH=%s", b),
		fmt.Sprintf("DEX_COMMIT=%s", c),
		fmt.Sprintf("DEX_BUILD_DATE=%s", buildDate),
		fmt.Sprintf("DEX_BUILD_YEAR=%s", buildYear),
		fmt.Sprintf("DEX_ARCH=%s", arch),
		fmt.Sprintf("DEX_BUILD_HASH=%s", buildHash),
		fmt.Sprintf("DEX_STT_DEVICE=%s", device),
	)

	pythonCmd.Stdout = os.Stdout
	pythonCmd.Stderr = os.Stderr

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		_ = pythonCmd.Process.Signal(os.Interrupt)
	}()

	if err := pythonCmd.Run(); err != nil {
		log.Printf("Service exited with error: %v", err)
		os.Exit(1)
	}
}
