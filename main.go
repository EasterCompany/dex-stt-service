package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/EasterCompany/dex-stt-service/utils"
)

const (
	ServiceName = "dex-stt-service"
)

var (
	version   = "0.0.0"
	branch    = "unknown"
	commit    = "unknown"
	buildDate = "unknown"
	arch      = runtime.GOARCH
	startTime = time.Now()

	mu      sync.Mutex
	isReady = false
)

type TranscribeResponse struct {
	Text        string  `json:"text"`
	Language    string  `json:"language"`
	Probability float64 `json:"probability"`
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "version" {
		fmt.Printf("%s.%s.%s.%s.%s\n", version, branch, commit, buildDate, arch)
		os.Exit(0)
	}

	flag.Parse()

	// Async verification of assets (provisioned by dex-cli during build)
	go func() {
		home, _ := os.UserHomeDir()
		binDir := filepath.Join(home, "Dexter", "bin", "stt")
		sttBin := filepath.Join(binDir, "dex-net-stt")
		modelPath := filepath.Join(home, "Dexter", "models", "dex-net-stt.bin")

		for i := 0; i < 60; i++ {
			if _, err := os.Stat(sttBin); err == nil {
				if _, err := os.Stat(modelPath); err == nil {
					mu.Lock()
					isReady = true
					mu.Unlock()
					log.Println("STT Assets verified and ready.")
					return
				}
			}
			time.Sleep(2 * time.Second)
		}
		log.Println("Warning: STT Assets not found after 2 minutes. Service will remain in initializing state.")
	}()

	http.HandleFunc("/transcribe", handleTranscribe)
	http.HandleFunc("/hibernate", handleHibernate)
	http.HandleFunc("/wakeup", handleWakeup)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/service", handleService)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8202"
	}

	log.Printf("Starting Dexter STT Service (Neural STT Kernel) on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func handleTranscribe(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	ready := isReady
	mu.Unlock()

	if !ready {
		http.Error(w, "STT engine initializing (waiting for assets)", http.StatusServiceUnavailable)
		return
	}

	var audioPath string
	if strings.Contains(r.Header.Get("Content-Type"), "multipart/form-data") {
		err := r.ParseMultipartForm(10 << 20)
		if err != nil {
			http.Error(w, "Parse error", http.StatusBadRequest)
			return
		}
		filePath := r.FormValue("file_path")
		if filePath != "" {
			audioPath = filePath
		} else {
			http.Error(w, "file_path required", http.StatusBadRequest)
			return
		}
	}

	if audioPath == "" {
		http.Error(w, "No audio source", http.StatusBadRequest)
		return
	}

	home, _ := os.UserHomeDir()
	binDir := filepath.Join(home, "Dexter", "bin", "stt")
	sttBin := filepath.Join(binDir, "dex-net-stt")
	modelPath := filepath.Join(home, "Dexter", "models", "dex-net-stt.bin")

	cmd := exec.Command(sttBin, "-m", modelPath, "-f", audioPath, "-nt")

	// CRITICAL: Isolated library path for STT to avoid conflicts with llama-server
	cmd.Env = append(os.Environ(), fmt.Sprintf("LD_LIBRARY_PATH=%s:%s", binDir, os.Getenv("LD_LIBRARY_PATH")))

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		log.Printf("Neural STT Kernel failed: %v, Stderr: %s", err, stderr.String())
		http.Error(w, "Transcription failed", http.StatusInternalServerError)
		return
	}

	resp := TranscribeResponse{
		Text:        strings.TrimSpace(out.String()),
		Language:    "en",
		Probability: 1.0,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func handleHibernate(w http.ResponseWriter, r *http.Request) {
	_, _ = w.Write([]byte(`{"status":"ok","message":"process-idle"}`))
}

func handleWakeup(w http.ResponseWriter, r *http.Request) {
	_, _ = w.Write([]byte(`{"status":"ok","message":"ready"}`))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	_, _ = w.Write([]byte("OK"))
}

func handleService(w http.ResponseWriter, r *http.Request) {
	vParts := strings.Split(version, ".")
	major, minor, patch := "0", "0", "0"
	if len(vParts) >= 3 {
		major, minor, patch = vParts[0], vParts[1], vParts[2]
	}

	report := map[string]interface{}{
		"version": map[string]interface{}{
			"str": fmt.Sprintf("%s.%s.%s.%s.%s", version, branch, commit, buildDate, arch),
			"obj": map[string]interface{}{
				"major": major, "minor": minor, "patch": patch,
				"branch": branch, "commit": commit, "build_date": buildDate, "arch": arch,
			},
		},
		"health": map[string]interface{}{
			"status": "OK",
			"uptime": time.Since(startTime).String(),
		},
		"metrics": utils.GetMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(report)
}
