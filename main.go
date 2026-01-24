package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	ServiceName = "dex-stt-service"
	ModelUrl    = "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin"
	RepoUrl     = "https://github.com/ggml-org/whisper.cpp.git"
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

	// Async setup
	go func() {
		if err := ensureAssets(); err != nil {
			log.Printf("Asset setup failed: %v", err)
		} else {
			mu.Lock()
			isReady = true
			mu.Unlock()
			log.Println("STT Assets ready.")
		}
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

	log.Printf("Starting Dexter STT Service (Whisper-Go) on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func ensureAssets() error {
	home, _ := os.UserHomeDir()
	binDir := filepath.Join(home, "Dexter", "bin")
	whisperBin := filepath.Join(binDir, "whisper-cli")
	modelsDir := filepath.Join(home, "Dexter", "models", "whisper")
	modelPath := filepath.Join(modelsDir, "ggml-medium-distil.bin")

	// 1. whisper-cli binary
	if _, err := os.Stat(whisperBin); os.IsNotExist(err) {
		log.Println("Whisper binary missing. Building from source via CMake...")
		if err := buildWhisper(binDir, whisperBin); err != nil {
			return fmt.Errorf("failed to build whisper: %w", err)
		}
	}

	// 2. Model
	_ = os.MkdirAll(modelsDir, 0755)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Println("Downloading Distil-Whisper GGML model...")
		if err := downloadFile(ModelUrl, modelPath); err != nil {
			return fmt.Errorf("failed to download model: %w", err)
		}
	}

	return nil
}

func buildWhisper(binDir, destBin string) error {
	tmpDir := "/tmp/whisper-build"
	_ = os.RemoveAll(tmpDir)
	_ = os.MkdirAll(tmpDir, 0755)
	defer func() { _ = os.RemoveAll(tmpDir) }()

	log.Println("Cloning whisper.cpp...")
	cloneCmd := exec.Command("git", "clone", "--depth", "1", RepoUrl, tmpDir)
	if err := cloneCmd.Run(); err != nil {
		return err
	}

	buildDir := filepath.Join(tmpDir, "build")
	_ = os.MkdirAll(buildDir, 0755)

	log.Println("Configuring whisper-cli with CMake (Standalone Strategy)...")
	// Attempt to force static build while ensuring examples are built
	cmakeArgs := []string{
		"..",
		"-DWHISPER_BUILD_EXAMPLES=ON",
		"-DBUILD_SHARED_LIBS=OFF",
		"-DGGML_STATIC=ON",
		"-DGGML_SHARED=OFF",
		"-DWHISPER_ALL_EXTERNAL=OFF",
		"-DCMAKE_BUILD_TYPE=Release",
	}

	if _, err := exec.LookPath("nvcc"); err == nil {
		log.Println("NVCC found, enabling CUDA support.")
		cmakeArgs = append(cmakeArgs, "-DWHISPER_CUDA=ON")
	}

	configCmd := exec.Command("cmake", cmakeArgs...)
	configCmd.Dir = buildDir
	configOutput, _ := configCmd.CombinedOutput()
	log.Printf("CMake Config Output:\n%s", string(configOutput))

	log.Println("Building whisper-cli...")
	buildCmd := exec.Command("cmake", "--build", ".", "--config", "Release", "--target", "whisper-cli", "-j", fmt.Sprintf("%d", runtime.NumCPU()))
	buildCmd.Dir = buildDir
	buildOutput, err := buildCmd.CombinedOutput()
	if err != nil {
		log.Printf("Build failed:\n%s", string(buildOutput))
		return fmt.Errorf("cmake build failed: %w", err)
	}

	// Move binary (use 'mv' to handle cross-device links)
	_ = os.MkdirAll(binDir, 0755)

	// Search for the binary in the build directory
	findBin := exec.Command("find", buildDir, "-name", "whisper-cli", "-type", "f")
	binPathBytes, _ := findBin.Output()
	sourceBin := strings.TrimSpace(string(binPathBytes))

	if sourceBin == "" {
		return fmt.Errorf("could not find whisper-cli binary in build directory")
	}

	log.Printf("Found whisper-cli at: %s", sourceBin)
	mvCmd := exec.Command("mv", sourceBin, destBin)
	if err := mvCmd.Run(); err != nil {
		return fmt.Errorf("failed to move binary to destination: %w", err)
	}

	// Robustly find and copy all shared libraries (just in case something was linked dynamically)
	// Even if we requested static, dependencies might have built as shared
	log.Println("Capturing any built shared libraries from the entire build tree...")
	findLibsCmd := fmt.Sprintf(`find %s -name "*.so*" -exec cp -Pd {} %s \;`, tmpDir, binDir)
	err = exec.Command("bash", "-c", findLibsCmd).Run()
	if err != nil {
		log.Printf("Warning: capture libs command reported error (may be fine if none exist): %v", err)
	}

	return nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer func() { _ = out.Close() }()
	_, err = io.Copy(out, resp.Body)
	return err
}

func handleTranscribe(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	ready := isReady
	mu.Unlock()

	if !ready {
		http.Error(w, "STT engine initializing", http.StatusServiceUnavailable)
		return
	}

	// Support both multipart and JSON (for redis_key/file_path)
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
	binDir := filepath.Join(home, "Dexter", "bin")
	whisperBin := filepath.Join(binDir, "whisper-cli")
	modelPath := filepath.Join(home, "Dexter", "models", "whisper", "ggml-medium-distil.bin")

	// whisper-cli -m <model> -f <file> -nt (no timestamps)
	cmd := exec.Command(whisperBin, "-m", modelPath, "-f", audioPath, "-nt")

	// CRITICAL: Ensure we check local bin for shared libraries
	cmd.Env = append(os.Environ(), fmt.Sprintf("LD_LIBRARY_PATH=%s:%s", binDir, os.Getenv("LD_LIBRARY_PATH")))

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		log.Printf("Whisper failed: %v, Stderr: %s", err, stderr.String())
		http.Error(w, "Transcription failed", http.StatusInternalServerError)
		return
	}

	text := strings.TrimSpace(out.String())

	resp := TranscribeResponse{
		Text:        text,
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
	mu.Lock()
	ready := isReady
	mu.Unlock()
	if !ready {
		http.Error(w, "initializing", http.StatusServiceUnavailable)
		return
	}
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
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(report)
}
