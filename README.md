# Fuzzy-Evolutionary ICU Arrhythmia Pipeline

A client-side web application for real-time ICU arrhythmia monitoring. It combines fuzzy logic with genetic algorithm optimization for robust ECG QRS detection and arrhythmia classification under severe noise and interference (baseline wander, motion artifact, EMG burst).

Everything runs directly in the browser with zero build steps or server dependencies.

---

## 🚀 Quick Start: How to Push to GitHub & Publish Live

### Step 1: Initialize & Commit (Already Done Locally)
If you haven't yet initialized Git, run:
```bash
git init -b main
git add .
git commit -m "Initial commit: ICU Arrhythmia Pipeline Web App"
```

### Step 2: Create a Repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Enter a repository name (for example: `icu-arrhythmia-pipeline`).
3. Leave it **Public** (required for free GitHub Pages).
4. **Do not** check "Add a README" or ".gitignore" (these already exist in this project).
5. Click **Create repository**.

### Step 3: Link & Push
Copy the commands shown on GitHub or run the following (replace `<your-username>` with your GitHub username):
```bash
git remote add origin https://github.com/<your-username>/icu-arrhythmia-pipeline.git
git branch -M main
git push -u origin main
```

---

## 🌐 How to Turn It Into a Live Website (GitHub Pages)

### Option A: Standard Branch Deployment (Easiest & Fastest)
1. On GitHub, navigate to your repository.
2. Click **Settings** (tab at the top) &rarr; **Pages** (in the left sidebar under *Code and automation*).
3. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`.
   - **Branch**: Select `main` and folder `/ (root)`.
4. Click **Save**.
5. Wait 1–2 minutes. Refresh the page to see your live URL:
   ```
   https://<your-username>.github.io/icu-arrhythmia-pipeline/
   ```

### Option B: GitHub Actions Workflow (Automatic)
This repository includes a ready-to-use GitHub Actions workflow located at `.github/workflows/deploy.yml`.
1. Go to **Settings** &rarr; **Pages**.
2. Under **Build and deployment** &rarr; **Source**, select **GitHub Actions**.
3. Every push to the `main` branch will automatically build and publish the site.

---

## ⚡ Alternative Free 1-Click Hosting

- **Vercel**: Import the GitHub repository at [vercel.com/new](https://vercel.com/new). No build configuration needed; click Deploy.
- **Netlify**: Drag-and-drop this entire folder directly at [app.netlify.com/drop](https://app.netlify.com/drop) or link via GitHub.

---

## 🧪 Testing Locally

You can test the site locally in any of the following ways:
- **Direct**: Double-click `index.html` to open it in your browser.
- **Local HTTP server** (Python):
  ```bash
  python -m http.server 8000
  ```
  Then open `http://localhost:8000` in your browser.
- **Node.js**:
  ```bash
  npx serve .
  ```

---

## 🌟 Application Features

1. **Live ICU Monitor**:
   - Real-time simulation of Normal Sinus Rhythm, Ventricular Tachycardia (VT), and Ventricular Fibrillation (VF).
   - Toggle baseline wander, motion artifacts, and severe EMG noise.
   - Live canvas showing raw vs. cleaned ECG with detected R-peaks.
   - Fuzzy logic inference engine combining heart rate, rhythm regularity (RR CV), and signal quality.

2. **RR-Feature Pipeline**:
   - Synthetic generator for Normal (N), Atrial Fibrillation (A), Other (O), Bradycardia, and Tachycardia.
   - Computes Heart Rate, R-peak count, Average RR, SDNN, RMSSD, NN50, and RR variation.

3. **ECG File Upload & Analysis**:
   - Drag & drop or upload custom CSV, TXT, or TSV ECG recordings.
   - Client-side parsing and interactive column and sampling rate selection (default 300 Hz).
   - Includes a sample test file: `sample_ecg.csv`.

4. **Evolution Lab (Genetic Algorithm)**:
   - Optimizes four detector parameters (baseline filter window, smoothing window, peak threshold, integration window).
   - Real-time generational fitness tracking with elitism.

5. **Automated Test Suite**:
   - 7 automated verification checks asserting sensitivity, precision, latency budgets, and buffer stability.

---

## 📁 Repository Structure

```
├── .github/
│   └── workflows/
│       └── deploy.yml      # Automated GitHub Pages CI/CD workflow
├── .gitignore              # Git ignore file
├── index.html              # Main website entry point (for web hosting & GitHub Pages)
├── FRONTEND.html           # Original frontend source file
├── sample_ecg.csv          # Sample ECG recording for upload testing
└── README.md               # Repository documentation and deployment guide
```
