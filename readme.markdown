# Pushing Your Project from VS Code to GitHub

Follow these steps to push your OCR project from VS Code to a new GitHub repository.

## Prerequisites
- **VS Code**: Installed with Git integration.
- **Git**: Installed and configured (`git --version` to check).
- **GitHub Account**: Logged in, with a personal access token if needed.
- **Files**: `fine_tune_trocr_gpu.py`, `Mansi_Dixit_OCR_Report.tex`, and `README.md`.

## Steps
1. **Prepare Your Project Folder**:
   - Create a folder (e.g., `handwritten-text-recognition`).
   - Add:
     - `fine_tune_trocr_gpu.py` (download from Kaggle or copy from artifact `f7b8c9d0`).
     - `Mansi_Dixit_OCR_Report.tex` (copy from artifact `e74f9be0-d40b-4f23-a76e-0fd72f9c24c4`).
     - `README.md` (copy from artifact `a9b0c1d2-e3f4-45a5-b6c7-h8i9j0k1l2m3`).
   - **Note**: Exclude the model (`fine_tuned_trocr_epoch_5`) due to size; note its location in README.

2. **Open in VS Code**:
   - Open VS Code.
   - File -> Open Folder -> Select `handwritten-text-recognition`.

3. **Initialize Git Repository**:
   - Open the terminal in VS Code (Terminal -> New Terminal).
   - Run:
     ```bash
     git init
     ```

4. **Add Files**:
   - Stage all files:
     ```bash
     git add .
     ```

5. **Commit Changes**:
   - Commit with a message:
     ```bash
     git commit -m "Initial commit: OCR project with TrOCR fine-tuning"
     ```

6. **Create GitHub Repository**:
   - Go to [GitHub](https://github.com).
   - Click “New” (top-right).
   - Name: `handwritten-text-recognition`.
   - Description: “Handwritten text recognition using TrOCR on IAM dataset.”
   - Public/Private: Choose Public for visibility.
   - **Do not** initialize with README (we already have one).
   - Click “Create repository”.

7. **Link to GitHub**:
   - Copy the repo URL (e.g., `https://github.com/your-username/handwritten-text-recognition.git`).
   - In VS Code terminal, link the remote:
     ```bash
     git remote add origin https://github.com/your-username/handwritten-text-recognition.git
     ```

8. **Push to GitHub**:
   - Push your commit:
     ```bash
     git push -u origin main
     ```
   - If prompted, enter your GitHub username and password (or personal access token).

9. **Verify on GitHub**:
   - Visit your repo on GitHub.
   - Ensure `fine_tune_trocr_gpu.py`, `Mansi_Dixit_OCR_Report.tex`, and `README.md` are uploaded.

10. **Troubleshooting**:
    - **Authentication Error**: Generate a personal access token in GitHub (Settings -> Developer settings -> Personal access tokens -> Generate new token) and use it instead of a password.
    - **Push Rejected**: Run `git pull origin main --rebase` to resolve conflicts, then push again.
    - **Git Not Found**: Install Git and ensure it’s in your PATH.