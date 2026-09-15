# GitHub from two real accounts (L and Z) + deploy

The assignment wants **meaningful commits from both members**. GitHub’s contributor graph uses the **commit email**. That email must be the one on L’s GitHub account, or Z’s. Do this as two people, not one person pretending to be two.

Do **not** invent a fake second student. If L and Z are your real group, use their real GitHub accounts.

---

## 0. Each person, once

On GitHub.com, L and Z each:

1. Create/sign in to their own account.
2. Settings → Emails → copy the **noreply** address  
   `something@users.noreply.github.com`  
   (or a verified personal email).
3. Install [Git](https://git-scm.com/download/win) and [GitHub Desktop](https://desktop.github.com/) **or** use GitHub in the browser.

On **each person’s computer** (or each time that person sits at the shared PC):

```text
git config --global user.name "L Full Name"
git config --global user.email "L_NOREPLY@users.noreply.github.com"
```

Z uses Z’s name and Z’s email. Never leave the other person’s email in `git config` when you commit.

---

## 1. L creates the public repo

1. L signs in at https://github.com/new
2. Owner: **laibasan** (or zohaib-2548)
3. Name: `aml-assignment-01-solar` (any clear name)
4. Public
5. **Do not** add a README (the project already has one)
6. Create repository

Invite the other person: repo → **Settings → Collaborators → Add people**
→ `laibasan` and `zohaib-2548`. The other person accepts the invite.

---

## 2. First commit on this PC (only the person who is committing)

Open PowerShell in `E:\AML project`:

```text
git init -b main
git add .
git status
```

Check that `.venv`, `__pycache__`, and `.env` are **not** listed.

If **L** is making this first commit:

```text
git config user.name "L Full Name"
git config user.email "L_NOREPLY@users.noreply.github.com"
git commit -m "Add Plant 1 data pipeline, regression solvers, and results."
git remote add origin https://github.com/L_USERNAME/REPO_NAME.git
git push -u origin main
```

GitHub will ask L to log in (browser or Personal Access Token).

This first commit is large. That is fine. The **next** commits should be Z’s.

---

## 3. Z adds a real second commit (required)

Z should change files Z can explain in the viva, for example:

- `app/` (front end)
- `communication/medium_blog.md` and `communication/linkedin_post.md`
- README run instructions / screenshot
- a small comment or README “how to launch the app” tweak Z actually understands

On the PC, **switch identity to Z** before committing:

```text
git config user.name "Z Full Name"
git config user.email "Z_NOREPLY@users.noreply.github.com"
```

Then:

```text
git add app communication README.md
git commit -m "Add Flask predictor, blog draft, and run instructions."
git push origin main
```

GitHub login this time must be **Z**. If Windows cached L’s credentials:

- GitHub Desktop: File → Options → Accounts → Sign out, sign in as Z  
  **or**
- Windows Credential Manager → remove `github.com` credentials, then push again as Z.

On GitHub, open **Insights → Contributors**. You should see **two** people. Click a commit: the author must match L or Z.

Make **at least 3–4 commits each**, not one dump + one dummy line. Examples of later commits:

- L: `src/prepare.py` missing-value note, Table 1 wording  
- Z: frontend label text, screenshot in `results/figures/10_frontend.png`  
- L: `src/regression.py` cost-function comment  
- Z: LinkedIn draft URL placeholders  

Each person should be able to explain **their** commits.

---

## 4. Split that looks honest in a viva

Suggested ownership (adjust to what each of you actually did):

| Person | Own these and commit them |
| --- | --- |
| L | `src/load_data.py`, `prepare.py`, `regression.py`, `train_eval.py`, `results/tables`, `results/analysis.md` |
| Z | `src/eda.py`, `fetch_weather.py`, `app/`, `communication/`, README front-end section, screenshot |

Do not use `git commit --author="Z <fake>"` while logged in as L. Examiners can ask Z to walk through the diff.

---

## 5. Deploy the Flask app (Render)

The app already has `wsgi.py` and a `Procfile`. It serves saved weights; it does **not** retrain.

1. Push `main` to GitHub (both people’s commits already on `main`).
2. Go to https://render.com → Sign in with **GitHub** (L’s account is enough).
3. **New → Web Service →** select `L_USERNAME/REPO_NAME`.
4. Settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn wsgi:app --bind 0.0.0.0:$PORT`
   - **Instance:** Free
5. Deploy. Wait until the service is Live.
6. Open the `onrender.com` URL, enter hour `12`, shortwave `800`, temp `32`, cloud `20`, press Predict. You should see a kW value.
7. Put that URL in the README and in the Medium blog.

Alternative: [Railway](https://railway.app) or [PythonAnywhere](https://www.pythonanywhere.com) with the same `gunicorn wsgi:app` start command.

---

## 6. After deploy, finish the assignment extras

1. Paste the GitHub URL and the live app URL into `README.md`.
2. Publish `communication/medium_blog.md` on Medium; put the two URLs at the bottom.
3. Paste `communication/linkedin_post.md`, replace `[GITHUB_URL]` and `[MEDIUM_URL]`, post it.
4. Submit: repo link, Medium link, LinkedIn link.

---

## What will get you in trouble

- One student, two GitHub accounts, no second person who can viva the code  
- `--author` spoofing, or committing as Z without Z present  
- Empty “fix typo” commits only from the second account  

GitHub showing two contributors is not enough. The viva is.
