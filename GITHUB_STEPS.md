# Publish from your Mac

The package is ready for upload. It has not been published to GitHub.

1. Extract `pie-headlines-github-ready.zip`. Open the resulting `pie-headlines` folder in VS Code.
2. On GitHub, create an empty repository named `pie-headlines`. Choose its visibility. Do not initialize a README, license, or .gitignore; those choices can be made from the local project.
3. Open a terminal in the extracted folder and run the following commands one at a time:

```bash
git init -b main
git add .
git status --short
git commit -m "Add headline experiment prediction study and reproducible results"
```

4. Copy the repository's HTTPS URL from GitHub, then replace `YOUR_GITHUB_USERNAME` below:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/pie-headlines.git
git push -u origin main
```

Use your existing GitHub authentication or VS Code's GitHub sign-in. Do not paste access tokens into project files. If Git requests a name and email for the commit, configure your own identity; a GitHub-provided no-reply email is an option.

Suggested repository description:

> Predict headline experiment CTR differences with frozen MiniLM embeddings and ridge regression. PySpark preprocessing, grouped holdout evaluation, and paired cluster bootstrap uncertainty.

The small public-data derivatives are deliberately included with attribution. Raw data, intermediate Parquet, encoder weights, virtual environments, and SSH keys are excluded by .gitignore.

Once pushed, review the rendered README and notebook. The post draft is `JOURNAL_POST.md`; add the real repository URL before publishing it elsewhere. AWS does not need to be restarted.
