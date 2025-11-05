# GitHub Pages Setup Guide

This guide will help you set up GitHub Pages for the vrAnalysis2 documentation.

## Step 1: Enable GitHub Pages in Repository Settings

**Note:** You can do this step before or after pushing the workflow. The `gh-pages` branch will be automatically created by the GitHub Actions workflow on its first run.

1. Go to your repository on GitHub: `https://github.com/landoskape/vrAnalysis`
2. Click on **Settings** (in the repository navigation bar)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Source**: `Deploy from a branch`
   - **Branch**: `gh-pages` (this branch will be created automatically by the workflow)
   - **Folder**: `/ (root)`
5. Click **Save**

## Step 2: Verify GitHub Actions Permissions

The GitHub Actions workflow needs write permissions to deploy:

1. In the same **Settings** page, go to **Actions** → **General**
2. Scroll to **Workflow permissions**
3. Select **Read and write permissions**
4. Check **Allow GitHub Actions to create and approve pull requests** (optional but recommended)
5. Click **Save**

## Step 3: Push the Workflow

The workflow file is already created at `.github/workflows/docs.yml`. Simply commit and push it:

```bash
git add .github/workflows/docs.yml mkdocs.yml
git commit -m "Add GitHub Actions workflow for documentation deployment"
git push origin main
```

## Step 4: Monitor the Deployment

1. Go to the **Actions** tab in your repository
2. You should see a workflow run called "Deploy Documentation"
3. Click on it to see the build progress
4. **Important:** The workflow will automatically create the `gh-pages` branch on its first run
5. Once it completes successfully:
   - The `gh-pages` branch will be created (if it doesn't exist)
   - Your documentation will be available at: `https://landoskape.github.io/vrAnalysis/`

## Troubleshooting

### Workflow Fails

If the workflow fails:
1. Check the **Actions** tab for error messages
2. Common issues:
   - Missing dependencies (should be handled by the workflow)
   - Python version issues (workflow uses Python 3.11)
   - Build errors (check mkdocs build locally first)

### Documentation Not Updating

- Make sure you pushed to the `main` branch
- The workflow only runs on pushes to `main` (or manual triggers)
- Wait a few minutes for GitHub Pages to update (can take 5-10 minutes)

### Custom Domain

If you want to use a custom domain:
1. Add a `CNAME` file to the `docs/` directory with your domain
2. Update the workflow file to set `cname: true`
3. Configure DNS settings as per GitHub Pages instructions

## Manual Deployment

You can also manually trigger the workflow:
1. Go to **Actions** tab
2. Select "Deploy Documentation" workflow
3. Click **Run workflow**
4. Select branch and click **Run workflow**

## Local Testing

Before pushing, test the build locally:

```bash
# Install dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Build the site
mkdocs build

# Serve locally to preview
mkdocs serve
```

