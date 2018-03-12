---
categories: posts
---

# Register domain

# Create a GitHub Pagest repository

# Customise

## Install Jekyll theme

Jekyll, the open source project that powers GitHub Pages, introduced shared themes.  
Edit the `_config.yaml` file. If you don't have one create it. Then add the following line in it:

```
remote_theme: "mmistakes/so-simple-theme"
```

Save the file and commit on the GitHub repository.

## Troubleshooting

When hosting with GitHub Pages you'll need to push up a commit to force a rebuild with the latest theme release.

An empty commit will get the job done too if you don't have anything to push at the moment:

```
git commit --allow-empty -m "Force rebuild of site"
```

# Publishing a jupter notebook 

In order to publish a jupyer notebook to a blog you need to first convert it to either markdown or html. You do this by calling the following command:

```
jupyter nbconvert --to markdown
```

echo "[+] Interpreting the notebook path"
NOTEBOOK_FILE=$1
NOTEBOOK_BASE=`basename ${NOTEBOOK_FILE}`
NOTEBOOK_EXT="${NOTEBOOK_BASE##*.}"
NOTEBOOK_NAM="${NOTEBOOK_BASE%.*}"
echo "BLOG_PATH=${BLOG_PATH}"
echo "NOTEBOOK_FILE=${NOTEBOOK_FILE}"
echo "NOTEBOOK_BASE=${NOTEBOOK_BASE}"
echo "NOTEBOOK_EXT=${NOTEBOOK_EXT}"
echo "NOTEBOOK_NAM=${NOTEBOOK_NAM}"

echo "[+] Capturing the current file path" 
ORIG_DIR=`pwd`
echo "ORIG_DIR=${ORIG_DIR}"

echo "[+] Get the current day in the desired format"
TODAY=`date +"%Y-%m-%d"`
echo "TODAY=${TODAY}"

echo "[+] Copy the notebook into the blog directory"
cp "${NOTEBOOK_FILE}" "${BLOG_PATH}/_posts/${NOTEBOOK_NAM}.${NOTEBOOK_EXT}"

echo "[+] Switch to the blog directory"
cd "${BLOG_PATH}/_posts"

echo "[+] Generate the notebook"
workon deep; jupyter nbconvert "${NOTEBOOK_NAM}.${NOTEBOOK_EXT}" --to markdown --NbConvertApp.output_files_dir="../assets/images/{notebook_name}_files" --output "${TODAY}-${NOTEBOOK_NAM}.md"

echo "[+] Save the copied notebook to the _notebooks directory"
mv "${NOTEBOOK_NAM}.${NOTEBOOK_EXT}" "../_notebooks/${NOTEBOOK_NAM}.${NOTEBOOK_EXT}"

echo "[+] Switch back to the original directory"
cd "${ORIG_DIR}"