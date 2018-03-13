---
categories: posts
---

# Register domain

# Create a GitHub Pages repository

Let's asssume you already have a GitHub account. The only thing you need to do in order to create a GitHub Pages repository is to create a new one and name it `github_name.github.io`.

# Customise

## Install Jekyll theme

Jekyll, the open source project that powers GitHub Pages, introduced shared themes.  
Edit the `_config.yaml` file. If you don't have one create it. Then add the following line in it:

```
remote_theme: "mmistakes/so-simple-theme"
```

Save the file and commit on the GitHub repository.

Btw, there are tons of other themes published on GitHub that you can try.

## Troubleshooting

When hosting with GitHub Pages you'll need to push up a commit to force a rebuild with the latest theme release.

An empty commit will get the job done too if you don't have anything to push at the moment:

```
git commit --allow-empty -m "Force rebuild of site"
```

# Publishing a jupter notebook 

In order to publish a jupyer notebook to a GitHubPages blog you need to first convert it to either markdown or html. 

## Converting to markdown

You usually do this by calling the following command:

```
jupyter nbconvert --to markdown
```

Keep in mind that the usuall pattern for most themes is to store all the posts under the `_posts` directory.

## Images, assets and path consistency

Unfortunately the above doesn't work that easily because all the images and additional filetypes referenced by the notebook are saved in the calling directory under a `_file` suffix.

Also, depending on your choosen theme, assets are usually stored in specially named directory (mine is called `assets`). So in order to have the .md published in `_posts`, the images under `assets/image` and the paths keep the correct referencing, you need to issue the following command:

```
jupyter nbconvert ${NOTEBOOK_NAME} --to markdown --NbConvertApp.output_files_dir="../assets/images/{notebook_name}_files" --output ${NOTEBOOK_NAME}.md
``` 

## Other bells and wistles

Other conventions:

* a post should have `YYYY-MM-DD` prefix for it to be indexed and sorted by data.
* the notebook should be stored under `_notebooks` directory

In the end, because all the required steps were <tfoot></tfoot> labourious to do by hand each time, I've written the following bash function and added it to my `~/.bashrc`:

```
function publish() {
	echo "[+] Interpreting the notebook path"
	NOTEBOOK_FILE=$1
	NOTEBOOK_BASE=`basename ${NOTEBOOK_FILE}`
	NOTEBOOK_EXT="${NOTEBOOK_BASE##*.}"
	NOTEBOOK_NAM="${NOTEBOOK_BASE%.*}"

	echo "[+] Capturing the current file path" 
	ORIG_DIR=`pwd`
	echo "ORIG_DIR=${ORIG_DIR}"

	echo "[+] Get the current day in the desired format"
	TODAY=`date +"%Y-%m-%d"`

	echo "[+] Copy the notebook into the blog directory"
	cp "${NOTEBOOK_FILE}" "${BLOG_PATH}/_posts/${NOTEBOOK_NAM}.${NOTEBOOK_EXT}"

	echo "[+] Switch to the blog directory"
	cd "${BLOG_PATH}/_posts"

	echo "[+] Generate the notebook"
	jupyter nbconvert "${NOTEBOOK_NAM}.${NOTEBOOK_EXT}" --to markdown --NbConvertApp.output_files_dir="../assets/images/{notebook_name}_files" --output "${TODAY}-${NOTEBOOK_NAM}.md"

	echo "[+] Save the copied notebook to the _notebooks directory"
	mv "${NOTEBOOK_NAM}.${NOTEBOOK_EXT}" "../_notebooks/${NOTEBOOK_NAM}.${NOTEBOOK_EXT}"

	echo "[+] Switch back to the original directory"
	cd "${ORIG_DIR}"
}
```









