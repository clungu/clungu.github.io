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

