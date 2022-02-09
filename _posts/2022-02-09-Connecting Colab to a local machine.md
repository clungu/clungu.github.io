---
tags:
    - dl
mathjax: true
comments: true
title:  Connecting Colab to a local machine
header:
  teaser: /assets/images/2022-02-09-Connecting_Colab_to_a_local_machine_files/eng_colab_instance_types.png
---



20220209191322

---

![Image not found: ../Users/cristi/Documents/Zettelkasten/eng_colab_localhost.png](./eng_colab_localhost.png "Image not found: ./eng_colab_localhost.png")

Google Colab is a great offering (at least at the moment is) because it is free and allows you seamless access to both GPU instances and TPU instances that you can train the models on.

![eng_colab_instance_types.png](/assets/images/2022-02-09-Connecting_Colab_to_a_local_machine_files/eng_colab_instance_types.png)

The usual workflow (and recommended one) for developing in Colab is (also discussed in [[20220128192053]] Switch GPU to CPU using the aws-cli):
1. [CPU] prototype the ETL pipeline(data)
2. [CPU] develop and debug the model (make it work)
3. [GPU] train the model (fitting)
4. [CPU] debug the model (evaluate)
5. [GPU] optimise the model (performance)
6. [GPU] hyperparameter optimisations 
7. [CPU] analyse hyperparamter optimisation results
8. [multi GPU] run the optimised model on the full dataset

Because with Colab the change of instance type (CPU / GPU) is as quick as selecting an element of a dropdown, you can use the pattern above most of the time, but only for points 1.-3.

The problem with Colab is that it lacks persistence. Once you leave it unattended for more than 30 minutes (and depending on the instance type) it will timeout and disconnect from the remote Jupyter host which means losing that instance state (most of the time).  

This means that when you are at 3. you can see how the training progresses, that the model is working, and if you are lucky enough and the model isn't too demanding, see it finish. But after this point, you need to save the model and go back to the CPU. You can do this via the Google Drive addon, but the free space for that is limited (15 GB) and shared between all your Google apps (Docs, Spreadsheets, Drive, etc..).

Even if you manage to solve (or are not bothered by) the storage issue, there is still the nagging problem of runtime disconnects after a period of inactivity, and most of the steps after 3. are long-running activities. Not to mention that excess usage of GPU will temporarily ban you (1 day) from instantiating a new machine.

On the other hand, you could try with your own hosted server (say on AWS) and make it switch runtimes (CPU/GPU) almost as easily as in Colab (see [[20220128192053]] Switch GPU to CPU using the aws-cli for how you'd do this). But this means one of:
* [Colab -> AWS] downloading the notebook and importing it on the remote machine  
* [AWS only] working exclusively on the AWS instance

The first option causes needless friction and you also end up with two versions of the same work (the Colab one, and the post-Colab one that probably contains lots of fixes).

The second option maintains a single timeline of the work and is almost friction-less but you lose some Colab nice features:
* ability to share and comment on the notebook
* ability to see the notebook even without the instance running 
* nicer autocomplete 

## Best of both worlds

There is though a third possibility:
* Use Colab always (where you will store the notebooks)
* When closing in on steps 3.-8. switch to a private AWS server

You can do this because Colab allows you (for now) to connect to a local jupyter environment (with some quirks). This means that you can have the workflow below:  

1. [Google][CPU] prototype the ETL pipeline(data)
2. [Google][CPU] develop and debug the model (make it work)
3. [Google][GPU] train the model (fitting)
4. [Private][GPU] train the model (until finish)
5. [Private][CPU] debug the model (evaluate)
6. [Private][GPU] optimise the model (performance)
7. [Private][GPU] hyperparameter optimisations 
8. [Private][CPU] analyse hyperparamter optimisation results
9. [Private][multi GPU] run the optimised model on the full dataset

The setting for switching to a private server is in the top right corner:

![Image not found: ../Users/cristi/Documents/Zettelkasten/eng_colab_localhost.png](./eng_colab_localhost.png "Image not found: ./eng_colab_localhost.png")

To make it work you need to do the following steps (assuming you already have a private server with Jupyter installed on it):

* (One time) On your **server** install / enable the needed [Colaboratory extensions](https://github.com/googlecolab/jupyter_http_over_ws/)
```bash
pip install jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
```

* On your **server** start jupyter using the following command
```bash
jupyter notebook \
  --ip=0.0.0.0 \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --password='' \
  --NotebookApp.port_retries=0 \
  --NotebookApp.token='<some_long_token_here>'
```

* On your **laptop** forward port 8888 to the server port 8888. For some reason, choosing other port than 8888 on the local machine doesn't work with Colab, so YOU MUST use 8888
```bash
ssh -i ~/.ssh/key.pem ubuntu@10.10.110.100 -L 8888:localhost:8888 -N
```

* On your **Colab** page (opened on the laptop used to port forward 8888) choose the "Connect to a local runtime" from the dropdown menu of "Reconnect" and set the following address:

```html
http://localhost:8888/?token=<some_long_token_here>
```

That's it, Colab should connect to the private server. The setting (URL) for the remote will be persisted (to you) on the notebook but the people you share this with will not be able to see it. Couple this with [[20220128192053]] Switch GPU to CPU using the aws-cli and you might just have a clean and efficient way of developing [#dl](/tags/#dl) models!