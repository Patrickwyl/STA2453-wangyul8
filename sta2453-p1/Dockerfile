FROM ubuntu:20.04 as shell_sta2453

RUN echo "STA2453 UBUNTU SHELL"
RUN apt-get update -y
RUN apt-get install curl software-properties-common vim -y

RUN apt-get install pip -y

WORKDIR /home/app_user/app
RUN pip install torch==1.8.1
RUN pip install scipy==1.6.3
RUN pip install matplotlib==3.3.2
RUN pip install numpy==1.21
RUN pip install pandas 
RUN echo 'alias python=python3' >> ~/.bashrc
# RUN echo "$TERM"
RUN echo 'export PS1="\[$(tput setaf 165)\](sta2453)\[$(tput setaf 171)\] \[$(tput setaf 219)\]\w\[$(tput sgr0)\] $: "' >> ~/.bashrc





