FROM ubuntu:14.04

RUN apt-get upgrade
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy scikit-learn matplotlib --ignore-installed six --user
RUN pip3 install openml==0.7.0 --ignore-installed six --user

# Download OpenML-100 statically
RUN python3 -c "import openml; [openml.tasks.get_task(task_id) for task_id in openml.study.get_study(14).tasks if task_id != 34536]"
RUN python3 -c "import openml; task = openml.tasks.get_task(145804); task.get_X_and_y()"

# Copy files (not cache any more)
RUN mkdir /root/project
COPY . /root/project
WORKDIR /root/project
RUN mkdir calc

