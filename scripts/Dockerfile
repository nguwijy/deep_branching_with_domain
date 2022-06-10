FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
        sudo \
        tzdata \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ARG username=nguwijy
ARG userid=1005

ARG home=/home/${username}
ARG workdir=${home}/deep_branching_with_domain

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
    && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
    && chmod 0440 /etc/sudoers.d/${username}

WORKDIR ${workdir}
RUN chown ${username}:${username} ${workdir}

USER ${username}
WORKDIR ${workdir}

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --user -r requirements.txt
RUN rm requirements.txt

ENV TZ="Asia/Singapore"
ENV CUDA_LAUNCH_BLOCKING=1
ENV PATH="/home/${username}/deep_branching_with_domain:/home/${username}/.local/bin:${PATH}"

CMD ["bash"]
