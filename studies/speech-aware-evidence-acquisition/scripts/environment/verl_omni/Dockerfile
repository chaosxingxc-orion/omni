# Preferred path: exact NGA image already loaded on the build host.
# No replacement of torch/CANN/vLLM/Transformers, no package resolution.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL omni.purpose="verl-omni-lora-environment-probe" omni.training_verified="false"
COPY omni-tools /opt/omni-verl
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
WORKDIR /opt/omni-verl
