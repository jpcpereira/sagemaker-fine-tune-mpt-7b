# Clone llm-foundry package from MosaicML
# This is where the training script is hosted
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry

# Install required packages
pip install -e ".[gpu]"
pip install git+https://github.com/mosaicml/composer.git@dev

# Run training script with fine-tuning configuration
composer scripts/train/train.py /opt/ml/code/finetuning_config.yaml

ls /tmp/checkpoints
# Convert Composer checkpoint to HuggingFace model format
python scripts/inference/convert_composer_to_hf.py \
    --composer_path /tmp/checkpoints/latest-rank0.pt \
    --hf_output_path /opt/ml/model/hf_fine_tuned_model \
    --output_precision bf16

# Print content of model artifact directory
ls /opt/ml/model/
