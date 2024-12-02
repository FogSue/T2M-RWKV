1. Download the dataset from [this link](https://huggingface.co/datasets/r72snp/humanml3d/tree/main) and extract it into the `./dataset` directory.  
2. Run the three scripts under `./dataset/prepare/` (excluding `_model.sh`).  
3. Modify the `train_gpt.sh` script to configure the training parameters. The training code is located in `./train_t2m_trans.py`, and the model implementation is in `./models/t2m_trans.py`.  
