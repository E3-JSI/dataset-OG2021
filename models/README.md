# Models

The folder dedicated to storing model bin files.

## Naming Convention

Each filename must following the given naming convention:

```bash
# task   - the task the model was trained to solve
#          Possible tasks: NER, SSIM
# lmodel - The language model used to solve the task.
# tdata  - The short name of the dataset.
# ext    - The extension of the model file.
#          Possible options:
#            - pth: Pytorch extension.
#            - ckpt: Pytorch lightning extension.

{task}_{lmodel}_{tdata}.{ext}
```
