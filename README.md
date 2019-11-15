# query_intent
This repository contains code for the CIKM paper:

Manchanda, Saurav, Mohit Sharma, and George Karypis. "Intent Term Weighting in E-commerce Queries." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM, 2019.

and its extended version:

Manchanda, Saurav, Mohit Sharma, and George Karypis. "Intent term selection and refinement in e-commerce queries." arXiv preprint arXiv:1908.08564 (2019).

```@inproceedings{manchanda2019intent,
  title={Intent Term Weighting in E-commerce Queries},
  author={Manchanda, Saurav and Sharma, Mohit and Karypis, George},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2345--2348},
  year={2019},
  organization={ACM}
}
```

```@article{manchanda2019intent,
  title={Intent term selection and refinement in e-commerce queries},
  author={Manchanda, Saurav and Sharma, Mohit and Karypis, George},
  journal={arXiv preprint arXiv:1908.08564},
  year={2019}
}
```

Please contact Saurav (manch043@umn.edu) for any questions.

## Dependencies
The code is tested with Python 3.7.4 and PyTorch 1.2.0

## CONTEXTUAL TERM-WEIGHTING (CTW)
CTW is implemented in context_term_weighting.py. 
The program can be run by giving the command-line arguments as follows:
```usage: python context_term_weighting.py  --data_folder <data folder location> 
                                                --embedding_size <Size of the input word embeddings (300 by default)>
                                                --hidden_size_gru <Number of nodes in the hidden layer of GRU (Default 256)>
                                                --hidden_size_mlp <Number of nodes in the hidden layer of MLP (Default 10)>
                                                --dropout <Dropout (Default 0.25)> --num_epochs <Number of training epochs (Default 20)>
                                                --batch_size <Batch size (Default 512)> 
                                                --num_layers_gru <Number of layers in GRU (Default 2)>
                                                --num_layers_mlp <Number of layers in MLP (Default 2)>
                                                --learning_rate <Learning rate (Default 0.001)> --weight_decay <L2 regularization (Default 1e-5)>
                                                --use_cuda <Cuda device to use, negative for cpu (Default -1)> 
                                                --seed <Seed for initializations, (Default 0)> 
                                                --update_embed <Whether to train the embeddings (Default 1)>
                                                --pretrained <Whether to use pretrained embeddings (Default 1; vectors.txt file should be present for this option to work)>
                                                --max_grad_norm <Maximum norm of the gradient, for gradient clipping (Default 1.0)>
                                                --output_file <Path to the output file, to write importance weights>
```

