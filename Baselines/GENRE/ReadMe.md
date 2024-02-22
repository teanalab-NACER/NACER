To Run the GENRE model on our Benchmark after download its repository:

1. Convert the QBLink dataset to the Kilt format using the file `filter_questions.py`
2. Convert the Kilt format to fairseq using `convert_kilt_to_fairseq.py`
3. Run `train.sh` to fine-tune the BART model
4. Evaluate the model using `evaluate_fine_tunned_QBlink.py`

The paper related to GENRE is:
De Cao, Nicola, et al. "Autoregressive entity retrieval." arXiv preprint arXiv:2010.00904 (2020).